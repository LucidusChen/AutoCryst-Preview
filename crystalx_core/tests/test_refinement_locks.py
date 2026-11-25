"""Instruction effects are tested only on hand-authored synthetic models."""

from dataclasses import replace
from decimal import Decimal
import hashlib

import pytest

from crystalx_core.domain.hypothesis import LockDirective, LockMode
from crystalx_core.io.shelx_reader import logical_lines
from crystalx_core.io.shelx_res import read_res
from crystalx_core.refinement.locks import LockApplicationError, apply_locks


BASE = """TITL synthetic lock-adapter fixture
CELL 0.71073 10 11 12 90 90 90
LATT -1
SYMM -X,Y,-Z
SFAC C N
UNIT 2 1
L.S. 4
WGHT 0.1 0.2
FVAR 0.5 0.3
C1 1 0.125 -0.25 0.375 11.0 0.05
N1 2 0.4 0.5 0.6 -21.0 0.06
C2 1 -0.1 0.2 0.3 10.5 0.04
HKLF 4
END
"""


def hard(*atom_ids):
    return LockDirective(LockMode.HARD_LOCK, tuple(atom_ids), "synthetic independent review")


def soft(line="DFIX 1.8 0.04 C1 N1"):
    return LockDirective(LockMode.SOFT_RESTRAINT, ("C1", "N1"),
                         "synthetic distance hypothesis", instruction_lines=(line,))


def test_empty_lock_set_restores_exact_baseline_and_digest():
    text = BASE.replace("\n", "\r\n") + "\r\n"
    unlocked = apply_locks(text, {})
    assert unlocked.text == text
    assert unlocked.applied_ids == ()
    assert unlocked.baseline_sha256 == unlocked.output_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert unlocked.records == ()


def test_approved_hard_lock_changes_only_coordinate_fields_and_can_be_parsed(tmp_path):
    result = apply_locks(BASE, {"anchor": hard("C1")}, approved_hard_lock_ids=frozenset({"anchor"}))
    assert "C1 1 10.125 9.75 10.375 11.0 0.05\n" in result.text
    assert "N1 2 0.4 0.5 0.6 -21.0 0.06\n" in result.text
    for line in BASE.splitlines():
        if not line.startswith("C1 "):
            assert line in result.text.splitlines()
    target = tmp_path / "synthetic.ins"
    target.write_text(result.text)
    model = read_res(target)
    atom = next(atom for atom in model.atoms if atom.label == "C1")
    assert (Decimal(str(atom.x)) - 10, Decimal(str(atom.y)) - 10,
            Decimal(str(atom.z)) - 10) == (Decimal(".125"), Decimal("-.25"), Decimal(".375"))
    assert result.records[0].effect == "fixed_fractional_coordinates"
    assert result.records[0].hard_lock_review_approved is True
    assert apply_locks(BASE, {}).text == BASE


def test_hard_lock_has_no_implicit_approval_from_reason_or_metadata():
    directive = replace(hard("C1"), metadata={"approved": "true", "AI_probability": "1"})
    with pytest.raises(LockApplicationError, match="review approval"):
        apply_locks(BASE, {"anchor": directive})
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"anchor": directive}, approved_hard_lock_ids=frozenset({"C1"}))


def test_soft_lock_emits_real_dfix_before_hklf_and_preserves_model(tmp_path):
    result = apply_locks(BASE, {"distance": soft()})
    for line in BASE.splitlines():
        assert line in result.text.splitlines()
    assert result.text.index("DFIX 1.8 0.04 C1 N1") < result.text.index("HKLF 4")
    target = tmp_path / "synthetic.ins"
    target.write_text(result.text)
    restraints = [line for line in logical_lines(target) if line.instruction == "DFIX"]
    assert len(restraints) == 1
    assert tuple(restraints[0].tokens[1:]) == ("1.8", "0.04", "C1", "N1")
    assert result.records[0].instruction_lines == ("DFIX 1.8 0.04 C1 N1",)
    assert not result.records[0].hard_lock_review_approved


def test_distance_targets_can_supply_the_same_explicit_restraint():
    targets = LockDirective(LockMode.SOFT_RESTRAINT, ("C1", "N1"), "synthetic",
                           target_distances=(("C1", "N1", 1.8, 0.04),))
    assert apply_locks(BASE, {"distance": targets}).text == apply_locks(BASE, {"distance": soft()}).text
    both = replace(targets, instruction_lines=("DFIX 1.800 0.040 C1 N1",))
    assert apply_locks(BASE, {"distance": both}).text == apply_locks(BASE, {"distance": targets}).text
    with pytest.raises(LockApplicationError, match="disagree"):
        apply_locks(BASE, {"distance": replace(targets, instruction_lines=("DFIX 1.9 0.04 C1 N1",))})


@pytest.mark.parametrize("mode", [LockMode.EXPLORATORY, LockMode.FREE])
def test_exploratory_or_free_adds_no_fix_and_preserves_occupancy_links(mode):
    result = apply_locks(BASE, {"candidate": LockDirective(mode, ("N1",), "synthetic no new fixing")})
    for line in BASE.splitlines():
        assert line in result.text.splitlines()
    assert "N1 2 0.4 0.5 0.6 -21.0 0.06" in result.text
    assert "DFIX" not in result.text
    assert result.records[0].effect == "no_new_constraint"


def test_deterministic_fresh_rebuild_and_branch_rollback():
    accepted = {"anchor": hard("C2")}
    trial = {"new-distance": soft(), **accepted}
    approvals = frozenset({"anchor"})
    previous = apply_locks(BASE, accepted, approved_hard_lock_ids=approvals)
    changed = apply_locks(BASE, trial, approved_hard_lock_ids=approvals)
    reversed_order = apply_locks(BASE, dict(reversed(tuple(trial.items()))), approved_hard_lock_ids=approvals)
    assert changed == reversed_order
    assert previous.text != changed.text
    assert apply_locks(BASE, accepted, approved_hard_lock_ids=approvals) == previous
    assert apply_locks(BASE, {}).text == BASE
    with pytest.raises(LockApplicationError, match="original baseline"):
        apply_locks(changed.text, accepted, approved_hard_lock_ids=approvals)


@pytest.mark.parametrize("line", [
    "AFIX 1", "SFAC C N O", "UNIT 1 1", "OMIT 0 0 1", "HKLF 3", "L.S. 99",
    "EXYZ C1 N1", "DFIX 1.8 C1 N1", "DFIX 1.8 0.04 C1 N1 C2 N1",
    "DFIX 1.8 0.04 C1 N1\nOMIT 1 1 1", "DFIX 1.8 0.04 C1 N1 ! extra",
    "DFIX 1.8 0.04 C1 N1 =", "DFIX 1.8 0.04 C1 N1;UNIT 0 0",
    " DFIX 1.8 0.04 C1 N1", "DFIX 1.8 0.04 C1_$1 N1", "DFIX 1.8 0.04 $C N1",
    "DFIX 1.8 0.04 C1_3 N1", "DFIX 1.8 0.04 C1 > N1",
])
def test_unknown_or_overbroad_restraint_instructions_fail_closed(line):
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"unsafe": soft(line)})


@pytest.mark.parametrize("value", ["0", "-1", "15", "21", "nan", "inf", "1e99"])
def test_invalid_distance_codes_block(value):
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"invalid": soft(f"DFIX {value} 0.04 C1 N1")})


@pytest.mark.parametrize("value", ["0", "-0.04", "nan", "inf", "1e99", "1e-99"])
def test_invalid_esds_block(value):
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"invalid": soft(f"DFIX 1.8 {value} C1 N1")})


def test_duplicate_pairs_do_not_silently_double_weight():
    with pytest.raises(LockApplicationError, match="Duplicate DFIX"):
        apply_locks(BASE, {"a": soft(), "b": soft("DFIX 1.8 0.04 N1 C1")})
    with pytest.raises(LockApplicationError, match="Duplicate DFIX"):
        apply_locks(BASE.replace("HKLF", "DFIX 1.8 0.04 C1 N1\nHKLF"), {"new": soft()})


def test_restraint_cannot_affect_unnamed_or_missing_sites():
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"a": replace(soft(), atom_ids=("C1",))})
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"a": soft("DFIX 1.8 0.04 C1 O1")})
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"a": soft("DFIX 1.8 0.04 C1 C1")})
    with pytest.raises(LockApplicationError, match="actual emitted"):
        apply_locks(BASE, {"a": replace(soft(), atom_ids=("C1", "N1", "C2"))})


@pytest.mark.parametrize("prefix,suffix", [
    ("AFIX 1\n", "AFIX 0\n"), ("PART 1\n", "PART 0\n"),
    ("RESI 1 TEST\n", "RESI 0\n"), ("EXYZ C1 N1\n", ""),
])
def test_targets_in_unaudited_coupling_or_identity_context_are_blocked(prefix, suffix):
    text = BASE.replace("C1 1", prefix + "C1 1").replace("HKLF 4", suffix + "HKLF 4")
    with pytest.raises(LockApplicationError):
        apply_locks(text, {"a": hard("C1")}, approved_hard_lock_ids=frozenset({"a"}))
    with pytest.raises(LockApplicationError):
        apply_locks(text, {"b": soft()})


def test_dfix_insertion_needs_neutral_final_residue_context():
    text = BASE.replace("C2 1", "RESI 3 TEST\nC2 1")
    with pytest.raises(LockApplicationError, match="neutral"):
        apply_locks(text, {"distance": soft()})


def test_ambiguous_atom_label_case_differences_still_block():
    text = BASE.replace("HKLF 4", "c1 1 0.2 0.3 0.4 11 0.05\nHKLF 4")
    with pytest.raises(LockApplicationError, match="ambiguous"):
        apply_locks(text, {"a": hard("C1")}, approved_hard_lock_ids=frozenset({"a"}))


def test_same_atom_in_two_reviewed_locks_is_not_double_encoded():
    result = apply_locks(BASE, {"a": hard("C1"), "b": hard("C1")},
                         approved_hard_lock_ids=frozenset({"a", "b"}))
    assert "C1 1 10.125 9.75 10.375 11.0 0.05" in result.text
    assert result.applied_ids == ("a", "b")


@pytest.mark.parametrize("value", ["21", "-21", "15", "5", "nan"])
def test_coordinate_fvar_or_ambiguous_encoding_is_not_reinterpreted(value):
    text = BASE.replace("C1 1 0.125", f"C1 1 {value}")
    with pytest.raises(LockApplicationError):
        apply_locks(text, {"a": hard("C1")}, approved_hard_lock_ids=frozenset({"a"}))


def test_existing_fixed_coordinates_are_not_fixed_twice():
    text = BASE.replace("C1 1 0.125 -0.25 0.375", "C1 1 10.125 9.75 10.375")
    result = apply_locks(text, {"a": hard("C1")}, approved_hard_lock_ids=frozenset({"a"}))
    assert "C1 1 10.125 9.75 10.375 11.0 0.05" in result.text


def test_exploratory_and_hard_on_same_atom_are_contradictory():
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"a": hard("C1"), "b": LockDirective(LockMode.EXPLORATORY, ("C1",), "synthetic")},
                    approved_hard_lock_ids=frozenset({"a"}))


def test_nonreversible_directives_and_raw_hard_instructions_block():
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"a": replace(hard("C1"), reversible=False)}, approved_hard_lock_ids=frozenset({"a"}))
    with pytest.raises(LockApplicationError):
        apply_locks(BASE, {"a": replace(hard("C1"), instruction_lines=("AFIX 1",))},
                    approved_hard_lock_ids=frozenset({"a"}))


def test_anisotropic_continuation_and_all_other_fields_survive():
    text = BASE.replace("C1 1 0.125 -0.25 0.375 11.0 0.05",
                        "C1 1 0.125 -0.25 0.375 21.0 0.04 0.05 0.06 =\n  0.001 0.002 0.003")
    result = apply_locks(text, {"a": hard("C1")}, approved_hard_lock_ids=frozenset({"a"}))
    assert "C1 1 10.125 9.75 10.375 21.0 0.04 0.05 0.06 =\n  0.001 0.002 0.003" in result.text


def test_res_output_comments_with_equals_and_q_peaks_are_retained():
    text = BASE + "REM synthetic diagnostic = not a continuation\nREM next diagnostic = retained\nQ1 1 0.1 0.2 0.3 11 0.05 1\n"
    result = apply_locks(text, {"a": soft()})
    assert result.text.endswith(text[len(BASE):])
    with pytest.raises(LockApplicationError):
        apply_locks(text, {"q": hard("Q1")}, approved_hard_lock_ids=frozenset({"q"}))


def test_rem_before_atoms_does_not_turn_following_atom_into_continuation():
    text = BASE.replace("C1 1", "REM synthetic equals = just a comment\nC1 1")
    assert "DFIX 1.8 0.04 C1 N1" in apply_locks(text, {"a": soft()}).text


@pytest.mark.parametrize("instruction", ["DFIX 1.8 0.04 C1 =\n N1", "DFIX_* 1.8 0.04 C1 N1", "EXYZ $C $N"])
def test_unaudited_existing_selectors_cannot_hide_duplicate_or_ineffective_constraints(instruction):
    text = BASE.replace("HKLF", instruction + "\nHKLF")
    with pytest.raises(LockApplicationError):
        apply_locks(text, {"a": soft()})


def test_reason_cannot_inject_an_instruction_into_output():
    result = apply_locks(BASE, {"a": replace(soft(), reason="synthetic\nOMIT 1 1 1")})
    assert "OMIT" not in result.text
    assert len(result.records[0].reason_sha256) == 64


@pytest.mark.parametrize("change", [
    lambda text: text.replace("HKLF 4", "HKLF 4\nHKLF 4"),
    lambda text: text.replace("HKLF 4\n", ""),
    lambda text: text.replace("C1 1", "+hidden.ins\nC1 1"),
    lambda text: text.replace("C1 1", "MOVE 0 0 0 1\nC1 1"),
])
def test_baseline_structure_escapes_are_blocked(change):
    with pytest.raises(LockApplicationError):
        apply_locks(change(BASE), {"a": soft()})
