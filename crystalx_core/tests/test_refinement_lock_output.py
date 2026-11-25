"""A narrow, hand-authored managed-output dialect; no actual SHELXL is run."""

from dataclasses import replace
import hashlib

import pytest

from crystalx_core.domain.hypothesis import LockDirective, LockMode
from crystalx_core.refinement.locks import LockApplicationError, apply_locks, normalize_managed_output
from crystalx_core.workflow.reconstruction import read_snapshot


BASE = """TITL invented managed-output fixture
CELL 0.7 10 11 12 90 90 90
LATT -1
SFAC C N O
UNIT 2 1 0
WGHT 0.1 0.2
DFIX 1.4 0.05 N1 C2
FVAR 0.5
C1 1 0.125 -0.25 0.375 11.0 0.05
N1 2 0.4 0.5 0.6 11.0 0.06
C2 1 -0.1 0.2 0.3 10.5 0.04
HKLF 4
END
"""


def directives():
    return {
        "anchor": LockDirective(LockMode.HARD_LOCK, ("C1",), "invented external review"),
        "distance": LockDirective(LockMode.SOFT_RESTRAINT, ("C1", "N1"), "invented distance hypothesis",
                                  instruction_lines=("DFIX 1.8 0.04 C1 N1",)),
    }


APPROVED = frozenset({"anchor"})


def normalize(output=None, *, baseline=BASE, active=None, application=None, approved=APPROVED):
    active = directives() if active is None else active
    application = application or apply_locks(baseline, active, approved_hard_lock_ids=approved)
    return normalize_managed_output(application.text if output is None else output, application,
        baseline_ins_text=baseline, active_locks=active, approved_hard_lock_ids=approved)


def test_unchanged_controlled_output_restores_exact_base_and_pins():
    active = directives()
    application = apply_locks(BASE, active, approved_hard_lock_ids=APPROVED)
    result = normalize(application=application)
    assert result.text == BASE
    assert result.raw_output_sha256 == application.output_sha256
    assert result.lock_free_sha256 == result.baseline_sha256 == hashlib.sha256(BASE.encode()).hexdigest()
    assert result.application_sha256 == application.output_sha256
    assert result.removed_lock_ids == ("anchor", "distance")
    assert result.decoded_atom_ids == ("C1",)
    assert "DFIX 1.4 0.05 N1 C2" in result.text
    assert "DFIX 1.8 0.04 C1 N1" not in result.text


def test_free_site_refinement_and_appended_diagnostics_peaks_survive_and_parse(tmp_path):
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    text = application.text.replace("N1 2 0.4 0.5 0.6 11.0 0.06", "N1 2 0.41 0.51 0.61 11.0 0.07")
    text = text.replace("FVAR 0.5", "FVAR 0.55")
    tail = "REM R1 = 0.1234 for 20 Fo > 4sig(Fo) and 0.1357 for all 25 data\nQ1 1 0.3 0.4 0.5 11 0.05 1.2\n"
    result = normalize(text + tail, application=application)
    assert result.text.endswith(tail)
    assert "N1 2 0.41 0.51 0.61 11.0 0.07" in result.text
    assert "FVAR 0.55" in result.text
    assert "C1 1 0.125 -0.25 0.375 11.0 0.05" in result.text
    path = tmp_path / "invented-lock-free.res"
    path.write_text(result.text)
    snapshot = read_snapshot(path, reflection_sha256="a" * 64)
    assert snapshot.sites[0].fractional == (.125, -.25, .375)
    assert snapshot.sites[1].fractional == (.41, .51, .61)


def test_crlf_output_and_exact_baseline_reversal():
    baseline = BASE.replace("\n", "\r\n")
    result = normalize(baseline=baseline)
    assert result.text == baseline


@pytest.mark.parametrize("edit", [
    lambda text: text.replace("REM CrystalX active constraints BEGIN\n", ""),
    lambda text: text.replace("REM CrystalX active constraints END\n", ""),
    lambda text: text.replace("REM CrystalX lock anchor hard_lock\n", ""),
    lambda text: text.replace("REM CrystalX active constraints BEGIN", "REM CRYSTALX ACTIVE CONSTRAINTS BEGIN"),
    lambda text: text.replace("DFIX 1.8 0.04 C1 N1", " DFIX 1.8 0.04 C1 N1"),
    lambda text: text.replace("DFIX 1.8 0.04 C1 N1", "DFIX 1.800 0.04 C1 N1"),
    lambda text: text.replace("DFIX 1.8 0.04 C1 N1", "DFIX 1.9 0.04 C1 N1"),
    lambda text: text.replace("DFIX 1.8 0.04 C1 N1\n", ""),
    lambda text: text.replace("REM CrystalX active constraints END", "DFIX 2.0 0.2 C1 C2\nREM CrystalX active constraints END"),
    lambda text: text + "REM CrystalX lock unknown hard_lock\n",
    lambda text: text + "REM CrystalX active constraints BEGIN\n",
])
def test_missing_reformatted_or_extra_managed_effects_are_not_guessed(edit):
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    with pytest.raises(LockApplicationError):
        normalize(edit(application.text), application=application)


def test_moved_managed_block_is_an_unsupported_dialect():
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    text = application.text
    first = text.index("REM CrystalX active constraints BEGIN")
    last = text.index("REM CrystalX active constraints END") + len("REM CrystalX active constraints END\n")
    block = text[first:last]
    moved = text[:first] + text[last:]
    moved = moved.replace("C1 1", block + "C1 1")
    with pytest.raises(LockApplicationError):
        normalize(moved, application=application)


@pytest.mark.parametrize("edit", [
    lambda text: text.replace("10.125", "10.126"),
    lambda text: text.replace("10.125", "0.125"),
    lambda text: text.replace("9.75", "-0.25"),
    lambda text: text.replace("10.375", "21.0"),
    lambda text: text.replace("N1 2 0.4", "N1 2 10.4"),
])
def test_fixed_coordinates_cannot_drift_disappear_or_spread_to_free_sites(edit):
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    with pytest.raises(LockApplicationError):
        normalize(edit(application.text), application=application)


def test_equivalent_fixed_numeric_format_is_allowed_without_coordinate_tolerance():
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    result = normalize(application.text.replace("10.125", "10.125000"), application=application)
    assert result.text == BASE


@pytest.mark.parametrize("edit", [
    lambda text: text.replace("WGHT 0.1 0.2", "WGHT 0.11 0.2"),
    lambda text: text.replace("UNIT 2 1 0", "UNIT 3 1 0"),
    lambda text: text.replace("SFAC C N O", "SFAC C O N"),
    lambda text: text.replace("HKLF 4", "HKLF 3"),
    lambda text: text.replace("DFIX 1.4 0.05 N1 C2\n", ""),
    lambda text: text.replace("DFIX 1.4 0.05 N1 C2", "DFIX 1.5 0.05 N1 C2"),
    lambda text: text.replace("C1 1", "C1 2"),
    lambda text: text.replace("C2 1", "O2 1"),
    lambda text: text.replace("HKLF 4", "DFIX 2.2 0.1 C1 C2\nHKLF 4"),
    lambda text: text + "DFIX 2.2 0.1 C1 C2\n",
    lambda text: text.replace("N1 2 0.4 0.5 0.6 11.0 0.06", "N1 2 0.4 0.5 0.6 1.0 0.06"),
    lambda text: text.replace("N1 2 0.4 0.5 0.6 11.0 0.06", "N1 2 0.4 0.5 0.6 11.0 10.06"),
])
def test_existing_scientific_instructions_elements_and_freedom_are_preserved(edit):
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    with pytest.raises(LockApplicationError):
        normalize(edit(application.text), application=application)


def test_atom_reordering_or_loss_is_not_silently_normalized():
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    row = "N1 2 0.4 0.5 0.6 11.0 0.06\n"
    moved = application.text.replace(row, "").replace("C1 1", row + "C1 1")
    with pytest.raises(LockApplicationError):
        normalize(moved, application=application)
    with pytest.raises(LockApplicationError):
        normalize(application.text.replace(row, ""), application=application)


def test_records_and_baseline_pins_must_reproduce_from_trusted_ledger():
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    for forged in (replace(application, output_sha256="f" * 64),
                   replace(application, baseline_sha256="f" * 64),
                   replace(application, records=application.records[1:])):
        with pytest.raises(LockApplicationError):
            normalize(application.text, application=forged)
    with pytest.raises(LockApplicationError):
        normalize(application.text, application=application, baseline=BASE.replace("0.125", "0.126"))
    with pytest.raises(LockApplicationError):
        normalize(application.text, application=application, approved=frozenset())
    with pytest.raises(LockApplicationError):
        normalize(application.text, application=application, active={"distance": directives()["distance"]})


def test_no_active_locks_preserves_real_free_site_refinement_but_not_unknown_constraints():
    application = apply_locks(BASE, {})
    text = BASE.replace("N1 2 0.4", "N1 2 0.41")
    result = normalize(text, active={}, application=application)
    assert result.text == text
    assert result.removed_lock_ids == result.decoded_atom_ids == ()
    with pytest.raises(LockApplicationError):
        normalize(text.replace("N1 2 0.41", "N1 2 10.41"), active={}, application=application)


def test_preexisting_fixed_coordinates_are_never_unfixed_by_managed_cleanup():
    baseline = BASE.replace("C1 1 0.125", "C1 1 10.125")
    result = normalize(baseline=baseline)
    assert "C1 1 10.125 -0.25 0.375" in result.text
    assert result.text == baseline


def test_soft_only_reversal_keeps_preexisting_restraints_and_free_values():
    active = {"distance": directives()["distance"]}
    application = apply_locks(BASE, active)
    output = application.text.replace("C1 1 0.125", "C1 1 0.13")
    result = normalize(output, active=active, application=application)
    assert "C1 1 0.13" in result.text
    assert result.decoded_atom_ids == ()
    assert "DFIX 1.4 0.05 N1 C2" in result.text


def test_exploratory_audit_only_block_can_be_removed_without_touching_atom_values():
    active = {"explore": LockDirective(LockMode.EXPLORATORY, ("C1",), "invented exploratory site")}
    application = apply_locks(BASE, active)
    assert normalize(active=active, application=application).text == BASE


def test_anisotropic_continuations_and_legitimate_free_components_are_preserved():
    baseline = BASE.replace("N1 2 0.4 0.5 0.6 11.0 0.06",
        "N1 2 0.4 0.5 0.6 11.0 0.04 0.05 0.06 =\n 0.001 -0.002 0.003")
    application = apply_locks(baseline, directives(), approved_hard_lock_ids=APPROVED)
    text = application.text.replace("0.04 0.05 0.06 =", "0.05 0.06 0.07 =").replace("0.001 -0.002 0.003", "0.002 -0.003 0.004")
    result = normalize(text, baseline=baseline, application=application)
    assert "0.05 0.06 0.07 =\n 0.002 -0.003 0.004" in result.text


@pytest.mark.parametrize("instruction", ["AFIX 1", "PART 1", "RESI 1 TEST", "EXYZ C1 N1", "MOVE 0 0 0 1"])
def test_unsupported_output_constraint_context_blocks(instruction):
    application = apply_locks(BASE, directives(), approved_hard_lock_ids=APPROVED)
    with pytest.raises(LockApplicationError):
        normalize(application.text.replace("C1 1", instruction + "\nC1 1"), application=application)
