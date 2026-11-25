"""Hand-authored synthetic regressions for composition and symmetry guards.

All labels, compositions, counts and instruction snippets below are invented for these tests.
No private archive case, expert answer or measured scientific value is represented here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_hkl, write_ins
from crystalx_core.domain.chemistry import ChemicalPrior, MetalPrior
from crystalx_core.domain.composition import CompositionSignal, assess_composition
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.solving import SelectionRuleVerdict
from crystalx_core.domain.symmetry import (
    ChiralityCompatibility,
    ImproperOperation,
    analyse_symmetry,
    parse_symm,
)
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.solvers.selector import StrategySelector
from crystalx_core.domain.ids import make_stage_id
from crystalx_core.domain.onion import StageKind
from crystalx_core.workflow.preconditions import PreconditionKind, check_preconditions

STAGE = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)


# --------------------------------------------------------------------------------------
# operator parsing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, determinant",
    [
        ("x, y, z", 1),
        ("-x, y+1/2, -z", 1),  # 2(1) screw: proper
        ("-x, -y, -z", -1),  # inversion
        ("x, -y, z", -1),  # mirror
        ("1/2+x, 1/2-y, 1/2+z", -1),  # glide
        ("-y, x-y, z", 1),  # 3-fold, hexagonal setting
        ("y, -x, -z", -1),  # -4
    ],
)
def test_operator_determinants_are_exact(text: str, determinant: int):
    """Integer arithmetic throughout: the sign of a determinant is never a rounding question."""
    operator = parse_symm(text)
    assert operator is not None, text
    assert operator.determinant == determinant


@pytest.mark.parametrize("text", ["3x, y, z", "2x+1/2, y, z", "x, y", "q, y, z", "x, y, z z", ""])
def test_an_unreadable_operator_is_refused_not_half_read(text: str):
    """Regression: "3x" once parsed as "+x" with the 3 silently dropped.

    A misread operator is worse than an unread one -- it produces a confident, wrong determinant.
    """
    assert parse_symm(text) is None


def test_an_unreadable_operator_downgrades_the_verdict_rather_than_crashing():
    result = analyse_symmetry(-1, ["-x,y,-z", "q,y,z"])
    assert result.compatibility is ChiralityCompatibility.UNKNOWN
    assert result.unparsed == ("q,y,z",)


def test_a_found_mirror_beats_an_unread_operator():
    """You cannot un-find a mirror: a decisive improper operation wins over an unknown."""
    result = analyse_symmetry(-1, ["x,-y,z", "q,y,z"])
    assert result.compatibility is ChiralityCompatibility.CONTAINS_IMPROPER


# --------------------------------------------------------------------------------------
# Sohncke classification
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, latt, symm, expected",
    [
        ("P1", -1, [], ChiralityCompatibility.SOHNCKE),
        ("P-1", 1, [], ChiralityCompatibility.CONTAINS_IMPROPER),
        ("P2(1)", -1, ["-x, y+1/2, -z"], ChiralityCompatibility.SOHNCKE),
        ("P2(1)/m", 1, ["-x, y+1/2, -z"], ChiralityCompatibility.CONTAINS_IMPROPER),
        (
            "P2(1)2(1)2(1)",
            -1,
            ["1/2-x,-y,1/2+z", "-x,1/2+y,1/2-z", "1/2+x,1/2-y,-z"],
            ChiralityCompatibility.SOHNCKE,
        ),
        (
            "Pna2(1)",
            -1,
            ["-x,-y,1/2+z", "1/2+x,1/2-y,1/2+z", "1/2-x,1/2+y,z"],
            ChiralityCompatibility.CONTAINS_IMPROPER,
        ),
        ("C2/c", 7, ["-x,y,1/2-z"], ChiralityCompatibility.CONTAINS_IMPROPER),
        ("no LATT declared", None, ["-x, y+1/2, -z"], ChiralityCompatibility.UNKNOWN),
    ],
)
def test_space_groups_are_classified_correctly(name, latt, symm, expected):
    assert analyse_symmetry(latt, symm).compatibility is expected, name


def test_a_positive_latt_is_reported_as_an_inversion_centre_even_with_no_symm():
    """SHELX adds the inversion centre implicitly; it is never a ``SYMM`` record."""
    result = analyse_symmetry(1, [])
    assert ImproperOperation.INVERSION_CENTRE in result.improper_operations
    assert "LATT 1 is positive" in result.explain()


# --------------------------------------------------------------------------------------
# composition trust
# --------------------------------------------------------------------------------------

VENDOR_REM = ("Synthetic instruction example", "Ins file created by synthetic fixture")


def test_a_real_composition_is_trusted():
    """``UNIT`` counts being multiples of Z is the definition of ``UNIT``, not a smell.

    An earlier version treated it as evidence of a template and rejected this exact composition.
    """
    trust = assess_composition(
        sfac=("C", "N", "O", "Cu"), unit=(24, 4, 8, 2), z=2, rem_lines=(), declared_metals=("Cu",)
    )
    assert trust.trusted
    assert trust.signals == () or CompositionSignal.DECLARED_METAL_ABSENT not in trust.signals


def test_all_counts_equal_is_a_placeholder():
    """Invented five-element declaration with the same arbitrary count in every slot."""
    trust = assess_composition(
        sfac=("C", "H", "N", "O", "Fe"), unit=(9,) * 5, z=9, rem_lines=VENDOR_REM
    )
    assert not trust.trusted
    assert CompositionSignal.ALL_COUNTS_EQUAL in trust.signals
    assert "identical" in trust.detail


def test_a_declared_metal_missing_from_sfac_makes_the_file_incomplete():
    """An invented organic declaration omits the independently declared copper prior."""
    trust = assess_composition(
        sfac=("C", "H", "N", "O"),
        unit=(28, 36, 8, 4),
        z=4,
        rem_lines=VENDOR_REM,
        declared_metals=("Cu",),
    )
    assert not trust.trusted
    assert CompositionSignal.DECLARED_METAL_ABSENT in trust.signals
    assert trust.missing_declared_metals == ("Cu",)


def test_a_vendor_file_with_no_atoms_is_a_starting_guess():
    trust = assess_composition(
        sfac=("C", "H", "N", "O"), unit=(28, 36, 8, 4), z=4, rem_lines=VENDOR_REM
    )
    assert not trust.trusted
    assert CompositionSignal.PRE_SOLUTION_VENDOR_FILE in trust.signals
    # The assumed formula unit is shown so the crystallographer can see what was invented.
    assert "C7H9N2O" in trust.detail


def test_the_same_vendor_file_is_trusted_once_it_has_atoms():
    """A vendor file that has been refined has had a human through it."""
    trust = assess_composition(
        sfac=("C", "H", "N", "O"),
        unit=(28, 36, 8, 4),
        z=4,
        rem_lines=VENDOR_REM,
        has_atom_records=True,
    )
    assert trust.trusted


def test_atom_records_are_detected_in_the_reader(tmp_path: Path):
    bare = read_ins_header(write_ins(tmp_path / "bare.ins"))
    assert not bare.has_atom_records

    with_atoms = read_ins_header(
        write_ins(tmp_path / "atoms.ins", sfac=("C", "N", "O", "Cu"), extra_lines=("Cu1   4  0.25000  0.50000  0.75000  11.00000  0.02000",))
    )
    assert with_atoms.has_atom_records


def test_rem_lines_are_kept_not_skipped(tmp_path: Path):
    header = read_ins_header(
        write_ins(tmp_path / "rem.ins", extra_lines=("REM Ins file created by synthetic fixture",))
    )
    assert any("synthetic fixture" in line for line in header.rem_lines)


# --------------------------------------------------------------------------------------
# the rules, and the block
# --------------------------------------------------------------------------------------


def _diagnose(tmp_path: Path, **ins_kwargs) -> DataDiagnosis:
    ins = write_ins(tmp_path / "case.ins", **ins_kwargs)
    hkl = write_hkl(tmp_path / "case.hkl", include_friedel=True)
    header = read_ins_header(ins)
    return DataDiagnosis(
        reflections=scan_hkl(hkl, cell=header.cell, hklf_hint=header.hklf_format), ins=header
    )


def test_r03_refuses_to_read_a_placeholder_as_evidence_of_no_heavy_atom(tmp_path: Path):
    """A placeholder cannot establish absence of an owner-declared heavy element."""
    diagnosis = _diagnose(
        tmp_path,
        sfac=("C", "H", "N", "O"),
        unit=(28, 36, 8, 4),
        extra_lines=("REM Ins file created by synthetic fixture",),
    )
    rule = StrategySelector().select(diagnosis, ChemicalPrior(), stage_id=STAGE).rule(
        "R03_HEAVY_ELEMENT_PRESENT"
    )
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.INSUFFICIENT_INFORMATION, (
        "a placeholder must not be read as proof that there is no heavy atom"
    )
    assert rule.inputs["composition_trusted"] == "false"


def test_r03_uses_the_prior_when_the_file_is_a_placeholder(tmp_path: Path):
    diagnosis = _diagnose(
        tmp_path,
        sfac=("C", "H", "N", "O"),
        unit=(28, 36, 8, 4),
        extra_lines=("REM Ins file created by synthetic fixture",),
    )
    prior = ChemicalPrior(metals=(MetalPrior(element="Cu"),))
    rule = StrategySelector().select(diagnosis, prior, stage_id=STAGE).rule(
        "R03_HEAVY_ELEMENT_PRESENT"
    )
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.PASS
    assert "Cu (Z=29)" in rule.detail


def test_r10_fails_for_a_peptide_in_a_centrosymmetric_group(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=1, symm=("-x, y+1/2, -z",))
    rule = StrategySelector().select(
        diagnosis, ChemicalPrior(is_peptide_like=True), stage_id=STAGE
    ).rule("R10_CHIRALITY_CONSISTENT")
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.FAIL


def test_r10_is_not_evaluated_when_handedness_is_not_claimed(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=1, symm=("-x, y+1/2, -z",))
    rule = StrategySelector().select(diagnosis, ChemicalPrior(), stage_id=STAGE).rule(
        "R10_CHIRALITY_CONSISTENT"
    )
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.NOT_EVALUATED


def test_a_peptide_in_a_centrosymmetric_group_blocks_the_run(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=1, symm=("-x, y+1/2, -z",))
    report = check_preconditions(diagnosis, ChemicalPrior(is_peptide_like=True))

    assert not report.may_start
    blocker = report.blockers[0]
    assert blocker.kind is PreconditionKind.CHIRALITY_IMPOSSIBLE
    assert "SHELXT" in blocker.remedy, "a block must say what to do instead"
    assert "--racemic" in blocker.override_hint, "a block must say how to overrule it"


def test_declaring_the_preparation_racemic_lifts_the_block(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=1, symm=("-x, y+1/2, -z",))
    prior = ChemicalPrior(is_peptide_like=True, enantiopure=False)
    assert check_preconditions(diagnosis, prior).may_start


def test_the_override_flag_also_lifts_it_and_is_not_the_default(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=1, symm=("-x, y+1/2, -z",))
    prior = ChemicalPrior(is_peptide_like=True)
    assert not check_preconditions(diagnosis, prior).may_start
    assert check_preconditions(diagnosis, prior, allow_improper_space_group=True).may_start


def test_a_sohncke_group_does_not_block(tmp_path: Path):
    diagnosis = _diagnose(tmp_path, latt=-1, symm=("-x, y+1/2, -z",))
    assert check_preconditions(diagnosis, ChemicalPrior(is_peptide_like=True)).may_start


def test_a_missing_composition_is_advisory_not_blocking(tmp_path: Path):
    """SHELXT can run without a composition. It should say so, not stop."""
    diagnosis = _diagnose(
        tmp_path,
        latt=-1,
        sfac=("C", "H", "N", "O"),
        unit=(28, 36, 8, 4),
        extra_lines=("REM Ins file created by synthetic fixture",),
    )
    report = check_preconditions(diagnosis, ChemicalPrior())
    assert report.may_start
    assert report.advisories
    assert report.advisories[0].kind is PreconditionKind.COMPOSITION_UNUSABLE


# --------------------------------------------------------------------------------------
# answer-key leakage
# --------------------------------------------------------------------------------------


def test_a_reference_structure_is_ordinary_and_allowed():
    from crystalx_core.domain.chemistry import ReferenceStructure

    reference = ReferenceStructure(path=Path("synthetic_reference.cif"), label="invented distinct-measurement reference")
    assert reference.counts_as_evidence is False
    assert reference.may_order_search is True


def test_a_solution_of_the_same_measurement_cannot_be_used_as_a_reference():
    """The answer key and a reference look identical on disk; only the chemist knows which it is.

    Feeding the solved structure of the very reflection file being solved would let the pipeline
    reproduce a result it was supposed to derive, and every figure of merit downstream would be
    measuring nothing. So it is refused at construction rather than trusted to be used correctly.
    """
    from crystalx_core.domain.chemistry import ReferenceStructure

    with pytest.raises(ValueError) as excinfo:
        ReferenceStructure(
            path=Path("synthetic_held_out_answer.res"),
            label="verified solution of this same dataset",
            derived_from_same_measurement=True,
        )
    message = str(excinfo.value)
    assert "answer key" in message
    assert "grade the result afterwards" in message, "a refusal must say what to do instead"


def test_a_missing_non_metal_element_is_caught_too():
    """An invented declaration cannot describe independently expected missing halogens."""
    trust = assess_composition(
        sfac=("C", "Cu", "N", "O"),
        unit=(20, 2, 4, 6),
        z=1,
        declared_metals=("Cu",),
        declared_elements=("Cl", "Br", "H"),
    )
    assert not trust.trusted
    assert CompositionSignal.DECLARED_ELEMENT_ABSENT in trust.signals
    assert trust.missing_declared_elements == ("Cl", "Br", "H")
    assert CompositionSignal.DECLARED_METAL_ABSENT not in trust.signals, "Cu is present"


def test_declaring_only_what_is_present_keeps_the_composition_trusted():
    trust = assess_composition(
        sfac=("C", "Cu", "N", "O"),
        unit=(20, 2, 4, 6),
        z=1,
        declared_metals=("Cu",),
        declared_elements=("C", "N", "O"),
    )
    assert trust.trusted


def test_p1_passes_the_chirality_check_but_says_the_pass_is_uninformative():
    """LATT -1 with no SYMM is P1: Sohncke trivially, because it imposes nothing.

    Reporting that as a plain pass invites reading it as confirmation that the symmetry is right.
    A reduction left in P1 has postponed the space group, not determined it.
    """
    symmetry = analyse_symmetry(-1, [])
    assert symmetry.compatibility is ChiralityCompatibility.SOHNCKE
    assert symmetry.is_trivially_sohncke
    assert "postponed the space group rather than determined it" in symmetry.explain()


def test_a_real_sohncke_group_is_not_reported_as_trivial():
    symmetry = analyse_symmetry(-1, ["-x, y+1/2, -z"])
    assert symmetry.compatibility is ChiralityCompatibility.SOHNCKE
    assert not symmetry.is_trivially_sohncke
