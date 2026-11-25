"""The strategy selector.

The design commitment being tested: the crystallographer is never asked which solver to use, the
decision is made from measurable properties of their own files, and every rule that fired is
recorded. In particular SHELXT must not be hard-coded as the universal entry point, and the
SHELXC-D-E route must be *excluded with a reason* rather than silently absent when the data cannot
support it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_hkl, write_ins
from crystalx_core.domain.chemistry import ChemicalPrior, MetalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.ids import make_stage_id
from crystalx_core.domain.onion import StageKind
from crystalx_core.domain.solving import SelectionRuleVerdict, SolverKind
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolKind
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.solvers.selector import RULE_IDS, SelectionPolicy, StrategySelector
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver

STAGE = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)


def diagnose(case: dict[str, Path]) -> DataDiagnosis:
    header = read_ins_header(case["ins"])
    reflections = scan_hkl(case["hkl"], cell=header.cell, hklf_hint=header.hklf_format)
    return DataDiagnosis(reflections=reflections, ins=header)


def excluded_reason(trace, kind: SolverKind) -> str | None:
    for excluded_kind, reason in trace.excluded:
        if excluded_kind is kind:
            return reason
    return None


# --------------------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------------------


def test_every_documented_rule_is_evaluated(merged_case):
    trace = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    assert tuple(r.rule_id for r in trace.rules) == RULE_IDS


def test_merged_data_fails_the_friedel_rule_with_an_actionable_message(merged_case):
    trace = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    rule = trace.rule("R01_FRIEDEL_PAIRS_PRESENT")
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.FAIL
    assert "merged" in rule.detail
    assert "re-integrate" in rule.detail


def test_unmerged_data_passes_the_friedel_rule(anomalous_case):
    trace = StrategySelector().select(diagnose(anomalous_case), ChemicalPrior(), stage_id=STAGE)
    rule = trace.rule("R01_FRIEDEL_PAIRS_PRESENT")
    assert rule is not None and rule.verdict is SelectionRuleVerdict.PASS


def test_heavy_element_rule_uses_atomic_number_only(merged_case):
    """Silver is Z=47, comfortably above the declared threshold. No physics table is involved."""
    trace = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    rule = trace.rule("R03_HEAVY_ELEMENT_PRESENT")
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.PASS
    assert "Ag (Z=47)" in rule.detail


def test_pure_organic_fails_the_heavy_element_rule(light_unknown_sg_case):
    trace = StrategySelector().select(diagnose(light_unknown_sg_case), ChemicalPrior(), stage_id=STAGE)
    rule = trace.rule("R03_HEAVY_ELEMENT_PRESENT")
    assert rule is not None and rule.verdict is SelectionRuleVerdict.FAIL


def test_metal_prior_can_supply_the_heavy_element(light_unknown_sg_case):
    prior = ChemicalPrior(metals=(MetalPrior("Ag", oxidation_state=1),))
    trace = StrategySelector().select(diagnose(light_unknown_sg_case), prior, stage_id=STAGE)
    rule = trace.rule("R03_HEAVY_ELEMENT_PRESENT")
    assert rule is not None and rule.verdict is SelectionRuleVerdict.PASS


def test_space_group_rule_reflects_the_instruction_file(merged_case, light_unknown_sg_case):
    known = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    unknown = StrategySelector().select(
        diagnose(light_unknown_sg_case), ChemicalPrior(), stage_id=STAGE
    )
    assert known.rule("R04_SPACE_GROUP_KNOWN").verdict is SelectionRuleVerdict.PASS
    assert unknown.rule("R04_SPACE_GROUP_KNOWN").verdict is SelectionRuleVerdict.FAIL


def test_anomalous_strength_rule_admits_it_cannot_decide(merged_case):
    """The honest gap: f'' is not hard-coded, so R07 reports insufficient information."""
    trace = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    rule = trace.rule("R07_ANOMALOUS_SIGNAL_STRENGTH")
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.INSUFFICIENT_INFORMATION
    assert "unevaluated rather than approximated" in rule.detail
    assert "cctbx" in rule.detail


def test_anomalous_strength_rule_uses_a_registered_table(anomalous_case):
    class FakeTable:
        def f_double_prime(self, element: str, wavelength_a: float) -> float:
            return 4.3 if element == "Ag" else 0.1

    trace = StrategySelector(anomalous_table=FakeTable()).select(
        diagnose(anomalous_case), ChemicalPrior(), stage_id=STAGE
    )
    rule = trace.rule("R07_ANOMALOUS_SIGNAL_STRENGTH")
    assert rule is not None
    assert rule.verdict is SelectionRuleVerdict.PASS
    assert "Ag" in rule.detail


def test_a_broken_table_provider_does_not_crash_selection(anomalous_case):
    class ExplodingTable:
        def f_double_prime(self, element: str, wavelength_a: float) -> float:
            raise RuntimeError("no data for this element")

    trace = StrategySelector(anomalous_table=ExplodingTable()).select(
        diagnose(anomalous_case), ChemicalPrior(), stage_id=STAGE
    )
    assert trace.rule("R07_ANOMALOUS_SIGNAL_STRENGTH").verdict is (
        SelectionRuleVerdict.INSUFFICIENT_INFORMATION
    )
    assert trace.nominated


def test_reference_structure_is_recorded_as_advisory(tmp_path: Path, merged_case):
    from crystalx_core.domain.chemistry import ReferenceStructure

    reference = ReferenceStructure(path=tmp_path / "parent.cif", label="parent scaffold")
    assert reference.counts_as_evidence is False
    trace = StrategySelector().select(
        diagnose(merged_case), ChemicalPrior(reference_structures=(reference,)), stage_id=STAGE
    )
    rule = trace.rule("R08_REFERENCE_STRUCTURE_AVAILABLE")
    assert rule is not None
    assert "Advisory only" in rule.detail
    assert "never counts as evidence" in rule.description


# --------------------------------------------------------------------------------------
# nomination
# --------------------------------------------------------------------------------------


def test_merged_heavy_atom_data_prefers_patterson_over_shelxt(merged_case):
    """A dominant heavy atom with a known space group and no anomalous signal.

    SHELXS Patterson is the classical answer here, and it must outrank the dual-space default.
    """
    trace = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    kinds = trace.nominated_kinds
    assert SolverKind.SHELXS_PATT in kinds
    assert SolverKind.SHELXT in kinds
    assert kinds.index(SolverKind.SHELXS_PATT) < kinds.index(SolverKind.SHELXT)


def test_shelxc_shelxd_is_excluded_with_a_reason_on_merged_data(merged_case):
    trace = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    assert SolverKind.SHELXC_SHELXD not in trace.nominated_kinds
    reason = excluded_reason(trace, SolverKind.SHELXC_SHELXD)
    assert reason is not None
    assert "Friedel" in reason


def test_shelxc_shelxd_leads_when_anomalous_data_and_a_metal_are_present(anomalous_case):
    trace = StrategySelector().select(
        diagnose(anomalous_case), ChemicalPrior(metals=(MetalPrior("Ag"),)), stage_id=STAGE
    )
    assert trace.nominated_kinds[0] is SolverKind.SHELXC_SHELXD
    top = trace.nominated[0]
    assert top.use_anomalous
    assert "Ag" in top.heavy_atom_assumption
    assert any("anomalous" in reason.lower() for reason in top.nominated_because)


def test_anode_verification_is_nominated_when_the_data_allow_it(anomalous_case):
    trace = StrategySelector().select(
        diagnose(anomalous_case), ChemicalPrior(metals=(MetalPrior("Ag"),)), stage_id=STAGE
    )
    assert SolverKind.ANODE_VERIFY in trace.nominated_kinds


def test_shelxt_is_the_only_route_without_a_space_group(light_unknown_sg_case):
    trace = StrategySelector().select(
        diagnose(light_unknown_sg_case), ChemicalPrior(), stage_id=STAGE
    )
    assert trace.nominated_kinds == (SolverKind.SHELXT,)
    for kind in (SolverKind.SHELXS_PATT, SolverKind.SHELXS_TREF, SolverKind.SHELXD_ABINITIO):
        reason = excluded_reason(trace, kind)
        assert reason is not None and "space group" in reason


def test_shelxt_priority_rises_when_it_must_determine_the_space_group(
    merged_case, light_unknown_sg_case
):
    with_sg = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    without_sg = StrategySelector().select(
        diagnose(light_unknown_sg_case), ChemicalPrior(), stage_id=STAGE
    )
    a = next(b for b in with_sg.nominated if b.kind is SolverKind.SHELXT)
    b = next(x for x in without_sg.nominated if x.kind is SolverKind.SHELXT)
    assert b.priority > a.priority


def test_ab_initio_shelxd_is_excluded_below_atomic_resolution(merged_case):
    trace = StrategySelector().select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    reason = excluded_reason(trace, SolverKind.SHELXD_ABINITIO)
    assert reason is not None
    assert "atomic resolution" in reason


def test_solution_branches_are_capped_and_the_cut_is_recorded(anomalous_case):
    """``max_branches`` bounds the solution routes that compete for the time budget."""
    policy = SelectionPolicy(max_branches=2)
    trace = StrategySelector(policy).select(
        diagnose(anomalous_case), ChemicalPrior(metals=(MetalPrior("Ag"),)), stage_id=STAGE
    )
    solution_branches = [b for b in trace.nominated if b.kind is not SolverKind.ANODE_VERIFY]
    assert len(solution_branches) == 2
    assert any("max_branches" in reason for _, reason in trace.excluded)


def test_verification_branches_are_not_squeezed_out_by_the_cap(anomalous_case):
    """ANODE is the independent physical check on a metal assignment; a tight solver cap must not
    remove it, because that is exactly when an independent check matters most."""
    policy = SelectionPolicy(max_branches=1)
    trace = StrategySelector(policy).select(
        diagnose(anomalous_case), ChemicalPrior(metals=(MetalPrior("Ag"),)), stage_id=STAGE
    )
    assert SolverKind.ANODE_VERIFY in trace.nominated_kinds


def test_branch_ids_are_ordered_and_unique(anomalous_case):
    trace = StrategySelector().select(
        diagnose(anomalous_case), ChemicalPrior(metals=(MetalPrior("Ag"),)), stage_id=STAGE
    )
    ids = [str(b.branch_id) for b in trace.nominated]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------------------
# overrides, tool availability, autotracing
# --------------------------------------------------------------------------------------


def test_user_override_forces_a_branch_and_says_so(merged_case):
    policy = SelectionPolicy(forced_kinds=(SolverKind.SHELXD_ABINITIO,))
    trace = StrategySelector(policy).select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    forced = next(b for b in trace.nominated if b.kind is SolverKind.SHELXD_ABINITIO)
    assert "override" in " ".join(forced.nominated_because)
    assert trace.rule("R09_USER_OVERRIDE").verdict is SelectionRuleVerdict.PASS


def test_user_exclusion_removes_a_branch(merged_case):
    policy = SelectionPolicy(excluded_kinds=(SolverKind.SHELXT,))
    trace = StrategySelector(policy).select(diagnose(merged_case), ChemicalPrior(), stage_id=STAGE)
    assert SolverKind.SHELXT not in trace.nominated_kinds
    assert excluded_reason(trace, SolverKind.SHELXT) == "excluded by user override"


def test_unavailable_tools_exclude_a_branch_with_the_resolver_diagnosis(merged_case, toolbox):
    """A branch that cannot run because Rosetta is missing must say exactly that."""
    from conftest import CPU_X86_64, macho_thin, write_executable

    write_executable(toolbox / "shelxt", macho_thin(CPU_X86_64))
    resolver = ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=False),
        environ={"PATH": ""},
    )
    availability = resolver.resolve_all([ToolKind.SHELXT, ToolKind.SHELXS])

    trace = StrategySelector().select(
        diagnose(merged_case), ChemicalPrior(), stage_id=STAGE, availability=availability
    )
    assert SolverKind.SHELXT not in trace.nominated_kinds
    reason = excluded_reason(trace, SolverKind.SHELXT)
    assert reason is not None
    assert "tool unavailable" in reason
    assert "Rosetta" in reason


def test_available_tools_permit_the_branch(merged_case, toolbox):
    from conftest import CPU_ARM64, macho_thin, write_executable

    for name in ("shelxt", "shelxs"):
        write_executable(toolbox / name, macho_thin(CPU_ARM64))
    resolver = ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=True),
        environ={"PATH": ""},
    )
    availability = resolver.resolve_all([ToolKind.SHELXT, ToolKind.SHELXS])
    trace = StrategySelector().select(
        diagnose(merged_case), ChemicalPrior(), stage_id=STAGE, availability=availability
    )
    assert SolverKind.SHELXT in trace.nominated_kinds


def test_autotracing_stays_off_for_a_metal_organic_system(anomalous_case):
    """SHELXE main-chain tracing assumes polypeptide geometry. It must not default on."""
    trace = StrategySelector().select(
        diagnose(anomalous_case), ChemicalPrior(metals=(MetalPrior("Ag"),)), stage_id=STAGE
    )
    assert all(not branch.enable_autotracing for branch in trace.nominated)


def test_peptide_declaration_is_recorded_in_the_notes(anomalous_case):
    trace = StrategySelector().select(
        diagnose(anomalous_case),
        ChemicalPrior(metals=(MetalPrior("Ag"),), is_peptide_like=True),
        stage_id=STAGE,
    )
    assert any("peptide" in note for note in trace.notes)


def test_expected_sites_are_carried_into_the_branch(anomalous_case):
    prior = ChemicalPrior(metals=(MetalPrior("Ag", expected_sites=3),))
    trace = StrategySelector().select(diagnose(anomalous_case), prior, stage_id=STAGE)
    substructure = next(b for b in trace.nominated if b.kind is SolverKind.SHELXC_SHELXD)
    assert substructure.expected_sites == 3


def test_no_applicable_branch_is_a_reportable_outcome(light_unknown_sg_case):
    policy = SelectionPolicy(excluded_kinds=(SolverKind.SHELXT,))
    trace = StrategySelector(policy).select(
        diagnose(light_unknown_sg_case), ChemicalPrior(), stage_id=STAGE
    )
    assert trace.nominated == ()
    assert any("reportable outcome" in note or "not a crash" in note for note in trace.notes)
