"""Choosing the initial-solution portfolio from measurable properties of the user's own files.

The crystallographer is deliberately **not** asked which solver to use. That is a technical decision
and it is made here, from evidence, with every rule recorded in a :class:`SelectionTrace` that ends
up in the run directory as ``selection_trace.json``.

Where a rule needs information the core does not have, it reports ``INSUFFICIENT_INFORMATION``
rather than guessing. Rule R07 is the honest example: judging anomalous signal strength properly
needs ``f''(element, wavelength)``, so it stays unevaluated until a dispersion-table provider is
registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from crystalx_core.domain.chemistry import ChemicalPrior, atomic_number
from crystalx_core.domain.data import (
    AnomalousAvailability,
    AnomalousScatteringTable,
    DataDiagnosis,
    HklfFormat,
    SpaceGroupConfidence,
)
from crystalx_core.domain.ids import StageId, make_branch_id
from crystalx_core.domain.solving import (
    Applicability,
    ApplicabilityVerdict,
    BranchBudget,
    InitialSolutionBranch,
    SelectionRuleResult,
    SelectionRuleVerdict,
    SelectionTrace,
    SolverKind,
    SolverRole,
    capability_for,
)
from crystalx_core.domain.symmetry import ChiralityCompatibility
from crystalx_core.domain.tools import ToolAvailabilityReport
from crystalx_core.solvers.base import tool_requirements_met

__all__ = ["SelectionPolicy", "StrategySelector", "RULE_IDS"]

RULE_IDS = (
    "R01_FRIEDEL_PAIRS_PRESENT",
    "R02_HKLF_FORMAT",
    "R03_HEAVY_ELEMENT_PRESENT",
    "R04_SPACE_GROUP_KNOWN",
    "R05_RESOLUTION",
    "R06_DATA_QUALITY",
    "R07_ANOMALOUS_SIGNAL_STRENGTH",
    "R08_REFERENCE_STRUCTURE_AVAILABLE",
    "R09_USER_OVERRIDE",
    "R10_CHIRALITY_CONSISTENT",
)


@dataclass(frozen=True)
class SelectionPolicy:
    """Declared thresholds. Everything here appears in the trace, so nothing is a hidden constant."""

    heavy_atom_z_threshold: int = 20
    """Atomic number at or above which an element counts as a 'heavy atom' for phasing purposes.
    Uses Z only, which is unambiguous and needs no physical table."""

    atomic_resolution_d_min_a: float = 1.2
    """Ab initio dual-space direct methods need roughly this or better."""

    poor_data_mean_i_over_sigma: float = 3.0
    max_branches: int = 4
    include_verification_branches: bool = True
    """Nominate ANODE verification when the data can support it."""

    forced_kinds: tuple[SolverKind, ...] = ()
    """User override. Always wins, always recorded as an override."""

    excluded_kinds: tuple[SolverKind, ...] = ()
    default_branch_budget: BranchBudget = field(default_factory=BranchBudget)


class StrategySelector:
    """Turns a Stage-0 diagnosis into a prioritised, justified set of solver branches."""

    def __init__(
        self,
        policy: SelectionPolicy | None = None,
        *,
        anomalous_table: AnomalousScatteringTable | None = None,
    ) -> None:
        self.policy = policy or SelectionPolicy()
        self.anomalous_table = anomalous_table

    # ------------------------------------------------------------------ rules
    def evaluate_rules(
        self, diagnosis: DataDiagnosis, prior: ChemicalPrior
    ) -> tuple[SelectionRuleResult, ...]:
        return (
            self._r01(diagnosis),
            self._r02(diagnosis),
            self._r03(diagnosis, prior),
            self._r04(diagnosis),
            self._r05(diagnosis),
            self._r06(diagnosis),
            self._r07(diagnosis, prior),
            self._r08(prior),
            self._r09(),
            self._r10(diagnosis, prior),
        )

    def _r01(self, diagnosis: DataDiagnosis) -> SelectionRuleResult:
        availability = diagnosis.reflections.anomalous
        present = availability is AnomalousAvailability.FRIEDEL_PAIRS_PRESENT
        if availability is AnomalousAvailability.UNKNOWN:
            verdict = SelectionRuleVerdict.INSUFFICIENT_INFORMATION
            detail = "The reflection file could not be scanned for Friedel mates."
        elif present:
            verdict = SelectionRuleVerdict.PASS
            detail = (
                f"{diagnosis.reflections.n_friedel_pairs} Friedel pairs are present as separate "
                "records, so experimental phasing from the anomalous signal is possible."
            )
        elif availability is AnomalousAvailability.FRIEDEL_PAIRS_SPARSE:
            verdict = SelectionRuleVerdict.FAIL
            detail = (
                f"Only {diagnosis.reflections.n_friedel_pairs} Friedel pair(s) were found, covering "
                f"{diagnosis.reflections.friedel_coverage:.2%} of the records. That is the signature "
                "of re-measured or duplicated reflections in an otherwise merged file, not of "
                "unmerged anomalous data, so the SHELXC-D-E route stays closed. Supply the unmerged "
                "file if experimental phasing is wanted."
            )
        else:
            verdict = SelectionRuleVerdict.FAIL
            detail = (
                "Friedel mates are merged in this file. The SHELXC-D-E route cannot be run on it; "
                "re-integrate without merging Friedel pairs if experimental phasing is wanted."
            )
        return SelectionRuleResult(
            rule_id="R01_FRIEDEL_PAIRS_PRESENT",
            description="Unmerged Friedel pairs are a hard prerequisite for SHELXC-D-E and ANODE.",
            verdict=verdict,
            detail=detail,
            inputs={
                "anomalous_availability": availability.value,
                "n_friedel_pairs": str(diagnosis.reflections.n_friedel_pairs),
                "friedel_coverage": f"{diagnosis.reflections.friedel_coverage:.4f}",
            },
            affects=(SolverKind.SHELXC_SHELXD, SolverKind.ANODE_VERIFY),
        )

    def _r02(self, diagnosis: DataDiagnosis) -> SelectionRuleResult:
        hklf = diagnosis.ins.hklf_format or diagnosis.reflections.hklf_format
        twinned = hklf in (HklfFormat.HKLF5, HklfFormat.HKLF6)
        return SelectionRuleResult(
            rule_id="R02_HKLF_FORMAT",
            description="HKLF 5/6 indicates twinned data, which constrains the usable routes.",
            verdict=SelectionRuleVerdict.FAIL if twinned else SelectionRuleVerdict.PASS,
            detail=(
                f"HKLF {int(hklf)} indicates a twinned reflection file; solution and refinement "
                "both need a twin law and the ordinary routes may mislead."
                if twinned
                else f"HKLF {int(hklf)}: ordinary single-component data."
            ),
            inputs={"hklf": str(int(hklf))},
            affects=(SolverKind.SHELXT, SolverKind.SHELXS_TREF, SolverKind.SHELXD_ABINITIO),
        )

    def _r03(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> SelectionRuleResult:
        """Is there a heavy scatterer to anchor on?

        The subtlety is *whose* composition to believe. A diffractometer's automatic pipeline
        writes a placeholder ``SFAC``/``UNIT`` before it knows what the compound is, and reading
        that as fact is how an octanuclear silver assembly came to be reported as having no heavy
        atom at all. When the file's composition fails
        :func:`~crystalx_core.domain.composition.assess_composition`, the crystallographer's prior
        is used alone, and the absence of a prior becomes ``INSUFFICIENT_INFORMATION`` -- not
        ``FAIL``. Those are different: one says "there is no heavy atom", the other says "nobody
        has told me, and the file is not a witness".
        """
        trust = diagnosis.ins.composition_trust(
            declared_metals=prior.metal_elements, declared_elements=prior.declared_elements
        )

        candidates: dict[str, int] = {}
        if trust.trusted:
            for symbol in diagnosis.ins.sfac:
                try:
                    candidates[symbol] = atomic_number(symbol)
                except KeyError:
                    continue
        for metal in prior.metals:
            candidates[metal.element] = metal.z
        for symbol in prior.declared_elements:
            try:
                candidates[symbol] = atomic_number(symbol)
            except KeyError:
                continue

        threshold = self.policy.heavy_atom_z_threshold
        heavy = {sym: z for sym, z in candidates.items() if z >= threshold}
        suffix = "" if trust.trusted else f" The file's own composition was ignored: {trust.detail}"

        if heavy:
            verdict = SelectionRuleVerdict.PASS
            listed = ", ".join(f"{s} (Z={z})" for s, z in sorted(heavy.items(), key=lambda i: -i[1]))
            detail = f"Heavy scatterers at or above Z={threshold}: {listed}.{suffix}"
        elif not candidates:
            verdict = SelectionRuleVerdict.INSUFFICIENT_INFORMATION
            detail = (
                "No composition can be trusted: there is no metal prior, and the instruction "
                f"file's own SFAC/UNIT is a placeholder.{suffix} Declare the expected metal with "
                "--metal to open the heavy-atom routes."
            )
        elif not trust.trusted:
            verdict = SelectionRuleVerdict.INSUFFICIENT_INFORMATION
            detail = (
                f"Nothing you declared reaches Z={threshold}, and the file's composition cannot "
                f"be used to rule a heavy atom out either.{suffix}"
            )
        else:
            verdict = SelectionRuleVerdict.FAIL
            detail = (
                f"No declared element reaches Z={threshold}; heavy-atom routes have nothing to "
                "anchor on."
            )
        return SelectionRuleResult(
            rule_id="R03_HEAVY_ELEMENT_PRESENT",
            description="A heavy scatterer raises the priority of Patterson and substructure routes.",
            verdict=verdict,
            detail=detail,
            inputs={
                "sfac": ",".join(diagnosis.ins.sfac),
                "metal_prior": ",".join(prior.metal_elements),
                "z_threshold": str(threshold),
                **trust.as_inputs(),
            },
            affects=(SolverKind.SHELXS_PATT, SolverKind.SHELXC_SHELXD, SolverKind.ANODE_VERIFY),
        )

    def _r10(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> SelectionRuleResult:
        """A homochiral material cannot occupy a space group with an improper operation.

        This is not a heuristic and not a preference. An inversion centre, mirror, glide or
        rotoinversion maps a molecule onto its mirror image; if every molecule has the same
        handedness there is nothing for it to map onto. Failing this rule is a **hard block**, not
        a demotion -- see :func:`crystalx_core.workflow.preconditions.check_preconditions`.
        """
        symmetry = diagnosis.ins.symmetry()
        enantiopure, why = prior.is_enantiopure()

        if not enantiopure:
            verdict = SelectionRuleVerdict.NOT_EVALUATED
            detail = f"Handedness is not assumed here: {why}."
        elif symmetry.compatibility is ChiralityCompatibility.CONTAINS_IMPROPER:
            verdict = SelectionRuleVerdict.FAIL
            operations = ", ".join(op.value for op in symmetry.improper_operations)
            detail = (
                f"The declared space group contains an improper operation ({operations}), but "
                f"{why}. A homochiral material cannot crystallise in it. {symmetry.explain()} "
                "Either the space group is wrong -- an automatic pipeline over-assigning the "
                "centrosymmetric supergroup is the usual cause, and SHELXT can redetermine it -- "
                "or the material is not enantiopure, in which case say so explicitly."
            )
        elif symmetry.compatibility is ChiralityCompatibility.UNKNOWN:
            verdict = SelectionRuleVerdict.INSUFFICIENT_INFORMATION
            detail = f"The space group could not be classified. {symmetry.explain()}"
        else:
            verdict = SelectionRuleVerdict.PASS
            detail = f"Sohncke group, consistent with a homochiral material. {symmetry.explain()}"

        return SelectionRuleResult(
            rule_id="R10_CHIRALITY_CONSISTENT",
            description=(
                "A homochiral material must be in a Sohncke space group; any improper operation "
                "makes the declared group impossible."
            ),
            verdict=verdict,
            detail=detail,
            inputs={
                "latt": str(diagnosis.ins.latt),
                "n_symm": str(len(diagnosis.ins.symm)),
                "compatibility": symmetry.compatibility.value,
                "enantiopure_assumed": str(enantiopure).lower(),
            },
            affects=(),
        )

    def _r04(self, diagnosis: DataDiagnosis) -> SelectionRuleResult:
        confidence = diagnosis.space_group_confidence
        known = confidence is not SpaceGroupConfidence.UNKNOWN
        return SelectionRuleResult(
            rule_id="R04_SPACE_GROUP_KNOWN",
            description=(
                "SHELXS/SHELXD/SHELXE need a declared space group; SHELXT can determine one."
            ),
            verdict=SelectionRuleVerdict.PASS if known else SelectionRuleVerdict.FAIL,
            detail=(
                f"Space group taken from {confidence.value}: LATT="
                f"{diagnosis.ins.latt}, {len(diagnosis.ins.symm)} SYMM operators."
                if known
                else "No LATT/SYMM declared. Only SHELXT can proceed without a space group."
            ),
            inputs={
                "confidence": confidence.value,
                "latt": str(diagnosis.ins.latt),
                "n_symm": str(len(diagnosis.ins.symm)),
            },
            affects=(
                SolverKind.SHELXS_PATT,
                SolverKind.SHELXS_TREF,
                SolverKind.SHELXD_ABINITIO,
                SolverKind.SHELXC_SHELXD,
                SolverKind.SHELXT,
            ),
        )

    def _r05(self, diagnosis: DataDiagnosis) -> SelectionRuleResult:
        d_min = diagnosis.reflections.d_min_a
        limit = self.policy.atomic_resolution_d_min_a
        if d_min is None:
            verdict = SelectionRuleVerdict.INSUFFICIENT_INFORMATION
            detail = "Resolution was not computed; a unit cell is needed to convert hkl to d."
        elif d_min <= limit:
            verdict = SelectionRuleVerdict.PASS
            detail = f"d_min = {d_min:.2f} A meets the atomic-resolution requirement (<= {limit} A)."
        else:
            verdict = SelectionRuleVerdict.FAIL
            detail = (
                f"d_min = {d_min:.2f} A does not reach atomic resolution (<= {limit} A), so "
                "ab initio dual-space direct methods are not appropriate."
            )
        return SelectionRuleResult(
            rule_id="R05_RESOLUTION",
            description="Ab initio SHELXD needs true atomic resolution.",
            verdict=verdict,
            detail=detail,
            inputs={
                "d_min_a": f"{d_min:.3f}" if d_min is not None else "unknown",
                "threshold_a": str(limit),
            },
            affects=(SolverKind.SHELXD_ABINITIO,),
        )

    def _r06(self, diagnosis: DataDiagnosis) -> SelectionRuleResult:
        reflections = diagnosis.reflections
        mean_ratio = reflections.mean_i_over_sigma
        notes = [f"{reflections.n_reflections} reflections"]
        if reflections.completeness is None and reflections.completeness_unavailable is not None:
            notes.append(f"completeness not computed ({reflections.completeness_unavailable.reason})")
        if mean_ratio is None:
            verdict = SelectionRuleVerdict.INSUFFICIENT_INFORMATION
        elif mean_ratio < self.policy.poor_data_mean_i_over_sigma:
            verdict = SelectionRuleVerdict.FAIL
            notes.append(
                f"mean I/sigma = {mean_ratio:.1f}, below the {self.policy.poor_data_mean_i_over_sigma} "
                "threshold; expect every route to struggle"
            )
        else:
            verdict = SelectionRuleVerdict.PASS
            notes.append(f"mean I/sigma = {mean_ratio:.1f}")
        return SelectionRuleResult(
            rule_id="R06_DATA_QUALITY",
            description="Overall data quality; poor data downgrades every branch and may need review.",
            verdict=verdict,
            detail="; ".join(notes),
            inputs={
                "n_reflections": str(reflections.n_reflections),
                "mean_i_over_sigma": f"{mean_ratio:.3f}" if mean_ratio is not None else "unknown",
            },
        )

    def _r07(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> SelectionRuleResult:
        wavelength = diagnosis.wavelength_a
        elements = sorted({*diagnosis.ins.sfac, *prior.metal_elements})
        if self.anomalous_table is None:
            return SelectionRuleResult(
                rule_id="R07_ANOMALOUS_SIGNAL_STRENGTH",
                description=(
                    "Expected anomalous signal from f''(element, wavelength). The same silver "
                    "compound behaves very differently at Cu K-alpha and Mo K-alpha."
                ),
                verdict=SelectionRuleVerdict.INSUFFICIENT_INFORMATION,
                detail=(
                    "No dispersion-table provider is registered, so f'' is unknown. This rule is "
                    "left unevaluated rather than approximated; selection proceeds on R01 and R03. "
                    "Register a cctbx-backed AnomalousScatteringTable to enable it."
                ),
                inputs={
                    "wavelength_a": f"{wavelength:.5f}" if wavelength else "unknown",
                    "elements": ",".join(elements),
                },
                affects=(SolverKind.SHELXC_SHELXD, SolverKind.ANODE_VERIFY),
            )

        if wavelength is None or not elements:
            return SelectionRuleResult(
                rule_id="R07_ANOMALOUS_SIGNAL_STRENGTH",
                description="Expected anomalous signal from f''(element, wavelength).",
                verdict=SelectionRuleVerdict.INSUFFICIENT_INFORMATION,
                detail="Wavelength or composition is unknown.",
                inputs={"wavelength_a": str(wavelength), "elements": ",".join(elements)},
                affects=(SolverKind.SHELXC_SHELXD, SolverKind.ANODE_VERIFY),
            )

        contributions: dict[str, float] = {}
        for element in elements:
            try:
                contributions[element] = float(
                    self.anomalous_table.f_double_prime(element, wavelength)
                )
            except Exception:  # noqa: BLE001 - a provider gap must not crash selection
                continue
        if not contributions:
            verdict = SelectionRuleVerdict.INSUFFICIENT_INFORMATION
            detail = "The registered dispersion table returned no usable values."
        else:
            best_element = max(contributions, key=lambda e: contributions[e])
            best = contributions[best_element]
            verdict = SelectionRuleVerdict.PASS if best >= 1.0 else SelectionRuleVerdict.FAIL
            detail = (
                f"Strongest anomalous scatterer at {wavelength:.5f} A is {best_element} "
                f"with f'' = {best:.2f} e."
            )
        return SelectionRuleResult(
            rule_id="R07_ANOMALOUS_SIGNAL_STRENGTH",
            description="Expected anomalous signal from f''(element, wavelength).",
            verdict=verdict,
            detail=detail,
            inputs={
                "wavelength_a": f"{wavelength:.5f}",
                **{f"fdp_{k}": f"{v:.3f}" for k, v in sorted(contributions.items())},
            },
            affects=(SolverKind.SHELXC_SHELXD, SolverKind.ANODE_VERIFY),
        )

    def _r08(self, prior: ChemicalPrior) -> SelectionRuleResult:
        references = prior.reference_structures
        return SelectionRuleResult(
            rule_id="R08_REFERENCE_STRUCTURE_AVAILABLE",
            description=(
                "A topologically similar reference may order the search and seed restraints. "
                "It never counts as evidence."
            ),
            verdict=SelectionRuleVerdict.PASS if references else SelectionRuleVerdict.NOT_EVALUATED,
            detail=(
                f"{len(references)} reference structure(s) supplied: "
                + ", ".join(r.label or r.path.name for r in references)
                + ". Advisory only; any candidate they influence is flagged in its provenance."
                if references
                else "No reference structure supplied."
            ),
            inputs={"n_references": str(len(references))},
        )

    def _r09(self) -> SelectionRuleResult:
        forced = self.policy.forced_kinds
        excluded = self.policy.excluded_kinds
        active = bool(forced or excluded)
        return SelectionRuleResult(
            rule_id="R09_USER_OVERRIDE",
            description="An explicit user override wins over every other rule and is recorded as one.",
            verdict=SelectionRuleVerdict.PASS if active else SelectionRuleVerdict.NOT_EVALUATED,
            detail=(
                f"forced={[k.value for k in forced]} excluded={[k.value for k in excluded]}"
                if active
                else "No override."
            ),
            inputs={
                "forced": ",".join(k.value for k in forced),
                "excluded": ",".join(k.value for k in excluded),
            },
        )

    # ------------------------------------------------------------------ nomination
    def select(
        self,
        diagnosis: DataDiagnosis,
        prior: ChemicalPrior,
        *,
        stage_id: StageId,
        availability: ToolAvailabilityReport | None = None,
    ) -> SelectionTrace:
        """Produce the branch plan plus the full reasoning behind it."""
        rules = self.evaluate_rules(diagnosis, prior)
        by_id = {r.rule_id: r for r in rules}

        friedel = by_id["R01_FRIEDEL_PAIRS_PRESENT"].verdict is SelectionRuleVerdict.PASS
        heavy = by_id["R03_HEAVY_ELEMENT_PRESENT"].verdict is SelectionRuleVerdict.PASS
        sg_known = by_id["R04_SPACE_GROUP_KNOWN"].verdict is SelectionRuleVerdict.PASS
        atomic_res = by_id["R05_RESOLUTION"].verdict is SelectionRuleVerdict.PASS
        twinned = by_id["R02_HKLF_FORMAT"].verdict is SelectionRuleVerdict.FAIL
        chirality_impossible = by_id["R10_CHIRALITY_CONSISTENT"].verdict is SelectionRuleVerdict.FAIL

        scored: list[tuple[int, SolverKind, Applicability]] = []
        excluded: list[tuple[SolverKind, str]] = []

        def consider(kind: SolverKind, applicability: Applicability) -> None:
            if kind in self.policy.excluded_kinds:
                excluded.append((kind, "excluded by user override"))
                return
            if chirality_impossible and capability_for(kind).requires_space_group:
                # Hard check for fixed-space-group routes: they would work in the declared group,
                # which R10 shows a homochiral material cannot occupy. Discovery routes (SHELXT)
                # are judged on the group they choose instead.
                excluded.append(
                    (kind, "R10: the declared space group cannot hold the declared homochiral material, "
                     "and this route would use it as given")
                )
                return
            if applicability.verdict is ApplicabilityVerdict.APPLICABLE:
                scored.append((applicability.priority, kind, applicability))
            else:
                excluded.append((kind, "; ".join(applicability.reasons) or applicability.verdict.value))

        consider(SolverKind.SHELXT, self._shelxt(sg_known, twinned))
        consider(SolverKind.SHELXS_PATT, self._shelxs_patt(sg_known, heavy))
        consider(SolverKind.SHELXS_TREF, self._shelxs_tref(sg_known))
        consider(SolverKind.SHELXD_ABINITIO, self._shelxd_abinitio(sg_known, atomic_res))
        consider(SolverKind.SHELXC_SHELXD, self._shelxc_shelxd(sg_known, heavy, friedel))
        if self.policy.include_verification_branches:
            consider(SolverKind.ANODE_VERIFY, self._anode(friedel, heavy))

        # Tool availability turns an otherwise good branch into a recorded, explained exclusion.
        if availability is not None:
            still_applicable: list[tuple[int, SolverKind, Applicability]] = []
            for priority, kind, applicability in scored:
                ok, missing = tool_requirements_met(kind, availability)
                if ok:
                    still_applicable.append((priority, kind, applicability))
                else:
                    excluded.append((kind, "tool unavailable: " + "; ".join(missing)))
            scored = still_applicable

        r10_excluded = {kind for kind, reason in excluded if reason.startswith("R10:")}
        for forced in self.policy.forced_kinds:
            if forced in r10_excluded:
                # A route override does not lift R10. The way to do that is to declare the
                # preparation racemic, which changes the rule's input rather than ignoring it.
                continue
            if forced not in [kind for _, kind, _ in scored]:
                scored.append(
                    (
                        1000,
                        forced,
                        Applicability(
                            verdict=ApplicabilityVerdict.APPLICABLE,
                            priority=1000,
                            reasons=("forced by explicit user override",),
                        ),
                    )
                )
                excluded = [(k, r) for k, r in excluded if k is not forced]

        scored.sort(key=lambda item: (-item[0], item[1].value))

        # ``max_branches`` bounds how many *solution* routes compete for the time budget.
        # Verification branches such as ANODE are cheap and are the independent physical check on a
        # metal assignment, so they must not be squeezed out by the solver cap.
        solution_scored = [
            item for item in scored if capability_for(item[1]).role is not SolverRole.VERIFICATION
        ]
        verification_scored = [
            item for item in scored if capability_for(item[1]).role is SolverRole.VERIFICATION
        ]
        chosen = solution_scored[: self.policy.max_branches]
        for _priority, kind, _applicability in solution_scored[self.policy.max_branches :]:
            excluded.append((kind, f"below the max_branches={self.policy.max_branches} cut"))
        chosen = chosen + verification_scored

        branches: list[InitialSolutionBranch] = []
        for index, (priority, kind, applicability) in enumerate(chosen):
            branches.append(
                self._build_branch(
                    index=index,
                    kind=kind,
                    applicability=applicability,
                    priority=priority,
                    stage_id=stage_id,
                    diagnosis=diagnosis,
                    prior=prior,
                )
            )

        notes: list[str] = []
        if not branches:
            notes.append(
                "No initial-solution branch is applicable. This is a reportable outcome, not a "
                "crash: see the excluded list for the reason behind each route."
            )
        if twinned:
            notes.append(
                "Twinned data (HKLF 5/6): a twin law is required and results should be treated as "
                "provisional until it is established."
            )
        if prior.is_peptide_like:
            notes.append(
                "The user declared a peptide-like system, so SHELXE main-chain autotracing may be "
                "enabled where SHELXE is used."
            )

        return SelectionTrace(
            rules=rules,
            nominated=tuple(branches),
            excluded=tuple(excluded),
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------ per-solver logic
    def _shelxt(self, sg_known: bool, twinned: bool) -> Applicability:
        reasons = ["Dual-space recycling; the default for routine data."]
        priority = 70
        if not sg_known:
            priority += 25
            reasons.append(
                "No space group is declared and SHELXT is the only route that determines one."
            )
        if twinned:
            priority -= 20
            reasons.append("Twinned data reduce confidence in a dual-space solution.")
        return Applicability(ApplicabilityVerdict.APPLICABLE, priority, tuple(reasons))

    def _shelxs_patt(self, sg_known: bool, heavy: bool) -> Applicability:
        if not sg_known:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("SHELXS needs a declared space group (LATT/SYMM).",),
            )
        if not heavy:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("Patterson interpretation needs a dominant heavy scatterer.",),
            )
        return Applicability(
            ApplicabilityVerdict.APPLICABLE,
            85,
            (
                "A heavy scatterer is present and the space group is known: Patterson "
                "interpretation is the classical route to the heavy-atom positions.",
            ),
        )

    def _shelxs_tref(self, sg_known: bool) -> Applicability:
        if not sg_known:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("SHELXS needs a declared space group (LATT/SYMM).",),
            )
        return Applicability(
            ApplicabilityVerdict.APPLICABLE,
            35,
            ("Classical direct methods, kept as an independent cross-check against SHELXT.",),
        )

    def _shelxd_abinitio(self, sg_known: bool, atomic_res: bool) -> Applicability:
        reasons: list[str] = []
        if not sg_known:
            reasons.append("SHELXD needs a declared space group.")
        if not atomic_res:
            reasons.append(
                "Ab initio dual-space direct methods need true atomic resolution "
                f"(d_min <= {self.policy.atomic_resolution_d_min_a} A)."
            )
        if reasons:
            return Applicability(ApplicabilityVerdict.NOT_APPLICABLE, reasons=tuple(reasons))
        return Applicability(
            ApplicabilityVerdict.APPLICABLE,
            45,
            ("Atomic resolution is available, so ab initio dual-space solution is viable.",),
        )

    def _shelxc_shelxd(self, sg_known: bool, heavy: bool, friedel: bool) -> Applicability:
        reasons: list[str] = []
        if not friedel:
            reasons.append(
                "Friedel mates are merged in this reflection file, so there is no anomalous "
                "difference to work from."
            )
        if not sg_known:
            reasons.append("Experimental phasing needs a declared space group.")
        if reasons:
            return Applicability(ApplicabilityVerdict.NOT_APPLICABLE, reasons=tuple(reasons))
        priority = 90 if heavy else 60
        justification = [
            "Unmerged Friedel pairs are present: SHELXC can measure the anomalous signal and "
            "SHELXD can solve the substructure from it.",
        ]
        if heavy:
            justification.append("A heavy scatterer is declared, which is what the route needs.")
        else:
            justification.append(
                "No heavy scatterer is declared, so the anomalous signal may be weak; SHELXC's "
                "signal-versus-resolution table will settle it before SHELXD is run."
            )
        return Applicability(ApplicabilityVerdict.APPLICABLE, priority, tuple(justification))

    def _anode(self, friedel: bool, heavy: bool) -> Applicability:
        if not friedel:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("ANODE needs unmerged Friedel pairs to compute anomalous density.",),
            )
        if not heavy:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("No heavy scatterer is declared, so there is nothing to verify.",),
            )
        return Applicability(
            ApplicabilityVerdict.APPLICABLE,
            20,
            (
                "Anomalous difference density gives an independent, measured check on each metal "
                "assignment, rather than relying on a model score.",
            ),
        )

    def _build_branch(
        self,
        *,
        index: int,
        kind: SolverKind,
        applicability: Applicability,
        priority: int,
        stage_id: StageId,
        diagnosis: DataDiagnosis,
        prior: ChemicalPrior,
    ) -> InitialSolutionBranch:
        capability = capability_for(kind)
        heavy_elements = tuple(
            sorted(
                {
                    element
                    for element in {*diagnosis.ins.sfac, *prior.metal_elements}
                    if _z_or_zero(element) >= self.policy.heavy_atom_z_threshold
                }
            )
        )
        expected_sites: int | None = None
        declared = [m.expected_sites for m in prior.metals if m.expected_sites]
        if declared:
            expected_sites = sum(declared)

        autotracing = bool(
            prior.is_peptide_like and kind is SolverKind.SHELXE_DENSITY_MOD
        )
        return InitialSolutionBranch(
            branch_id=make_branch_id(index, kind.value),
            kind=kind,
            stage_id=stage_id,
            priority=priority,
            applicability=applicability,
            budget=self.policy.default_branch_budget,
            space_group_assumption=(
                f"LATT={diagnosis.ins.latt}" if diagnosis.ins.latt is not None else ""
            ),
            heavy_atom_assumption=heavy_elements if capability.benefits_from_heavy_atom else (),
            expected_sites=expected_sites,
            resolution_cut_a=diagnosis.resolution_policy.d_min_a,
            use_anomalous=capability.requires_unmerged_friedel_pairs,
            enable_autotracing=autotracing,
            nominated_because=applicability.reasons,
        )


def _z_or_zero(symbol: str) -> int:
    try:
        return atomic_number(symbol)
    except KeyError:
        return 0


def summarise_trace(trace: SelectionTrace) -> Sequence[str]:  # pragma: no cover - display helper
    """Human-readable one-liners, for the CLI."""
    lines = [f"{r.rule_id}: {r.verdict.value} - {r.detail}" for r in trace.rules]
    lines.append("")
    for branch in trace.nominated:
        lines.append(f"nominate {branch.branch_id} ({branch.kind.value}) priority={branch.priority}")
        for reason in branch.nominated_because:
            lines.append(f"    because {reason}")
    for kind, reason in trace.excluded:
        lines.append(f"exclude  {kind.value}: {reason}")
    return lines
