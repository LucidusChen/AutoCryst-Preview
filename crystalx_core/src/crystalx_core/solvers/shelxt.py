"""SHELXT backend: the first real vertical slice.

SHELXT does dual-space recycling and, unusually among the SHELX programs, **determines the space
group itself** within the Laue group. That is why it is the right default when the symmetry is not
already settled, and why it needs no ``LATT``/``SYMM`` in its input.

This module is a *backend*. Nothing in ``StrategySelector``, ``InitialSolutionBranch`` or the branch
machine knows it exists; adding SHELXS or SHELXD later requires no change to any of them.

**What is evidence-backed and what is configurable.** The invocation ``shelxt <stem>`` and the
primary output ``<stem>_a.res`` are backed by the imported upstream CrystalX code, which calls
``shelxt "$work_dir/$fname"`` and then reads ``{fname}_a.res``
(``inference_repo/scripts/run_demo.sh``, ``crystalx_infer/common/paths.py``). Everything else --
alternative space-group solutions ``<stem>_b.res``, ``<stem>_c.res`` …, and the ``<stem>.lxt``
listing -- is *discovered* rather than assumed: the adapter records every file the program created,
and this backend picks up whatever matches. Nothing is required to exist beyond ``<stem>_a.res``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.domain.chemistry import ChemicalPrior, atomic_number
from crystalx_core.domain.data import DataDiagnosis, HklfFormat, SpaceGroupConfidence
from crystalx_core.domain.hypothesis import (
    EvidenceKind,
    HeavyAtomHypothesis,
    HeavyAtomSite,
    HypothesisEvidence,
)
from crystalx_core.domain.ids import AtomId
from crystalx_core.domain.solving import (
    Applicability,
    ApplicabilityVerdict,
    BranchOutcome,
    BranchState,
    InitialSolutionBranch,
    SolverCapability,
    SolverKind,
    capability_for,
)
from crystalx_core.domain.tools import ToolKind
from crystalx_core.io.shelx_res import ResFile, read_res
from crystalx_core.solvers.context import SolverContext
from crystalx_core.tools.runner import ExecutionOutcome, ExecutionPlan

__all__ = ["ShelxtBackend", "ShelxtSolution", "SHELXT_PRIMARY_SOLUTION_SUFFIX"]

#: The solution SHELXT writes for its best space group. Backed by upstream CrystalX usage.
SHELXT_PRIMARY_SOLUTION_SUFFIX = "_a.res"

#: Alternative space-group solutions, if the installation writes them.
_ALTERNATIVE_RE = re.compile(r"^(?P<stem>.+)_(?P<letter>[a-z])\.res$", re.IGNORECASE)


@dataclass(frozen=True)
class ShelxtSolution:
    """One trial structure SHELXT produced, usually one per candidate space group."""

    letter: str
    res_path: Path
    res: ResFile
    is_primary: bool

    @property
    def n_atoms(self) -> int:
        return self.res.n_atoms

    @property
    def elements(self) -> tuple[str, ...]:
        seen: list[str] = []
        for atom in self.res.atoms:
            element = self.res.element_of(atom).strip().capitalize()
            if element and element not in seen:
                seen.append(element)
        return tuple(seen)

    def space_group_summary(self) -> str:
        header = self.res.header
        parts = []
        if header.latt is not None:
            parts.append(f"LATT {header.latt}")
        parts.append(f"{len(header.symm)} SYMM")
        return ", ".join(parts)


@dataclass
class ShelxtBackend:
    """Runs SHELXT and turns its output into hypotheses the anchor stage can use."""

    context: SolverContext
    timeout_s: float = 900.0
    kind: SolverKind = field(default=SolverKind.SHELXT, init=False)

    # ------------------------------------------------------------------ contract
    def capability(self) -> SolverCapability:
        return capability_for(SolverKind.SHELXT)

    def applicability(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> Applicability:
        """SHELXT is nearly always usable; twinned data are the case where it should be distrusted."""
        reasons = ["Dual-space recycling; the default for routine data."]
        priority = 70
        hklf = diagnosis.ins.hklf_format or diagnosis.reflections.hklf_format
        if hklf in (HklfFormat.HKLF5, HklfFormat.HKLF6):
            priority -= 20
            reasons.append(
                "HKLF 5/6 indicates twinned data; a dual-space solution should be treated as "
                "provisional until a twin law is established."
            )
        if diagnosis.space_group_confidence is SpaceGroupConfidence.UNKNOWN:
            priority += 25
            reasons.append(
                "No space group is declared and SHELXT is the only route that determines one."
            )
        return Applicability(ApplicabilityVerdict.APPLICABLE, priority, tuple(reasons))

    def plan(self, branch: InitialSolutionBranch, *, work_dir: Path) -> tuple[ExecutionPlan, ...]:
        """A single invocation: ``shelxt <stem>`` in the branch's own directory."""
        case = self.context.case
        args = (case.stem, *self.context.args_for(ToolKind.SHELXT))
        executable = self.context.executable_path(ToolKind.SHELXT)
        return (
            ExecutionPlan(
                argv=(executable, *args),
                cwd=work_dir,
                timeout_s=branch.budget.max_wall_clock_s or self.timeout_s,
                expected_outputs=(f"{case.stem}{SHELXT_PRIMARY_SOLUTION_SUFFIX}",),
                declared_inputs=case.declared_inputs,
                label="shelxt",
                tool_kind=ToolKind.SHELXT,
            ),
        )

    def parse(
        self, branch: InitialSolutionBranch, outcomes: tuple[ExecutionOutcome, ...]
    ) -> BranchOutcome:
        """Read every solution SHELXT left behind and build one hypothesis per solution."""
        if not outcomes:
            return BranchOutcome(
                branch_id=branch.branch_id,
                state=BranchState.FAILED,
                summary="SHELXT was never invoked",
                failure_reason="no command was executed",
            )
        outcome = outcomes[-1]
        work_dir = outcome.plan.cwd
        elapsed = sum(o.duration_s for o in outcomes)

        if not outcome.succeeded:
            return BranchOutcome(
                branch_id=branch.branch_id,
                state=BranchState.FAILED,
                summary=outcome.diagnostic.message,
                failure_reason=outcome.diagnostic.message,
                diagnostic_code=outcome.diagnostic.kind.value,
                wall_clock_s=elapsed,
            )

        solutions, read_warnings = self.collect_solutions(work_dir)
        if not solutions:
            return BranchOutcome(
                branch_id=branch.branch_id,
                state=BranchState.FAILED,
                summary=(
                    "SHELXT exited cleanly but no solution file could be read. "
                    "A silent no-op is not a solution."
                ),
                failure_reason="; ".join(read_warnings) or "no *_?.res produced",
                diagnostic_code="no_solution_parsed",
                wall_clock_s=elapsed,
            )

        hypotheses = [self.build_hypothesis(branch, s) for s in solutions]
        primary = solutions[0]
        return BranchOutcome(
            branch_id=branch.branch_id,
            state=BranchState.SUCCEEDED,
            summary=(
                f"SHELXT produced {len(solutions)} solution(s); best has {primary.n_atoms} atoms "
                f"over elements {', '.join(primary.elements) or 'unknown'}"
            ),
            figures_of_merit={
                "n_solutions": float(len(solutions)),
                "n_atoms_primary": float(primary.n_atoms),
                "n_heavy_sites_primary": float(len(hypotheses[0].sites)) if hypotheses else 0.0,
            },
            produced_paths=tuple(str(s.res_path) for s in solutions),
            hypothesis_ids=tuple(h.hypothesis_id for h in hypotheses),
            wall_clock_s=elapsed,
        )

    # ------------------------------------------------------------------ helpers
    def collect_solutions(self, work_dir: Path) -> tuple[list[ShelxtSolution], list[str]]:
        """Find and read every ``<stem>_?.res`` in the working directory, primary first."""
        stem = self.context.case.stem
        warnings: list[str] = []
        found: list[ShelxtSolution] = []

        for path in sorted(work_dir.glob(f"{stem}_*.res")):
            match = _ALTERNATIVE_RE.match(path.name)
            if match is None or match.group("stem") != stem:
                continue
            letter = match.group("letter").lower()
            try:
                res = read_res(path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not lose the others
                warnings.append(f"{path.name}: {exc}")
                continue
            found.append(
                ShelxtSolution(
                    letter=letter,
                    res_path=path,
                    res=res,
                    is_primary=(letter == "a"),
                )
            )

        found.sort(key=lambda s: (not s.is_primary, s.letter))
        return found, warnings

    def build_hypothesis(
        self, branch: InitialSolutionBranch, solution: ShelxtSolution
    ) -> HeavyAtomHypothesis:
        """Extract the heavy-atom sites from a SHELXT solution.

        SHELXT assigns elements from peak heights, which is precisely the step CrystalX replaces, so
        every site here is flagged ``element_is_assumed``. The positions are the useful part; the
        element labels are a starting guess.
        """
        threshold = self.context.heavy_atom_z_threshold
        sites: list[HeavyAtomSite] = []
        for atom in solution.res.atoms:
            element = solution.res.element_of(atom).strip().capitalize()
            try:
                z = atomic_number(element)
            except KeyError:
                continue
            if z < threshold:
                continue
            sites.append(
                HeavyAtomSite(
                    atom_id=AtomId(atom.label),
                    element=element,
                    x=atom.x,
                    y=atom.y,
                    z=atom.z,
                    occupancy=atom.occupancy,
                    u_iso=atom.u_iso,
                    element_is_assumed=True,
                )
            )

        evidence = [
            HypothesisEvidence(
                kind=EvidenceKind.SHELXT_SOLUTION,
                summary=(
                    f"SHELXT solution '{solution.letter}' ({solution.space_group_summary()}) with "
                    f"{solution.n_atoms} atoms"
                ),
                value=float(solution.n_atoms),
                units="atoms",
                source_branch=branch.branch_id,
                artefact_path=str(solution.res_path),
            )
        ]
        disagreements = solution.res.label_element_disagreements()
        if disagreements:
            evidence.append(
                HypothesisEvidence(
                    kind=EvidenceKind.SHELXT_SOLUTION,
                    summary=(
                        f"{len(disagreements)} atom label(s) disagree with their SFAC element, "
                        f"e.g. {disagreements[0][0]}: SFAC says {disagreements[0][1]}, "
                        f"label says {disagreements[0][2]}"
                    ),
                    supports=False,
                    source_branch=branch.branch_id,
                    artefact_path=str(solution.res_path),
                )
            )

        return HeavyAtomHypothesis(
            hypothesis_id=f"{branch.branch_id}-{solution.letter}",
            sites=tuple(sites),
            origin_branch=branch.branch_id,
            evidence=tuple(evidence),
            confidence=None,
            notes=(
                "Element assignments come from SHELXT peak heights and are a starting guess, not "
                "a determination. Positions are the reliable part."
            ),
        )
