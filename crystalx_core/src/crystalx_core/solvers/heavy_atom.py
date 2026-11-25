"""Heavy-atom-first routes: SHELXC/SHELXD/SHELXE, SHELXS Patterson, and ANODE.

These are the routes the product actually needs for silver-containing coordination systems, and they
are defined here as first-class backends -- not as something to add after SHELXT. SHELXT happens to
be implemented first because it is the shortest vertical slice; that is an implementation order, not
a hierarchy.

**Readiness.** Every backend in this module is :attr:`BackendReadiness.PLAN_ONLY`. The command shapes
below are written from the documented usage of each program, but **none has been verified against a
real installation from this repository**, and none of the output parsers is written. The branch
machine therefore refuses to execute them unless a caller explicitly opts in, while ``--dry-run``
still shows exactly what would be run. When each shape is confirmed on the owner's machine and its
parser lands, the readiness flag flips to ``PRODUCTION`` and nothing else has to change.

The three roles stay separate, as they are scientifically separate:

* **SHELXC** prepares experimental-phasing input *and measures the anomalous signal against
  resolution*. That measurement is a Stage-0 decision input in its own right: if the signal is not
  there, SHELXD and SHELXE should never be paid for.
* **SHELXD** searches for the substructure.
* **SHELXE** improves phases, and by running both hands provides one of the strongest available
  tests that a substructure is real.

Autotracing is a separate matter again, and stays off unless the owner declared a peptide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.domain.chemistry import ChemicalPrior, atomic_number
from crystalx_core.domain.data import AnomalousAvailability, DataDiagnosis, SpaceGroupConfidence
from crystalx_core.domain.solving import (
    Applicability,
    ApplicabilityVerdict,
    BackendReadiness,
    BranchOutcome,
    BranchState,
    InitialSolutionBranch,
    SolverCapability,
    SolverKind,
    capability_for,
)
from crystalx_core.domain.tools import ToolKind
from crystalx_core.solvers.context import SolverContext
from crystalx_core.tools.runner import ExecutionOutcome, ExecutionPlan

__all__ = [
    "PlanOnlyBackend",
    "ShelxcShelxdBackend",
    "ShelxeDensityModBackend",
    "ShelxsPattersonBackend",
    "AnodeVerifyBackend",
    "build_shelxc_instructions",
    "heavy_atom_backends",
]


@dataclass
class PlanOnlyBackend:
    """Base for a backend whose command is known but whose parser is not written yet.

    ``parse`` deliberately returns a *failed* outcome with a distinct diagnostic code rather than
    raising or, worse, inventing a result. A run record must never contain a scientific conclusion
    that no code actually derived.
    """

    context: SolverContext
    kind: SolverKind = field(init=False)
    readiness: BackendReadiness = field(default=BackendReadiness.PLAN_ONLY, init=False)

    def capability(self) -> SolverCapability:
        return capability_for(self.kind)

    def parse(
        self, branch: InitialSolutionBranch, outcomes: tuple[ExecutionOutcome, ...]
    ) -> BranchOutcome:
        return BranchOutcome(
            branch_id=branch.branch_id,
            state=BranchState.FAILED,
            summary=(
                f"{self.kind.value} ran, but this build has no parser for its output, so no "
                "scientific conclusion can be drawn from it."
            ),
            failure_reason=(
                "backend is plan-only: the command shape is defined but the output parser is not "
                "implemented and has not been verified against a real installation"
            ),
            diagnostic_code="parser_not_implemented",
            produced_paths=tuple(
                str(Path(o.plan.cwd) / name) for o in outcomes for name in o.produced_outputs
            ),
            wall_clock_s=sum(o.duration_s for o in outcomes) if outcomes else None,
        )


def _no_anomalous_reason(diagnosis: DataDiagnosis) -> str:
    """Say *why* the anomalous routes are unavailable, distinguishing merged from sparse.

    "Friedel mates are merged" is wrong and misleading when the file does contain a couple of
    stray mates -- the crystallographer will go looking for a merging step that never happened.
    """
    reflections = diagnosis.reflections
    if reflections.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_SPARSE:
        return (
            f"Only {reflections.n_friedel_pairs} Friedel pair(s) are present, covering "
            f"{reflections.friedel_coverage:.2%} of the records. That is too few to be unmerged "
            "anomalous data; it looks like duplicated or re-measured reflections in a merged file."
        )
    if reflections.anomalous is AnomalousAvailability.UNKNOWN:
        return "The reflection file could not be scanned for Friedel mates."
    return (
        "Friedel mates are merged in this reflection file, so there is no anomalous difference "
        "to work from."
    )


def build_shelxc_instructions(
    *,
    stem: str,
    hkl_name: str,
    cell: tuple[float, float, float, float, float, float] | None,
    space_group: str,
    find_sites: int,
    heavy_element: str,
    mode: str = "SAD",
    maxm: int = 2,
) -> str:
    """Build the instruction script SHELXC reads on standard input.

    SHELXC is driven by instructions, not flags, which is why the process runner needed a scripted
    stdin mode from the start. The script is stored verbatim in the run record and its digest goes
    into the provenance entry, so the exact question asked of SHELXC is recoverable.
    """
    lines: list[str] = []
    if cell is not None:
        lines.append("CELL " + " ".join(f"{value:.4f}" for value in cell))
    if space_group:
        lines.append(f"SPAG {space_group}")
    lines.append(f"{mode.upper()} {hkl_name}")
    lines.append(f"FIND {int(find_sites)}")
    if heavy_element:
        lines.append(f"SFAC {heavy_element}")
    lines.append(f"MAXM {int(maxm)}")
    return "\n".join(lines) + "\n"


@dataclass
class ShelxcShelxdBackend(PlanOnlyBackend):
    """SHELXC then SHELXD: measure the anomalous signal, then solve the substructure.

    Two invocations, because they answer two different questions and the first can veto the second.

    Draft command shapes::

        shelxc <stem>          # instructions on stdin; writes <stem>_fa.ins and <stem>_fa.hkl
        shelxd <stem>_fa       # writes <stem>_fa.res and <stem>_fa.lst

    Unverified. See the module docstring.
    """

    kind: SolverKind = field(default=SolverKind.SHELXC_SHELXD, init=False)

    def applicability(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> Applicability:
        reasons: list[str] = []
        if diagnosis.reflections.anomalous is not AnomalousAvailability.FRIEDEL_PAIRS_PRESENT:
            reasons.append(_no_anomalous_reason(diagnosis))
        if diagnosis.space_group_confidence is SpaceGroupConfidence.UNKNOWN:
            reasons.append("Experimental phasing needs a declared space group.")
        if reasons:
            return Applicability(ApplicabilityVerdict.NOT_APPLICABLE, reasons=tuple(reasons))
        heavy = self._heaviest(diagnosis, prior)
        priority = 90 if heavy else 60
        return Applicability(
            ApplicabilityVerdict.APPLICABLE,
            priority,
            (
                "Unmerged Friedel pairs are present: SHELXC can measure the anomalous signal and "
                "SHELXD can solve the substructure from it.",
                (
                    f"Heaviest declared scatterer is {heavy}."
                    if heavy
                    else "No heavy scatterer is declared; SHELXC's signal table settles it first."
                ),
            ),
        )

    def plan(self, branch: InitialSolutionBranch, *, work_dir: Path) -> tuple[ExecutionPlan, ...]:
        case = self.context.case
        header = self.context.diagnosis.ins
        cell = None
        if header.cell is not None:
            c = header.cell
            cell = (c.a, c.b, c.c, c.alpha, c.beta, c.gamma)
        heavy = self._heaviest(self.context.diagnosis, self.context.prior) or ""
        script = build_shelxc_instructions(
            stem=case.stem,
            hkl_name=case.hkl_name,
            cell=cell,
            space_group=branch.space_group_assumption,
            find_sites=branch.expected_sites or 4,
            heavy_element=heavy,
        )
        shelxc = ExecutionPlan(
            argv=(self.context.executable_path(ToolKind.SHELXC), case.stem),
            cwd=work_dir,
            timeout_s=min(300.0, branch.budget.max_wall_clock_s),
            stdin_script=script,
            expected_outputs=(f"{case.stem}_fa.ins", f"{case.stem}_fa.hkl"),
            declared_inputs=case.declared_inputs,
            label="shelxc",
            tool_kind=ToolKind.SHELXC,
        )
        trials = branch.budget.solver_trials
        shelxd_args = (f"{case.stem}_fa", *self.context.args_for(ToolKind.SHELXD))
        shelxd = ExecutionPlan(
            argv=(self.context.executable_path(ToolKind.SHELXD), *shelxd_args),
            cwd=work_dir,
            timeout_s=branch.budget.max_wall_clock_s,
            expected_outputs=(f"{case.stem}_fa.res",),
            declared_inputs=(f"{case.stem}_fa.ins", f"{case.stem}_fa.hkl"),
            label=f"shelxd{'' if trials is None else f'_ntry{trials}'}",
            tool_kind=ToolKind.SHELXD,
        )
        return (shelxc, shelxd)

    @staticmethod
    def _heaviest(diagnosis: DataDiagnosis, prior: ChemicalPrior) -> str:
        best: tuple[int, str] | None = None
        for symbol in {*diagnosis.ins.sfac, *prior.metal_elements}:
            try:
                z = atomic_number(symbol)
            except KeyError:
                continue
            if best is None or z > best[0]:
                best = (z, symbol.strip().capitalize())
        return best[1] if best and best[0] >= 20 else ""


@dataclass
class ShelxeDensityModBackend(PlanOnlyBackend):
    """SHELXE for density modification, phase extension and the hand test.

    Two invocations by design: the original hand and the inverted one. Comparing their contrast and
    pseudo-free CC is how a substructure is confirmed, and running only one hand throws that away.

    Draft command shapes::

        shelxe <stem> <stem>_fa -s<solvent> -m<cycles>        # original hand
        shelxe <stem> <stem>_fa -s<solvent> -m<cycles> -i     # inverted hand

    ``-a`` (main-chain autotracing) is appended **only** when the owner declared a peptide-like
    system, because it assumes polypeptide geometry and misleads badly on a metal-organic small
    molecule. Unverified. See the module docstring.
    """

    kind: SolverKind = field(default=SolverKind.SHELXE_DENSITY_MOD, init=False)
    solvent_fraction: float = 0.45
    cycles: int = 20

    def applicability(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> Applicability:
        if diagnosis.space_group_confidence is SpaceGroupConfidence.UNKNOWN:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("Density modification needs a declared space group.",),
            )
        return Applicability(
            ApplicabilityVerdict.APPLICABLE,
            25,
            (
                "Runs after a substructure exists; the original-versus-inverted comparison is one "
                "of the strongest available tests that the substructure is real.",
            ),
        )

    def plan(self, branch: InitialSolutionBranch, *, work_dir: Path) -> tuple[ExecutionPlan, ...]:
        case = self.context.case
        base = [
            f"-s{self.solvent_fraction:g}",
            f"-m{int(self.cycles)}",
            *self.context.args_for(ToolKind.SHELXE),
        ]
        if branch.enable_autotracing:
            if not self.context.prior.is_peptide_like:
                raise ValueError(
                    "Autotracing was requested but the chemical prior does not declare a "
                    "peptide-like system. SHELXE main-chain tracing assumes polypeptide geometry."
                )
            base.insert(0, "-a")
        executable = self.context.executable_path(ToolKind.SHELXE)
        common = dict(
            cwd=work_dir,
            timeout_s=branch.budget.max_wall_clock_s,
            declared_inputs=(f"{case.stem}.hkl", f"{case.stem}_fa.res", f"{case.stem}_fa.hkl"),
            tool_kind=ToolKind.SHELXE,
        )
        original = ExecutionPlan(
            argv=(executable, case.stem, f"{case.stem}_fa", *base),
            expected_outputs=(f"{case.stem}.phs",),
            label="shelxe_original_hand",
            **common,
        )
        inverted = ExecutionPlan(
            argv=(executable, case.stem, f"{case.stem}_fa", *base, "-i"),
            expected_outputs=(f"{case.stem}_i.phs",),
            label="shelxe_inverted_hand",
            **common,
        )
        return (original, inverted)


@dataclass
class ShelxsPattersonBackend(PlanOnlyBackend):
    """SHELXS Patterson interpretation.

    The classical route to a dominant heavy atom when the space group is known and there is no
    usable anomalous signal. For a silver compound measured without keeping Friedel pairs, this is
    the correct answer, not a fallback.

    Draft command shape::

        shelxs <stem>_patt      # <stem>_patt.ins carries the PATT instruction

    The ``.ins`` this consumes is generated by ``crystalx_core.io.shelx_writer``, which is why the
    writer landed in the same slice. Unverified. See the module docstring.
    """

    kind: SolverKind = field(default=SolverKind.SHELXS_PATT, init=False)

    def applicability(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> Applicability:
        if diagnosis.space_group_confidence is SpaceGroupConfidence.UNKNOWN:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("SHELXS needs a declared space group (LATT/SYMM).",),
            )
        heavy = ShelxcShelxdBackend._heaviest(diagnosis, prior)
        if not heavy:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=("Patterson interpretation needs a dominant heavy scatterer.",),
            )
        return Applicability(
            ApplicabilityVerdict.APPLICABLE,
            85,
            (
                f"{heavy} is present and the space group is known: Patterson interpretation is the "
                "classical route to the heavy-atom positions.",
            ),
        )

    def plan(self, branch: InitialSolutionBranch, *, work_dir: Path) -> tuple[ExecutionPlan, ...]:
        case = self.context.case
        stem = f"{case.stem}_patt"
        return (
            ExecutionPlan(
                argv=(self.context.executable_path(ToolKind.SHELXS), stem),
                cwd=work_dir,
                timeout_s=branch.budget.max_wall_clock_s,
                expected_outputs=(f"{stem}.res",),
                declared_inputs=(f"{stem}.ins", f"{stem}.hkl"),
                label="shelxs_patt",
                tool_kind=ToolKind.SHELXS,
            ),
        )


@dataclass
class AnodeVerifyBackend(PlanOnlyBackend):
    """ANODE: anomalous difference density at each proposed site.

    This is the independent check the workflow needs before hard-locking a metal. It answers "is
    there really a silver here" with a measured observable rather than a model score, which is why
    it is exempt from the solver-count cap.

    Draft command shape::

        anode <stem>            # reads <stem>.hkl plus a model; writes anomalous peak output

    Unverified, and the output format in particular still needs checking against a real run before
    a parser is written. See the module docstring.
    """

    kind: SolverKind = field(default=SolverKind.ANODE_VERIFY, init=False)

    def applicability(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> Applicability:
        if diagnosis.reflections.anomalous is not AnomalousAvailability.FRIEDEL_PAIRS_PRESENT:
            return Applicability(
                ApplicabilityVerdict.NOT_APPLICABLE,
                reasons=(
                    "ANODE needs unmerged Friedel pairs to compute anomalous density. "
                    + _no_anomalous_reason(diagnosis),
                ),
            )
        if not ShelxcShelxdBackend._heaviest(diagnosis, prior):
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

    def plan(self, branch: InitialSolutionBranch, *, work_dir: Path) -> tuple[ExecutionPlan, ...]:
        case = self.context.case
        return (
            ExecutionPlan(
                argv=(self.context.executable_path(ToolKind.ANODE), case.stem),
                cwd=work_dir,
                timeout_s=min(300.0, branch.budget.max_wall_clock_s),
                expected_outputs=(),
                declared_inputs=(f"{case.stem}.hkl", f"{case.stem}.res"),
                label="anode",
                tool_kind=ToolKind.ANODE,
            ),
        )


def heavy_atom_backends(context: SolverContext) -> tuple[PlanOnlyBackend, ...]:
    """Every heavy-atom-first backend, ready to register.

    They are all plan-only today. Registering them anyway is deliberate: the selector can nominate
    them, ``--dry-run`` shows the crystallographer exactly what would run, and the run record states
    plainly that the parser is missing -- which is much better than pretending the routes do not
    exist.
    """
    return (
        ShelxcShelxdBackend(context),
        ShelxeDensityModBackend(context),
        ShelxsPattersonBackend(context),
        AnodeVerifyBackend(context),
    )
