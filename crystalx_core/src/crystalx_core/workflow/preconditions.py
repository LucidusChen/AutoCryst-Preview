"""Things that must be true before a solve is allowed to start.

A selection rule *ranks*; a precondition *stops*. The difference matters. If the declared space
group is one the material cannot physically occupy, there is no useful sense in which SHELXT is a
better or worse choice than SHELXD -- the question is malformed, and the honest response is to
refuse and say why.

Preconditions are checked once, at Stage 0, against the diagnosis and the chemical prior. Each one
that fails produces a :class:`Precondition` carrying the evidence, the consequence, and a concrete
remedy, and the run does not begin.

Every block here is **overridable in principle** -- the crystallographer knows things the file does
not -- but the override has to be an explicit, recorded act, never a default.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.symmetry import ChiralityCompatibility

__all__ = [
    "PreconditionKind",
    "Precondition",
    "PreconditionReport",
    "check_preconditions",
]


class PreconditionKind(str, enum.Enum):
    """What went wrong. The value is the stable identifier written into the run record."""

    CHIRALITY_IMPOSSIBLE = "chirality_impossible"
    """The declared space group contains an improper operation and the material is homochiral."""

    CHIRALITY_UNRESOLVED_START = "chirality_unresolved_start"
    """The declared space group cannot hold a homochiral structure, but the route that will run
    determines the space group itself within the Laue group (SHELXT). The declaration is recorded as
    an unresolved starting assumption; the group the solver *chooses* is checked after the solve and
    that check stays a hard block. Advisory, not blocking."""

    COMPOSITION_UNUSABLE = "composition_unusable"
    """The instruction file's composition is a pre-solution placeholder and no prior was given, so
    nothing downstream has a composition to work from. Advisory, not blocking: SHELXT can run
    without one."""


@dataclass(frozen=True)
class Precondition:
    """One failed precondition, stated so a crystallographer can agree or disagree with it."""

    kind: PreconditionKind
    blocking: bool
    summary: str
    evidence: str
    remedy: str
    override_hint: str = ""

    def render(self) -> str:
        lines = [f"[{self.kind.value}] {self.summary}", f"  evidence: {self.evidence}", f"  remedy:   {self.remedy}"]
        if self.override_hint:
            lines.append(f"  override: {self.override_hint}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PreconditionReport:
    """The verdict on whether a solve may begin."""

    preconditions: tuple[Precondition, ...] = ()

    @property
    def blockers(self) -> tuple[Precondition, ...]:
        return tuple(p for p in self.preconditions if p.blocking)

    @property
    def advisories(self) -> tuple[Precondition, ...]:
        return tuple(p for p in self.preconditions if not p.blocking)

    @property
    def may_start(self) -> bool:
        return not self.blockers

    def summary(self) -> str:
        if self.may_start:
            extra = f" ({len(self.advisories)} advisory note(s))" if self.advisories else ""
            return f"preconditions satisfied{extra}"
        return "refused to start: " + "; ".join(p.summary for p in self.blockers)


def check_preconditions(
    diagnosis: DataDiagnosis,
    prior: ChemicalPrior,
    *,
    allow_improper_space_group: bool = False,
    space_group_from_solver: bool = False,
) -> PreconditionReport:
    """Decide whether this diagnosis may proceed to a solve.

    ``allow_improper_space_group`` is the recorded override for the chirality block. It exists
    because a crystallographer may legitimately know the preparation was racemic and simply not
    have said so; it defaults to off so that the knowledge has to be supplied, not assumed.

    ``space_group_from_solver`` is route context, not an override: set it only when every route
    that will execute determines the space group itself within the Laue group (SHELXT). Every Laue
    class contains a Sohncke point group, so a centrosymmetric *declaration* does not force a
    centrosymmetric *answer*; the conflict is then recorded as an unresolved starting assumption
    and the solver's chosen group must pass the chirality check afterwards. Fixed-space-group
    routes keep the hard block.
    """
    found: list[Precondition] = []

    symmetry = diagnosis.ins.symmetry()
    enantiopure, why = prior.is_enantiopure()
    conflict = (
        enantiopure
        and symmetry.compatibility is ChiralityCompatibility.CONTAINS_IMPROPER
        and not allow_improper_space_group
    )
    if conflict:
        operations = ", ".join(op.value for op in symmetry.improper_operations)
        evidence = f"{symmetry.explain()} The improper operation(s) present: {operations}. Meanwhile {why}."
        if space_group_from_solver:
            found.append(
                Precondition(
                    kind=PreconditionKind.CHIRALITY_UNRESOLVED_START,
                    blocking=False,
                    summary=(
                        "the declared space group cannot hold a homochiral structure; it is kept "
                        "only as SHELXT's starting Laue group, and the group SHELXT chooses must "
                        "pass the chirality check before anything is refined"
                    ),
                    evidence=evidence,
                    remedy=(
                        "Nothing to do before the solve. If SHELXT also chooses a group with an "
                        "improper operation, the solution is not refined and its Sohncke-compatible "
                        "alternatives are listed for review."
                    ),
                    override_hint="If the preparation really is racemic or meso, declare it.",
                )
            )
        else:
            found.append(
                Precondition(
                    kind=PreconditionKind.CHIRALITY_IMPOSSIBLE,
                    blocking=True,
                    summary=(
                        "the declared space group cannot hold a homochiral structure, so solving in "
                        "it would produce a model that cannot be right"
                    ),
                    evidence=evidence,
                    remedy=(
                        "Let SHELXT determine the space group instead of taking it from the "
                        "instruction file -- it searches within the Laue group and will find the "
                        "Sohncke subgroup if one fits. An automatic pipeline choosing the "
                        "centrosymmetric supergroup is the usual cause."
                    ),
                    override_hint=(
                        "If the preparation really is racemic or meso, declare it: "
                        "ChemicalPrior(enantiopure=False), or pass --racemic."
                    ),
                )
            )

    trust = diagnosis.ins.composition_trust(
        declared_metals=prior.metal_elements, declared_elements=prior.declared_elements
    )
    if not trust.trusted and not prior.metals and not prior.declared_elements:
        found.append(
            Precondition(
                kind=PreconditionKind.COMPOSITION_UNUSABLE,
                blocking=False,
                summary="no usable composition: the file's own is a placeholder and none was declared",
                evidence=trust.detail,
                remedy=(
                    "Declare the expected metal and elements as a chemical prior (--metal Ag). "
                    "SHELXT can run without a composition, but element assignment, charge balance "
                    "and the heavy-atom routes all need one."
                ),
            )
        )

    return PreconditionReport(preconditions=tuple(found))
