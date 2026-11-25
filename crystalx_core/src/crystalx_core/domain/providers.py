"""Optional capabilities the core defines but deliberately does not implement.

`crystalx_core` has no third-party runtime dependency, and a test asserts it. That rule is what lets
the core install and run on a machine with no crystallographic software at all. But three things the
workflow wants genuinely need a symmetry library:

* **Niggli cell reduction**, so two cells can be compared as lattices rather than as six numbers.
* **Space-group tables**, so a group can be *named* rather than only classified.
* **Completeness**, which needs the number of unique reflections symmetry allows.

So the core declares the shape of those capabilities here and implements none of them. A provider is
registered at runtime by the application, never imported by this package.

## The registry never imports anything

`get_symmetry_provider()` returns whatever was registered, or `None`. It does **not** try to import
gemmi, cctbx or anything else. If it did, the zero-dependency test would pass while the package
quietly required a C++ extension in practice. Installing the extra and calling
`crystalx_core.providers.gemmi_provider.register()` is an explicit act.

## Absence is reported, never approximated

Every consumer of these protocols must handle `None` by reporting
:class:`~crystalx_core.domain.data.Unavailable` with a stated reason. A missing provider downgrades
a verdict; it never produces an estimate.

## gemmi does not overrule the core

:mod:`crystalx_core.domain.symmetry` decides Sohncke-vs-improper from the SHELX ``LATT`` sign and
operator determinants, exactly, with no dependency. That stays authoritative for the precondition
block. A provider is a *second opinion*: it can name the group and it can disagree, and a
disagreement is a finding to surface rather than a tie for one side to win silently. See
:class:`ChiralityCrossCheck`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "ReducedCell",
    "SpaceGroupInfo",
    "LatticeRelation",
    "LatticeComparison",
    "ChiralityCrossCheck",
    "CellReducer",
    "SymmetryProvider",
    "register_symmetry_provider",
    "get_symmetry_provider",
    "clear_symmetry_provider",
]


@dataclass(frozen=True)
class ReducedCell:
    """A Niggli-reduced cell, plus how to get back to the input."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float
    change_of_basis: str = ""
    """Operator taking the input cell to this one, as the provider renders it. Recorded so a
    comparison can be re-derived later."""

    @property
    def parameters(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)


@dataclass(frozen=True)
class SpaceGroupInfo:
    """A named space group. Only a provider with the tables can produce one."""

    number: int
    hermann_mauguin: str
    is_sohncke: bool
    is_centrosymmetric: bool
    laue_class: str = ""
    crystal_system: str = ""
    n_operations: int = 0


class LatticeRelation(str, enum.Enum):
    """How two cells are related, once both are reduced."""

    SAME_LATTICE = "same_lattice"
    """The reduced cells agree within tolerance: one is a re-setting of the other."""

    DIFFERENT_LATTICE = "different_lattice"
    """The reduced cells disagree. Equal *volume* does not make two lattices the same, and this is
    the distinction the enum exists to force callers to make."""

    UNDECIDABLE = "undecidable"
    """No reducer is registered, or a cell is missing."""


@dataclass(frozen=True)
class LatticeComparison:
    """The result of comparing a target cell with a reference cell.

    Used by rule ``R08``. Note what this is *not*: it is not a merge-compatibility test. Two
    datasets measured with different radiation can be compared perfectly well and must never be
    merged, so this type shares nothing with
    :func:`crystalx_core.domain.radiation.assess_merge_compatibility`.
    """

    relation: LatticeRelation
    target: ReducedCell | None = None
    reference: ReducedCell | None = None
    volume_ratio: float | None = None
    max_length_deviation_a: float | None = None
    max_angle_deviation_deg: float | None = None
    detail: str = ""

    @property
    def volume_agrees(self) -> bool | None:
        if self.volume_ratio is None:
            return None
        return abs(self.volume_ratio - 1.0) <= 0.02


@dataclass(frozen=True)
class ChiralityCrossCheck:
    """The core's own verdict against a provider's, side by side.

    Agreement is worth recording; disagreement is worth stopping for. Neither is allowed to
    overwrite the other, because the two are derived differently: the core reads the operators in
    the file, the provider looks the group up in a table. If they differ, the file and the table
    disagree about what the file says, and a human should look.
    """

    core_says_sohncke: bool | None
    provider_says_sohncke: bool | None
    provider_name: str | None = None
    space_group: SpaceGroupInfo | None = None

    @property
    def agree(self) -> bool | None:
        if self.core_says_sohncke is None or self.provider_says_sohncke is None:
            return None
        return self.core_says_sohncke == self.provider_says_sohncke

    def explain(self) -> str:
        if self.provider_says_sohncke is None and self.provider_name is None:
            return "No symmetry provider is registered; the core's own analysis stands alone."
        if self.provider_says_sohncke is None:
            return (
                f"{self.provider_name} could not identify this operator set as a tabulated space "
                "group, so there is no second opinion. The core's own analysis stands alone. "
                "An operator set matching no tabulated group is itself worth investigating."
            )
        if self.core_says_sohncke is None:
            return (
                f"The core could not classify the group from the file; {self.provider_name} says "
                f"sohncke={self.provider_says_sohncke}."
            )
        if self.agree:
            return (
                f"The core's operator analysis and {self.provider_name}'s tables agree: "
                f"sohncke={self.core_says_sohncke}."
            )
        return (
            f"DISAGREEMENT: the core reads the file's operators as sohncke="
            f"{self.core_says_sohncke}, but {self.provider_name} says "
            f"{self.provider_says_sohncke}. One of them is misreading the file. Do not proceed on "
            "either verdict until this is resolved."
        )


@runtime_checkable
class CellReducer(Protocol):
    """Niggli reduction, so lattices can be compared rather than cell parameters."""

    def niggli_reduce(
        self, cell: tuple[float, float, float, float, float, float], *, centring: str = "P"
    ) -> ReducedCell:
        """Return the Niggli-reduced cell. Raises ``ValueError`` if reduction does not converge."""
        ...


@runtime_checkable
class SymmetryProvider(Protocol):
    """Space-group tables and the reflection counting that needs them."""

    name: str

    def space_group_by_name(self, symbol: str) -> SpaceGroupInfo | None:
        """Look up a Hermann-Mauguin symbol. ``None`` when the symbol is not recognised."""
        ...

    def space_group_from_operators(
        self, latt: int | None, symm: tuple[str, ...]
    ) -> SpaceGroupInfo | None:
        """Identify the group from SHELX ``LATT``/``SYMM`` records.

        ``None`` when the operator set does not match a tabulated group -- which is itself
        informative, and must not be rounded to the nearest match.
        """
        ...

    def n_unique_reflections(
        self,
        cell: tuple[float, float, float, float, float, float],
        space_group: str,
        d_min: float,
    ) -> int:
        """How many unique reflections symmetry allows to ``d_min``. The denominator of completeness."""
        ...


_registered: SymmetryProvider | None = None


def register_symmetry_provider(provider: SymmetryProvider) -> None:
    """Install a provider. Called by the application, never by this package."""
    global _registered
    _registered = provider


def get_symmetry_provider() -> SymmetryProvider | None:
    """The registered provider, or ``None``.

    Deliberately does not attempt an import. A caller that gets ``None`` must report the capability
    as unavailable with a reason, not estimate around it.
    """
    return _registered


def clear_symmetry_provider() -> None:
    """Unregister. Exists so tests can assert the no-provider path honestly."""
    global _registered
    _registered = None
