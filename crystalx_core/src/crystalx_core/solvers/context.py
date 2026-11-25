"""What a real backend needs in order to build a command.

Kept in its own module so that no solver-specific idea leaks into the core protocol. A backend is
handed a :class:`SolverContext` at construction; the protocol methods still take only a branch and a
working directory, which is what keeps ``StrategySelector`` and the branch machine solver-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.domain.tools import ToolKind
from crystalx_core.errors import ConfigError
from crystalx_core.tools.adapter import ExternalToolAdapter

__all__ = ["SolverContext", "StagedCase"]


@dataclass(frozen=True)
class StagedCase:
    """The file names a solver will see inside its own working directory.

    SHELX programs take a *stem*, not a filename, and read ``<stem>.ins`` and ``<stem>.hkl`` from the
    current directory. Staging therefore means copying the run's inputs into the branch directory
    under a predictable stem; the user's originals stay untouched in ``inputs/``.
    """

    stem: str
    ins_name: str = ""
    hkl_name: str = ""

    def __post_init__(self) -> None:
        if not self.stem or any(ch in self.stem for ch in "/\\ "):
            raise ConfigError(f"Invalid SHELX stem {self.stem!r}: must be a bare, space-free name")
        object.__setattr__(self, "ins_name", self.ins_name or f"{self.stem}.ins")
        object.__setattr__(self, "hkl_name", self.hkl_name or f"{self.stem}.hkl")

    @property
    def declared_inputs(self) -> tuple[str, ...]:
        return (self.ins_name, self.hkl_name)


@dataclass(frozen=True)
class SolverContext:
    """Everything a backend needs that is not part of the branch plan itself."""

    case: StagedCase
    diagnosis: DataDiagnosis
    prior: ChemicalPrior
    radiation: RadiationAssumption
    adapters: Mapping[ToolKind, ExternalToolAdapter] = field(default_factory=dict)
    heavy_atom_z_threshold: int = 20
    extra_args: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Per-tool extra command-line arguments, keyed by ``ToolKind.value``. An escape hatch for the
    crystallographer, recorded in provenance like anything else."""

    def adapter(self, kind: ToolKind) -> ExternalToolAdapter:
        try:
            return self.adapters[kind]
        except KeyError as exc:
            raise ConfigError(
                f"No adapter for {kind.value}. Resolve the tool first; a backend must not "
                "invent a path to an executable."
            ) from exc

    def has_adapter(self, kind: ToolKind) -> bool:
        return kind in self.adapters

    def args_for(self, kind: ToolKind) -> tuple[str, ...]:
        return tuple(self.extra_args.get(kind.value, ()))

    def executable_path(self, kind: ToolKind) -> str:
        """Path of the resolved executable, or the bare program name when only planning.

        Planning must work without a resolved tool so that ``--dry-run`` can show a user what would
        be executed on a machine where the program is not installed.
        """
        if self.has_adapter(kind):
            return str(self.adapter(kind).identity.path)
        return kind.value
