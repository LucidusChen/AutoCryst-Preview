"""Request, status and result types for the application service.

These are the contract between ``crystalx_core`` and any frontend -- the Gradio app today, Swift or
PySide6 later. They serialise to plain JSON-compatible dictionaries so that a non-Python client
can read a run directory directly.

Four facts are kept separate on purpose, because collapsing any two of them is how a demo ends up
claiming more than happened:

* **code readiness** -- is the adapter/parser implemented? (``PRODUCTION`` / ``PLAN_ONLY``)
* **tool availability** -- does static inspection say this machine can launch the program? That is
  *not* a launch test, and it is labelled as such.
* **process completion** -- did the run's stages finish, fail, get cancelled or get blocked?
* **scientific eligibility** -- may this model be ranked or called validated? For this milestone the
  answer is always no: geometry and charge-balance checks have no calculator and stay
  ``NOT_EVALUATED``, so a model can be *displayed and exported* but never ranked or certified.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.domain.chemistry import ATOMIC_NUMBERS, ChemicalPrior, MetalPrior
from crystalx_core.domain.tools import ToolKind
from crystalx_core.tools.runner import ResourceLimits

__all__ = [
    "RunMode",
    "RunPhase",
    "StageName",
    "StageState",
    "ModelAvailability",
    "RefinementStatus",
    "ValidationIssue",
    "ChemicalPriorInput",
    "RunConfig",
    "SolveRequest",
    "STAGE_ORDER",
    "TERMINAL_PHASES",
    "RUN_LIMITS",
]


class RunMode(str, enum.Enum):
    REAL = "real"
    """Real, locally installed programs."""

    MOCK_DEMO = "mock_demo"
    """Synthetic stand-ins on the built-in synthetic demo case only. Never applied to user data."""


class RunPhase(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    """All stages that could run have finished. Says nothing about whether the model is right."""

    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    """A precondition refused the run before any program started."""

    INTERRUPTED = "interrupted"
    """The run was in progress when the service stopped. Its record is kept as it was left."""


TERMINAL_PHASES = frozenset(
    {RunPhase.COMPLETED, RunPhase.FAILED, RunPhase.CANCELLED, RunPhase.BLOCKED, RunPhase.INTERRUPTED}
)


class StageName(str, enum.Enum):
    STAGE_INPUTS = "stage_inputs"
    DIAGNOSE = "diagnose"
    PRECONDITIONS = "preconditions"
    PLAN = "plan"
    SHELXT = "shelxt"
    SHELXL = "shelxl"
    PLATON = "platon"
    VIEWER = "viewer"


STAGE_ORDER = tuple(StageName)


class StageState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Did not run because an earlier stage produced nothing for it to work on."""

    BLOCKED = "blocked"
    """Did not run because a stated prerequisite is missing."""

    TOOL_UNAVAILABLE = "tool_unavailable"
    """Did not run because the program cannot be launched on this machine."""

    CANCELLED = "cancelled"


class ModelAvailability(str, enum.Enum):
    NONE = "none"
    SOLUTION_MODEL = "solution_model"
    """A SHELXT trial structure. Unrefined."""

    PRELIMINARY_REFINED_MODEL = "preliminary_refined_model"
    """After a bounded, isotropic SHELXL refinement. Still unvalidated."""


class RefinementStatus(str, enum.Enum):
    NOT_RUN = "not_run"
    FAILED = "failed"
    """The program failed, or produced no readable model."""

    ERRORS_REPORTED = "errors_reported"
    """SHELXL wrote error-marked (``** ... **``) lines. Until real output has verified which of these
    are fatal, the refined model is not offered; nonfatal ``Warning`` lines are reported separately."""

    RAN_CONVERGENCE_NOT_ASSESSED = "ran_convergence_not_assessed"
    """The cycles ran and a model was read. Convergence is not judged: the raw mean and maximum
    shift/su are reported instead (a small mean can hide a large maximum)."""


@dataclass(frozen=True)
class ValidationIssue:
    """A user-input problem. ``code`` is stable so a frontend can translate it."""

    code: str
    field: str
    message: str


_ELEMENT = re.compile(r"^[A-Z][a-z]?$")


def _normalise_element(token: str) -> str:
    token = token.strip()
    return token[:1].upper() + token[1:].lower() if token else token


@dataclass(frozen=True)
class ChemicalPriorInput:
    """What the crystallographer tells the run. Converted to :class:`ChemicalPrior` after checks."""

    metals: tuple[str, ...] = ()
    elements: tuple[str, ...] = ()
    peptide: bool = False
    racemic: bool = False
    composition: tuple[tuple[str, float], ...] = ()
    """Optional ``(element, count per cell)`` pairs replacing the file's SFAC/UNIT for the solver.
    Empty means "use the file's own". Recorded as an override when used."""

    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for field_name, values in (("metals", self.metals), ("elements", self.elements)):
            for token in values:
                symbol = _normalise_element(token)
                if not _ELEMENT.match(symbol) or symbol not in ATOMIC_NUMBERS:
                    issues.append(
                        ValidationIssue(
                            "unknown_element", field_name, f"{token!r} is not an element symbol"
                        )
                    )
        # peptide + racemic is a legitimate declaration (racemic protein crystallography exists); it
        # is resolved by ChemicalPrior.is_enantiopure(), not refused here.
        seen: set[str] = set()
        for symbol, count in self.composition:
            normal = _normalise_element(symbol)
            if normal not in ATOMIC_NUMBERS:
                issues.append(
                    ValidationIssue("unknown_element", "composition", f"{symbol!r} is not an element")
                )
            if not count > 0:
                issues.append(
                    ValidationIssue(
                        "non_positive_count", "composition", f"{symbol}: count must be positive"
                    )
                )
            if normal in seen:
                issues.append(
                    ValidationIssue("duplicate_element", "composition", f"{symbol} listed twice")
                )
            seen.add(normal)
        return issues

    def to_prior(self) -> ChemicalPrior:
        metals = tuple(MetalPrior(element=_normalise_element(m)) for m in self.metals)
        return ChemicalPrior(
            metals=metals,
            declared_elements=tuple(_normalise_element(e) for e in self.elements),
            is_peptide_like=self.peptide,
            enantiopure=False if self.racemic else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metals": list(self.metals),
            "elements": list(self.elements),
            "peptide": self.peptide,
            "racemic": self.racemic,
            "composition": [[s, c] for s, c in self.composition],
        }

    @classmethod
    def parse_composition(cls, text: str) -> tuple[tuple[str, float], ...]:
        """Parse ``"C 40 H 52 N 4 O 8 Zn 2"`` into pairs. Raises ``ValueError`` with a reason."""
        tokens = text.replace(",", " ").split()
        if not tokens:
            return ()
        if len(tokens) % 2:
            raise ValueError("composition needs element/count pairs, e.g. 'C 40 H 52 Zn 2'")
        pairs = []
        for symbol, count in zip(tokens[0::2], tokens[1::2]):
            try:
                pairs.append((_normalise_element(symbol), float(count)))
            except ValueError as exc:
                raise ValueError(f"{count!r} is not a number (after {symbol})") from exc
        return tuple(pairs)


#: Limits shared by every program in a run. CPU time is set per program, see
#: :meth:`RunConfig.limits_for`.
RUN_LIMITS = ResourceLimits(
    cpu_seconds=None,
    max_file_bytes=2 * 1024 * 1024 * 1024,
    max_open_files=1024,
    nice_increment=5,
)


@dataclass(frozen=True)
class RunConfig:
    """Bounds and locations for one run. Every value is written into the run record."""

    shelxt_timeout_s: float = 3600.0
    shelxl_cycles: int = 4
    shelxl_timeout_s: float = 900.0
    platon_timeout_s: float = 600.0
    limits: ResourceLimits = RUN_LIMITS
    cpu_seconds: int | None = None
    """Explicit CPU-time cap per program. ``None`` derives it from each program's wall-clock
    budget times the number of logical CPUs: SHELXT is multi-threaded and ``RLIMIT_CPU`` counts
    the CPU time of all threads, so a cap equal to the wall-clock budget would stop a parallel
    run long before its time was up.

    This is a generous *budget*, not a limit on the number of threads, and ``RLIMIT_CPU`` is per
    process: a child process inherits its own copy, so it is not an aggregate quota over a process
    tree. The wall-clock timeout and process-group cleanup are what bound a run as a whole. A true
    concurrency cap would need the program's own threading control."""

    toolbox_dir: Path | None = None
    tool_paths: tuple[tuple[str, Path], ...] = ()
    """Explicit ``(tool, executable)`` pairs, e.g. from the local tool configuration. They take
    precedence over every searched location; tool names are :class:`ToolKind` values."""

    extra_tool_dirs: tuple[Path, ...] = ()
    """Further directories to search after the conventional ones."""

    strict_native: bool = False
    """Strict native (no-Rosetta) mode: x86_64-only executables are refused on Apple Silicon and
    the search continues to native candidates. Off by default for the command line; the Web
    interface turns it on by default."""

    include_olex2: bool = True
    """Search an Olex2 installation for the programs. The local tool configuration can turn this
    off; the run must then not find a program there that the tool panel never showed."""

    include_known_locations: bool = True
    """Search the conventional install directories (``/usr/local/bin`` and the like)."""

    mock_step_seconds: float = 0.6
    """Simulated duration of each mock program; demo mode only."""

    def __post_init__(self) -> None:
        for name in ("shelxt_timeout_s", "shelxl_timeout_s", "platon_timeout_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"RunConfig.{name} must be positive")
        if not 1 <= self.shelxl_cycles <= 50:
            raise ValueError("RunConfig.shelxl_cycles must be between 1 and 50")
        if self.cpu_seconds is not None and self.cpu_seconds <= 0:
            raise ValueError("RunConfig.cpu_seconds must be positive")
        if self.toolbox_dir is not None:
            # Absolute now, against the caller's working directory: every program runs in its own
            # run directory, so a relative toolbox would point somewhere else by then. Symlinks are
            # kept as given; the resolver records the real path separately.
            toolbox = Path(self.toolbox_dir).expanduser()
            object.__setattr__(self, "toolbox_dir", Path(os.path.abspath(toolbox)))
        known = {kind.value for kind in ToolKind}
        pairs = []
        for name, path in self.tool_paths:
            if name not in known:
                raise ValueError(f"RunConfig.tool_paths: unknown tool {name!r}")
            pairs.append((name, Path(os.path.abspath(Path(path).expanduser()))))
        object.__setattr__(self, "tool_paths", tuple(pairs))
        object.__setattr__(
            self,
            "extra_tool_dirs",
            tuple(Path(os.path.abspath(Path(d).expanduser())) for d in self.extra_tool_dirs),
        )

    def limits_for(self, timeout_s: float) -> ResourceLimits:
        """The limits for one program with the given wall-clock budget."""
        cpu = self.cpu_seconds or int(math.ceil(timeout_s * (os.cpu_count() or 1)))
        return dataclasses.replace(self.limits, cpu_seconds=cpu)

    def to_dict(self) -> dict[str, object]:
        return {
            "shelxt_timeout_s": self.shelxt_timeout_s,
            "shelxl_cycles": self.shelxl_cycles,
            "shelxl_timeout_s": self.shelxl_timeout_s,
            "platon_timeout_s": self.platon_timeout_s,
            "limits": self.limits.to_record(),
            "cpu_seconds": (
                self.cpu_seconds
                if self.cpu_seconds is not None
                else f"per program: wall-clock budget x {os.cpu_count() or 1} logical CPUs"
            ),
            "toolbox_dir": str(self.toolbox_dir) if self.toolbox_dir else None,
            "tool_paths": {name: str(path) for name, path in self.tool_paths},
            "extra_tool_dirs": [str(d) for d in self.extra_tool_dirs],
            "strict_native": self.strict_native,
            "include_olex2": self.include_olex2,
            "include_known_locations": self.include_known_locations,
        }


@dataclass(frozen=True)
class SolveRequest:
    ins_path: Path
    hkl_path: Path
    prior: ChemicalPriorInput = field(default_factory=ChemicalPriorInput)
    config: RunConfig = field(default_factory=RunConfig)
    mode: RunMode = RunMode.REAL
    label: str = "solve"

    def validate(self) -> list[ValidationIssue]:
        issues = list(self.prior.validate())
        for field_name, path, suffixes in (
            ("ins", self.ins_path, (".ins", ".res")),
            ("hkl", self.hkl_path, (".hkl",)),
        ):
            if not Path(path).is_file():
                issues.append(ValidationIssue("missing_file", field_name, f"{path} does not exist"))
            elif Path(path).suffix.lower() not in suffixes:
                issues.append(
                    ValidationIssue(
                        "wrong_extension",
                        field_name,
                        f"{Path(path).name} should end in {' or '.join(suffixes)}",
                    )
                )
        return issues
