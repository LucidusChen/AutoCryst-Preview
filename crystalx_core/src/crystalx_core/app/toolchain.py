"""Which programs a run may use, and how they are invoked.

A :class:`Toolchain` is the only thing that differs between a real run and the application demo.
The pipeline asks it for tool availability and for adapters; it never constructs an adapter itself,
so a demo run cannot accidentally reach a real binary and a real run cannot accidentally get a
mock.

Availability here is **static inspection**: header bytes, executable bit, quarantine attribute and
host architecture. It is not a launch test. The first actual invocation is the evidence that a
program starts, and its diagnosis (for example ``arch_mismatch`` from "Bad CPU type in
executable") is reported by the stage that tried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from crystalx_core.app.models import RunConfig, RunMode
from crystalx_core.domain.tools import ToolAvailabilityReport, ToolDiagnostic, ToolKind, ToolResolution
from crystalx_core.errors import ToolResolutionError
from crystalx_core.tools.adapter import ExternalToolAdapter
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.tools.runner import ProcessRunner

__all__ = ["Toolchain", "RealToolchain", "PIPELINE_TOOLS"]

#: The programs this pipeline can use. Anything else is reported but never invoked.
PIPELINE_TOOLS = (
    ToolKind.SHELXT,
    ToolKind.SHELXL,
    ToolKind.PLATON_SPECIAL,
    ToolKind.PLATON,
    ToolKind.SHELXS,
    ToolKind.SHELXC,
    ToolKind.SHELXD,
    ToolKind.SHELXE,
    ToolKind.ANODE,
)


class Toolchain:
    """Base class. Subclasses provide :meth:`availability` and :meth:`_runner`."""

    mode: RunMode = RunMode.REAL

    def availability(self) -> ToolAvailabilityReport:  # pragma: no cover - abstract
        raise NotImplementedError

    def _runner(self) -> ProcessRunner | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def resolution(self, kind: ToolKind) -> ToolResolution | None:
        return self.availability().get(kind)

    def adapter(self, kind: ToolKind) -> ExternalToolAdapter | None:
        """An adapter for a statically usable tool, or ``None`` with the reason in the resolution."""
        resolution = self.resolution(kind)
        if resolution is None or not resolution.is_usable:
            return None
        try:
            return ExternalToolAdapter(resolution, runner=self._runner())
        except ToolResolutionError:  # pragma: no cover - guarded by is_usable above
            return None

    def platon(self) -> ToolResolution | None:
        """Prefer ``platon_special``; fall back when core resolution refuses it.

        A special build may still link X11 libraries; its name is not dependency evidence.
        """
        report = self.availability()
        last = None
        for kind in (ToolKind.PLATON_SPECIAL, ToolKind.PLATON):
            resolution = report.get(kind)
            if resolution is not None and resolution.is_usable:
                return resolution
            last = resolution or last
        return last

    def platon_adapter(self) -> ExternalToolAdapter | None:
        resolution = self.platon()
        if resolution is None or not resolution.is_usable:
            return None
        return self.adapter(resolution.kind)


@dataclass
class RealToolchain(Toolchain):
    """The user's own, locally installed programs, found by :class:`ToolResolver`."""

    toolbox_dir: Path | None = None
    runner: ProcessRunner | None = None
    resolver: ToolResolver | None = None
    explicit_paths: tuple[tuple[str, Path], ...] = ()
    extra_dirs: tuple[Path, ...] = ()
    allow_translation: bool = True
    """``False`` is strict native mode (see :class:`ResolverConfig.allow_translation`)."""

    include_olex2: bool = True
    include_known_locations: bool = True
    candidate_check: Callable[[ToolKind, Path], ToolDiagnostic | None] | None = None
    """An injected deeper check (see :class:`ResolverConfig.candidate_check`)."""

    _report: ToolAvailabilityReport | None = field(default=None, init=False, repr=False)

    mode = RunMode.REAL

    def __post_init__(self) -> None:
        if self.resolver is None:
            self.resolver = ToolResolver(
                ResolverConfig(
                    explicit_paths=dict(self.explicit_paths),
                    toolbox_dir=self.toolbox_dir,
                    extra_dirs=tuple(self.extra_dirs),
                    allow_translation=self.allow_translation,
                    include_olex2=self.include_olex2,
                    include_known_locations=self.include_known_locations,
                    candidate_check=self.candidate_check,
                )
            )

    @classmethod
    def for_config(cls, config: RunConfig, *, candidate_check=None) -> "RealToolchain":
        """The toolchain a run with this configuration uses: same locations, same policy."""
        return cls(
            toolbox_dir=config.toolbox_dir,
            explicit_paths=tuple(config.tool_paths),
            extra_dirs=tuple(config.extra_tool_dirs),
            allow_translation=not config.strict_native,
            include_olex2=config.include_olex2,
            include_known_locations=config.include_known_locations,
            candidate_check=candidate_check,
        )

    def availability(self) -> ToolAvailabilityReport:
        if self._report is None:
            assert self.resolver is not None
            self._report = self.resolver.resolve_all(PIPELINE_TOOLS)
        return self._report

    def _runner(self) -> ProcessRunner | None:
        return self.runner
