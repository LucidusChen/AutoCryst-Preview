"""The native tool inventory: what is installed, what a run would use, and what blocks it.

Selection is delegated to ``crystalx_core``'s resolver with the same locations and the same
strict-native policy a run uses, so this report and an actual run cannot disagree about *which*
file is chosen. On top of the resolver's checks (format, architecture, executable bit, quarantine)
the inventory adds what the resolver does not look at: linked libraries, code signature, minimum
macOS version, PLATON's resources, official-availability evidence and any launch-probe record.

Every tool has four separate status facts, never merged:

* ``present`` -- a candidate file exists;
* ``static`` -- ``ok`` / ``blocked`` / ``not_found`` from reading files only;
* ``launch`` -- ``not_tested`` unless an opt-in launch probe recorded this exact file;
* ``validated`` -- always ``not_verified`` here: only a real run with parsed output can say more.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.domain.tools import BinaryArch, HostPlatform, ResolutionSource, ToolDiagnosticKind, ToolKind
from crystalx_core.tools.registry import PLATON_PREFERENCE, spec_for
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver

from crystalx_native_tools.availability import OfficialAvailability, availability_for
from crystalx_native_tools.config import ToolsConfig
from crystalx_native_tools.host import HostFacts, detect_host
from crystalx_native_tools.inspect import candidate_check_for, FileInspection, inspect_file
from crystalx_native_tools.probe import ProbeStore
from crystalx_native_tools.resources import PlatonResources, inspect_platon_resources

__all__ = ["Blocker", "CandidateReport", "ToolStatus", "Inventory", "build_inventory", "PIPELINE_TOOLS", "INVENTORY_SCHEMA"]

INVENTORY_SCHEMA = "crystalx.native-tools/1"

#: What the current pipeline runs. The remaining SHELX programs belong to plan-only routes.
PIPELINE_TOOLS = ("shelxt", "shelxl", "platon_special", "platon")
ROLES = {
    "shelxt": "initial solution (space-group search)",
    "shelxl": "bounded isotropic refinement",
    "platon_special": "validation (preferred executable)",
    "platon": "validation fallback",
}

_NEXT_STEPS = {
    ToolDiagnosticKind.NOT_FOUND.value: "Install the program, put it in the toolbox directory, or name it in config.local.json.",
    ToolDiagnosticKind.NOT_EXECUTABLE.value: "Make it executable (chmod +x) -- the owner's own file; nothing here changes it.",
    ToolDiagnosticKind.QUARANTINED.value: (
        "macOS will refuse this file until its quarantine attribute is cleared; that is a security "
        "decision for the owner (for example 'xattr -d com.apple.quarantine <file>')."
    ),
}


@dataclass(frozen=True)
class Blocker:
    code: str
    message: str
    next_step: str = ""

    def to_record(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "next_step": self.next_step}


@dataclass(frozen=True)
class CandidateReport:
    path: str
    origin: str
    inspection: FileInspection
    resolver_verdict: str
    resolver_message: str
    native: str
    """``native_arm64``, ``universal_with_arm64``, ``x86_64_only``, ``native_for_host``, ``script``,
    ``foreign_format`` (a macOS executable inspected from another OS), ``not_macho`` or ``missing``."""

    blockers: tuple[Blocker, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return not self.blockers

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "origin": self.origin,
            "native": self.native,
            "resolver_verdict": self.resolver_verdict,
            "resolver_message": self.resolver_message,
            "blockers": [b.to_record() for b in self.blockers],
            "warnings": list(self.warnings),
            "inspection": self.inspection.to_record(),
        }


@dataclass(frozen=True)
class ToolStatus:
    tool: str
    role: str
    used_by_pipeline: bool
    selected: CandidateReport | None
    candidates: tuple[CandidateReport, ...]
    searched: int
    present: bool
    static: str
    launch: dict
    validated: str
    blockers: tuple[Blocker, ...]
    availability: OfficialAvailability | None
    resources: PlatonResources | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "role": self.role,
            "used_by_pipeline": self.used_by_pipeline,
            "present": self.present,
            "static": self.static,
            "launch": self.launch,
            "validated": self.validated,
            "selected": self.selected.to_record() if self.selected else None,
            "candidates": [c.to_record() for c in self.candidates],
            "searched_locations": self.searched,
            "blockers": [b.to_record() for b in self.blockers],
            "availability": self.availability.to_record() if self.availability else None,
            "resources": self.resources.to_record() if self.resources else None,
        }


@dataclass(frozen=True)
class Inventory:
    created_at: str
    host: HostFacts
    strict_native: bool
    config: ToolsConfig
    tools: tuple[ToolStatus, ...]
    validation_tool: str | None
    pipeline: str
    """``ready_static`` (every pipeline step has a statically usable program) or ``blocked``."""

    pipeline_blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def get(self, tool: str) -> ToolStatus | None:
        return next((t for t in self.tools if t.tool == tool), None)

    def to_record(self) -> dict[str, object]:
        return {
            "schema": INVENTORY_SCHEMA,
            "created_at": self.created_at,
            "host": self.host.to_record(),
            "strict_native": self.strict_native,
            "config": self.config.to_record(),
            "pipeline": self.pipeline,
            "pipeline_blockers": list(self.pipeline_blockers),
            "validation_tool": self.validation_tool,
            "tools": [t.to_record() for t in self.tools],
            "notes": list(self.notes),
        }


def _core_host(host: HostFacts) -> HostPlatform:
    arch = {"arm64": BinaryArch.ARM64, "x86_64": BinaryArch.X86_64}.get(host.hardware_arch, BinaryArch.UNKNOWN)
    return HostPlatform(system=host.system, arch=arch, rosetta_available=bool(host.rosetta_markers) if host.system == "Darwin" else None)


def _native_class(inspection: FileInspection, host: HostFacts) -> str:
    if not inspection.exists:
        return "missing"
    if inspection.format == "script":
        return "script"
    if inspection.macho is None:
        return "not_macho"
    arches = set(inspection.arches)
    has_arm = bool(arches & {"arm64", "arm64e"})
    if host.is_apple_silicon:
        if has_arm and len(arches) > 1:
            return "universal_with_arm64"
        if has_arm:
            return "native_arm64"
        if "x86_64" in arches:
            return "x86_64_only"
        return "not_macho"
    if host.system != "Darwin":
        return "foreign_format"  # a macOS executable seen from another operating system
    return "native_for_host" if inspection.run_slice else "not_macho"


def _candidate_report(
    resolver: ToolResolver,
    kind: ToolKind,
    path: Path,
    source: ResolutionSource,
    host: HostFacts,
) -> CandidateReport:
    identity, diagnostic = resolver.inspect_candidate(kind, path, source)
    inspection = inspect_file(path, host)
    native = _native_class(inspection, host)
    blockers: list[Blocker] = []
    warnings: list[str] = []
    if not diagnostic.kind.is_usable:
        blockers.append(
            Blocker(
                code=diagnostic.kind.value,
                message=diagnostic.message,
                next_step=diagnostic.remedy or _NEXT_STEPS.get(diagnostic.kind.value, ""),
            )
        )
    else:
        # Deeper blockers (libraries, signature, minimum macOS, Mach-O structure) come from the same
        # injected candidate check the resolver used, so a blocked candidate is never selected here
        # or in a run (review D5).
        if diagnostic.kind is ToolDiagnosticKind.OK_VIA_TRANSLATION:
            warnings.append(
                "x86_64 only: would run through Rosetta 2. A marker file is present, which is not proof "
                "that x86_64 programs launch on this machine."
            )
    for dep in inspection.dependencies:
        if dep.status == "missing_weak":
            warnings.append(f"optional library {dep.name} is missing")
    warnings.extend(f.message for f in inspection.findings if not f.blocking)
    return CandidateReport(
        path=str(path),
        origin=source.value,
        inspection=inspection,
        resolver_verdict=diagnostic.kind.value,
        resolver_message=diagnostic.message,
        native=native,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def build_inventory(
    config: ToolsConfig | None = None,
    *,
    strict_native: bool | None = None,
    tools: tuple[str, ...] = PIPELINE_TOOLS,
    host: HostFacts | None = None,
    probe_store: ProbeStore | None = None,
    environ: dict[str, str] | None = None,
    search_path: list[Path] | None = None,
) -> Inventory:
    """Inspect every candidate of every requested tool. Reads files only; runs nothing."""
    config = config or ToolsConfig()
    host = host or detect_host()
    strict = config.strict_native if strict_native is None else strict_native
    strict = True if strict is None else strict
    resolver = ToolResolver(
        ResolverConfig(
            explicit_paths=dict(config.tools),
            toolbox_dir=config.toolbox,
            search_path=search_path,
            include_known_locations=config.include_known_locations,
            include_olex2=config.include_olex2,
            extra_dirs=config.search_dirs,
            allow_translation=not strict,
            candidate_check=candidate_check_for(host),
        ),
        host=_core_host(host),
        environ=environ,
    )
    statuses: list[ToolStatus] = []
    for name in tools:
        kind = ToolKind(name)
        searched = resolver.candidate_paths(kind)
        reports = [
            _candidate_report(resolver, kind, path, source, host)
            for path, source in searched
            if path.exists()
        ]
        resolution = resolver.resolve(kind)
        selected = None
        if resolution.identity is not None:
            selected = next((r for r in reports if r.path == str(resolution.identity.path)), None)
        if selected is not None:
            blockers = selected.blockers
            static = "ok" if selected.usable else "blocked"
        elif reports:
            blockers = reports[0].blockers
            static = "blocked"
        else:
            blockers = (Blocker(ToolDiagnosticKind.NOT_FOUND.value, f"{name} was not found in any searched location",
                                _NEXT_STEPS[ToolDiagnosticKind.NOT_FOUND.value]),)
            static = "not_found"
        launch: dict = {"status": "not_tested"}
        if probe_store is not None and selected is not None:
            record = probe_store.latest(name, selected.inspection.sha256)
            if record is not None:
                launch = {"status": record.get("outcome"), "started": record.get("started"),
                          "probed_at": record.get("probed_at"), "message": record.get("message")}
        resources = None
        if name in ("platon", "platon_special"):
            subject = selected or (reports[0] if reports else None)
            resources = inspect_platon_resources(
                subject.inspection if subject else None,
                tool=name,
                configured_checkdef=config.checkdef,
                toolbox=config.toolbox,
                environ=environ,
                on_macos=host.system == "Darwin",
            )
        statuses.append(
            ToolStatus(
                tool=name,
                role=ROLES.get(name, f"{spec_for(kind).purpose} (not run by the current pipeline)"),
                used_by_pipeline=name in PIPELINE_TOOLS,
                selected=selected,
                candidates=tuple(reports),
                searched=len(searched),
                present=bool(reports),
                static=static,
                launch=launch,
                validated="not_verified",
                blockers=tuple(blockers),
                availability=availability_for(name),
                resources=resources,
            )
        )

    by_name = {s.tool: s for s in statuses}
    # Match Toolchain.platon(): the core-usable preferred file is what a real run selects.
    # Deeper inspection can block that choice, but must not advertise an unused fallback.
    validation_tool = next(
        (k.value for k in PLATON_PREFERENCE if k.value in by_name and by_name[k.value].selected is not None), None
    )
    pipeline_blockers = [n for n in ("shelxt", "shelxl") if n in by_name and by_name[n].static != "ok"]
    if any(k in by_name for k in ("platon", "platon_special")) and (
        validation_tool is None or by_name[validation_tool].static != "ok"
    ):
        pipeline_blockers.append("platon_special/platon")
    notes = [
        "Static inspection only: no program was started. 'static ok' is not proof that a program "
        "launches; the opt-in launch probe and an authorised real run are separate evidence.",
    ]
    if strict:
        notes.append("Strict native mode: x86_64-only programs are refused on Apple Silicon (no Rosetta 2); "
                     "scripts and files of unrecognised format are not used, because their own bytes cannot "
                     "show that native code will run.")
    notes.extend(f"configuration: {p}" for p in config.problems)
    return Inventory(
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        host=host,
        strict_native=strict,
        config=config,
        tools=tuple(statuses),
        validation_tool=validation_tool,
        pipeline="ready_static" if not pipeline_blockers else "blocked",
        pipeline_blockers=tuple(pipeline_blockers),
        notes=tuple(notes),
    )
