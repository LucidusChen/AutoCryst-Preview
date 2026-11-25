"""Static inspection of one executable file: identity, format, slices, libraries, signature.

Reads the file; never runs it. Library presence is only checked on macOS, where the paths mean
something: on any other host every library is reported ``not_checked`` rather than "missing".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from crystalx_core.tools.xattr import QUARANTINE_XATTR, has_xattr
from crystalx_core.workspace.hashing import sha256_file

from crystalx_native_tools.host import HostFacts
from crystalx_native_tools.macho import MachOFile, MachOSlice, read_macho

__all__ = ["DependencyStatus", "Finding", "FileInspection", "inspect_file", "candidate_check_for",
           "SYSTEM_LIBRARY_PREFIXES", "DEEP_BLOCKING_CODES"]

#: Libraries macOS provides from the dyld shared cache; since macOS 11 they are not files on disk.
SYSTEM_LIBRARY_PREFIXES = ("/usr/lib/", "/System/Library/")
_REQUIRED_KINDS = ("load", "reexport", "upward", "lazy")


@dataclass(frozen=True)
class Finding:
    """One fact found by inspection. ``code`` matches a core ``ToolDiagnosticKind`` value when it blocks."""

    code: str
    message: str
    blocking: bool = True


#: Codes the injected candidate check reports to the core resolver (the core checks the rest itself:
#: presence, regular file, executable bit, quarantine, architecture and strict-native policy).
DEEP_BLOCKING_CODES = ("malformed_binary", "not_an_executable", "wrong_platform", "unsigned_arm64",
                       "requires_newer_macos", "library_unavailable")

_REMEDIES = {
    "malformed_binary": "The file is damaged or incomplete; obtain a fresh copy of the program.",
    "not_an_executable": "Point the configuration at the program itself, not at a library or bundle.",
    "wrong_platform": "Use the macOS build of this program.",
    "unsigned_arm64": "Re-sign the owner's build (for example ad hoc) or rebuild it; macOS will not run it as it is.",
    "requires_newer_macos": "Use a build made for this macOS version, or update macOS.",
    "library_unavailable": (
        "Restore the library the program was linked against (for a Homebrew gfortran build: the gcc "
        "package; for an X11 PLATON build: XQuartz), or rebuild the program against what is installed."
    ),
}


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    kind: str
    status: str
    """``system_cache``, ``present``, ``missing``, ``missing_weak``, ``unresolved`` or ``not_checked``."""

    resolved: str | None = None
    arch_ok: bool | None = None
    note: str = ""

    @property
    def blocks_launch(self) -> bool:
        return self.status in ("missing", "unresolved") or self.arch_ok is False

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "status": self.status, "resolved": self.resolved,
                "arch_ok": self.arch_ok, "note": self.note}


@dataclass(frozen=True)
class FileInspection:
    path: str
    exists: bool
    realpath: str | None = None
    regular: bool = False
    executable: bool = False
    size: int | None = None
    mtime: float | None = None
    sha256: str | None = None
    quarantined: bool | None = None
    format: str = "missing"
    """``mach_o``, ``mach_o_fat``, ``script``, ``elf``, ``pe``, ``other``, ``missing``, ``unreadable``."""

    arches: tuple[str, ...] = ()
    run_slice: str | None = None
    """The slice this host would run natively (``arm64`` on Apple Silicon), if any."""

    min_os: str | None = None
    code_signature: dict | None = None
    dependencies: tuple[DependencyStatus, ...] = ()
    macho: MachOFile | None = field(default=None, repr=False)
    problems: tuple[str, ...] = ()
    findings: tuple[Finding, ...] = ()

    def first_blocking(self, codes: tuple[str, ...] = DEEP_BLOCKING_CODES) -> Finding | None:
        return next((f for f in self.findings if f.blocking and f.code in codes), None)

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "realpath": self.realpath,
            "regular": self.regular,
            "executable": self.executable,
            "size": self.size,
            "mtime": self.mtime,
            "sha256": self.sha256,
            "quarantined": self.quarantined,
            "format": self.format,
            "arches": list(self.arches),
            "run_slice": self.run_slice,
            "min_os": self.min_os,
            "code_signature": self.code_signature,
            "dependencies": [d.to_record() for d in self.dependencies],
            "problems": list(self.problems),
            "findings": [{"code": f.code, "message": f.message, "blocking": f.blocking} for f in self.findings],
        }


def _format_of(head: bytes, macho: MachOFile | None) -> str:
    if macho is not None:
        return "mach_o_fat" if macho.kind == "fat" else "mach_o"
    if head.startswith(b"#!"):
        return "script"
    if head.startswith(b"\x7fELF"):
        return "elf"
    if head.startswith(b"MZ"):
        return "pe"
    return "other"


def _version_tuple(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    try:
        return tuple(int(part) for part in text.split("."))
    except ValueError:
        return None


def _expand(name: str, executable_dir: Path) -> Path | None:
    for token in ("@executable_path", "@loader_path"):
        if name.startswith(token):
            return Path(os.path.normpath(str(executable_dir) + name[len(token):]))
    if name.startswith("/"):
        return Path(name)
    return None


def resolve_dependency(
    name: str,
    kind: str,
    *,
    executable_dir: Path,
    rpaths: tuple[str, ...],
    arch: str,
    on_macos: bool,
    exists: Callable[[Path], bool] = Path.is_file,
    read: Callable[[Path], MachOFile | None] = read_macho,
) -> DependencyStatus:
    """Where dyld would find ``name``, and whether it is there with a usable architecture."""
    if not on_macos:
        return DependencyStatus(name, kind, "not_checked", note="library paths are only checked on macOS")
    if name.startswith(SYSTEM_LIBRARY_PREFIXES):
        return DependencyStatus(
            name, kind, "system_cache",
            note="provided by macOS (dyld shared cache); not a file on disk since macOS 11",
        )
    candidates: list[Path] = []
    if name.startswith("@rpath/"):
        for rpath in rpaths:
            base = _expand(rpath, executable_dir)
            if base is not None:
                candidates.append(Path(os.path.normpath(str(base / name[len("@rpath/"):]))))
        if not candidates:
            return DependencyStatus(name, kind, "missing_weak" if kind == "weak" else "unresolved",
                                    note="@rpath library but no usable LC_RPATH entry")
    else:
        expanded = _expand(name, executable_dir)
        if expanded is None:
            return DependencyStatus(name, kind, "missing_weak" if kind == "weak" else "unresolved",
                                    note="relative install name")
        candidates.append(expanded)
    for candidate in candidates:
        if exists(candidate):
            library = read(candidate)
            arch_ok = None if library is None else library.slice_for(arch) is not None
            note = "" if arch_ok is not False else f"present but has no {arch} code"
            return DependencyStatus(name, kind, "present", resolved=str(candidate), arch_ok=arch_ok, note=note)
    status = "missing_weak" if kind == "weak" else "missing"
    return DependencyStatus(
        name, kind, status, resolved=str(candidates[0]),
        note="looked in: " + ", ".join(str(c) for c in candidates[:4]),
    )


def _run_slice(macho: MachOFile, host: HostFacts) -> MachOSlice | None:
    if host.system == "Darwin":
        return macho.slice_for(host.hardware_arch)
    return None


def inspect_file(path: Path | str, host: HostFacts, *, check_dependencies: bool = True) -> FileInspection:
    target = Path(path)
    if not target.exists():
        return FileInspection(path=str(target), exists=False, format="missing")
    findings: list[Finding] = []
    regular = target.is_file()
    executable = regular and os.access(target, os.X_OK)
    try:
        stat = target.stat()
        digest = sha256_file(target) if regular else None
        with open(target, "rb") as handle:
            head = handle.read(64)
    except OSError as exc:
        return FileInspection(path=str(target), exists=True, regular=regular, format="unreadable",
                              problems=(f"cannot read: {exc}",), findings=(Finding("unreadable", f"cannot read: {exc}"),))
    macho = read_macho(target) if regular else None
    fmt = _format_of(head, macho)
    if not regular:
        findings.append(Finding("not_a_file", "not a regular file"))
    elif not executable:
        findings.append(Finding("not_executable", "not marked executable"))
    quarantined = has_xattr(target, QUARANTINE_XATTR) if host.system == "Darwin" else None
    if quarantined:
        findings.append(Finding("quarantined", "carries the macOS quarantine attribute"))

    arches: tuple[str, ...] = ()
    run_slice_name = min_os = None
    signature = None
    dependencies: tuple[DependencyStatus, ...] = ()
    if macho is not None:
        arches = macho.arches
        findings.extend(Finding("malformed_binary", p) for p in macho.problems)
        chosen = _run_slice(macho, host)
        # Report the slice that would run, or (on other hosts / translation) the first slice.
        described = chosen or (macho.slices[0] if macho.slices else None)
        if chosen is not None:
            run_slice_name = chosen.arch
        if described is not None:
            findings.extend(Finding("malformed_binary", f"{described.arch}: {p}") for p in described.problems)
            if described.filetype and described.filetype != "execute":
                findings.append(Finding("not_an_executable",
                                        f"{described.arch}: a Mach-O {described.filetype}, not a main executable"))
            elif not described.problems and not described.has_entry_point:
                findings.append(Finding("malformed_binary",
                                        f"{described.arch}: no entry point (neither LC_MAIN nor LC_UNIXTHREAD)"))
            if described.platform is not None and described.platform != "macos":
                findings.append(Finding("wrong_platform", f"{described.arch}: built for {described.platform}, not macOS"))
            min_os = described.min_os
            sig = described.code_signature
            signature = {
                "slice": described.arch,
                "present": sig.present,
                "container_readable": sig.present and not sig.note,
                "cryptographically_verified": False,
                "adhoc": sig.adhoc,
                "linker_signed": sig.linker_signed,
                "identifier": sig.identifier,
                "team_id": sig.team_id,
                "note": sig.note,
            }
            if described.is_arm64 and not sig.present:
                findings.append(Finding("unsigned_arm64", "arm64 code without a code signature: macOS kills it at launch"))
            elif described.is_arm64 and sig.note:
                findings.append(Finding("unsigned_arm64",
                                        f"arm64 code signature container is unreadable ({sig.note}): macOS refuses it"))
            elif sig.note:
                findings.append(Finding("signature_unreadable", f"signature container: {sig.note}", blocking=False))
            host_version = _version_tuple(host.macos_version)
            needed = _version_tuple(min_os)
            if host.system == "Darwin" and host_version and needed and needed > host_version:
                findings.append(Finding("requires_newer_macos",
                                        f"built for macOS {min_os} or later; this Mac runs {host.macos_version}"))
            if check_dependencies:
                executable_dir = Path(os.path.realpath(target)).parent
                dependencies = tuple(
                    resolve_dependency(
                        lib.name,
                        lib.kind,
                        executable_dir=executable_dir,
                        rpaths=described.rpaths,
                        arch=described.arch,
                        on_macos=host.system == "Darwin",
                    )
                    for lib in described.libraries
                )
                for dep in dependencies:
                    if dep.blocks_launch:
                        findings.append(Finding("library_unavailable", f"library {dep.name}: {dep.status}"
                                                + (f" ({dep.note})" if dep.note else "")))

    return FileInspection(
        path=str(target),
        exists=True,
        realpath=os.path.realpath(target),
        regular=regular,
        executable=executable,
        size=stat.st_size,
        mtime=stat.st_mtime,
        sha256=digest,
        quarantined=quarantined,
        format=fmt,
        arches=arches,
        run_slice=run_slice_name,
        min_os=min_os,
        code_signature=signature,
        dependencies=dependencies,
        macho=macho,
        problems=tuple(f.message for f in findings),
        findings=tuple(findings),
    )


def candidate_check_for(host: HostFacts | None = None) -> Callable:
    """The deeper check the core resolver takes as ``candidate_check``.

    It runs only for candidates that passed the core's own checks, and returns a core diagnostic
    for the first blocking finding among :data:`DEEP_BLOCKING_CODES`, or ``None``. Using the same
    function for the tool panel, the capability report and a run makes all three select the same
    file -- a broken ``platon_special`` falls back to a usable ``platon`` everywhere.
    """
    from crystalx_core.domain.tools import ToolDiagnostic, ToolDiagnosticKind

    from crystalx_native_tools.host import detect_host

    facts: list[HostFacts] = [host] if host is not None else []

    def check(kind, path):
        if not facts:
            facts.append(detect_host())
        blocking = inspect_file(path, facts[0]).first_blocking()
        if blocking is None:
            return None
        return ToolDiagnostic(kind=ToolDiagnosticKind(blocking.code), message=f"{path}: {blocking.message}",
                              path=Path(path), remedy=_REMEDIES[blocking.code])

    return check
