"""Resolve external crystallographic programs, and explain precisely why one is unusable.

Resolution order, fixed:

1. explicit user configuration
2. the project-local ``toolbox/`` directory
3. ``PATH`` (including ``/usr/local/bin``, which is where the PLATON-on-Mac guide tells users to put
   these programs)
4. documented Olex2 and conventional installation locations

For every candidate path the resolver records what it found, so ``crystalx doctor`` can tell a user
"this is an x86_64 build and Rosetta 2 is not installed" instead of "not found".
"""

from __future__ import annotations

import os
import platform as _platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from crystalx_core.domain.tools import (
    BinaryArch,
    BinaryFormat,
    HostPlatform,
    ResolutionSource,
    ToolAvailabilityReport,
    ToolDiagnostic,
    ToolDiagnosticKind,
    ToolIdentity,
    ToolKind,
    ToolResolution,
)
from crystalx_core.tools.binfmt import probe_binary
from crystalx_core.tools.registry import PLATON_PREFERENCE, known_install_dirs, olex2_candidate_dirs, spec_for
from crystalx_core.tools.xattr import QUARANTINE_XATTR, has_xattr
from crystalx_core.workspace.hashing import sha256_file

__all__ = [
    "ToolResolver",
    "ResolverConfig",
    "detect_host_platform",
    "rosetta_is_installed",
    "ROSETTA_MARKER_PATHS",
    "ROSETTA_REMEDY",
    "STRICT_NATIVE_REMEDY",
    "NATIVE_UNVERIFIED_REMEDY",
]

#: Files macOS creates when Rosetta 2 appears to be installed. Presence is checked; nothing is
#: executed. A marker alone cannot prove a particular x86_64 binary will launch, so an actual tool
#: invocation remains the final operational test.
ROSETTA_MARKER_PATHS: tuple[Path, ...] = (
    Path("/Library/Apple/usr/libexec/oah/libRosettaRuntime"),
    Path("/Library/Apple/usr/share/rosetta/rosetta"),
    Path("/usr/libexec/rosetta/oahd"),
)

ROSETTA_REMEDY = (
    "This is an Intel (x86_64) build running on an Apple Silicon Mac. Install Rosetta 2 with:\n"
    "    softwareupdate --install-rosetta --agree-to-license"
)

STRICT_NATIVE_REMEDY = (
    "Use a native arm64 (or universal) build of this program: check the official distribution for "
    "an Apple Silicon build, or point the configuration at one you already have. A wrapper cannot "
    "turn x86_64 code into arm64 code; only a native executable, or legitimately obtained source "
    "that its terms allow you to build locally, removes the Rosetta dependency. Alternatively turn "
    "strict native mode off to allow Rosetta 2 -- which must then be installed and working; "
    "installing it is the machine owner's decision."
)

NATIVE_UNVERIFIED_REMEDY = (
    "Name the native executable itself (for example under 'tools' in config.local.json) instead of "
    "a wrapper script. Strict native mode only uses files whose own format shows native code; turn "
    "it off only if you accept running an unverified wrapper."
)

_QUARANTINE_XATTR = QUARANTINE_XATTR

_ARCH_ALIASES = {
    "x86_64": BinaryArch.X86_64,
    "amd64": BinaryArch.X86_64,
    "arm64": BinaryArch.ARM64,
    "aarch64": BinaryArch.ARM64,
    "i386": BinaryArch.I386,
    "i686": BinaryArch.I386,
    "armv7l": BinaryArch.ARM32,
}


def rosetta_is_installed(
    *, exists: Callable[[Path], bool] | None = None
) -> bool:
    """True when a Rosetta 2 marker file is present. Never runs anything.

    This is deliberately a *presence* probe, not a promise that translation is operational. The
    first opt-in tool invocation is the evidence that an x86_64 executable actually launches.
    """
    check = exists or (lambda p: p.exists())
    return any(check(marker) for marker in ROSETTA_MARKER_PATHS)


def _sysctl_int(name: str) -> int | None:
    """Read a Darwin host fact without launching a command or adding a dependency."""
    import ctypes
    try:
        query = ctypes.CDLL(None, use_errno=True).sysctlbyname
    except (OSError, AttributeError):
        return None
    value = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    query.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                      ctypes.c_void_p, ctypes.c_size_t]
    query.restype = ctypes.c_int
    if query(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return value.value


def detect_host_platform(
    *,
    system: str | None = None,
    machine: str | None = None,
    exists: Callable[[Path], bool] | None = None,
) -> HostPlatform:
    """Describe native host architecture, not a possibly translated Python's architecture.

    Explicit system/machine arguments describe a synthetic host and bypass hardware probing.
    If Darwin hardware queries are unavailable, the process architecture is retained; that
    fallback is not independent evidence that the process is untranslated.
    """
    resolved_system = system or _platform.system()
    resolved_machine = (machine or _platform.machine()).lower()
    arch = _ARCH_ALIASES.get(resolved_machine, BinaryArch.UNKNOWN)
    if resolved_system == "Darwin" and system is None and machine is None:
        if _sysctl_int("hw.optional.arm64") == 1 or _sysctl_int("sysctl.proc_translated") == 1:
            arch = BinaryArch.ARM64
    rosetta: bool | None = None
    if resolved_system == "Darwin" and arch is BinaryArch.ARM64:
        rosetta = rosetta_is_installed(exists=exists)
    return HostPlatform(system=resolved_system, arch=arch, rosetta_available=rosetta)


@dataclass(frozen=True)
class ResolverConfig:
    """Where to look, in order."""

    explicit_paths: Mapping[str, Path] = field(default_factory=dict)
    """``ToolKind.value -> absolute path``. Highest precedence; always recorded as an override."""

    toolbox_dir: Path | None = None
    search_path: Sequence[Path] | None = None
    """``None`` means "read ``PATH`` from the environment"."""

    include_known_locations: bool = True
    include_olex2: bool = True
    extra_dirs: Sequence[Path] = ()
    allow_translation: bool = True
    """``False`` is strict native mode: an x86_64-only build on Apple Silicon is rejected
    (``translation_refused``) instead of being planned through Rosetta 2, and a script or a file
    of unrecognised format is rejected as ``native_unverified`` (its own bytes cannot show what
    it will run). Either way the search goes on to the next candidate."""

    candidate_check: Callable[[ToolKind, Path], ToolDiagnostic | None] | None = field(default=None, compare=False)
    """A deeper, injected check of a candidate that passed the checks above, e.g. linked libraries,
    code signature, minimum macOS and Mach-O structure (``crystalx_native_tools`` provides one).
    Returning an unusable diagnostic rejects that candidate and the search continues, so the tool
    panel, the capability report and a run all select the same file."""


class ToolResolver:
    """Finds executables and classifies what is wrong when it cannot use one."""

    def __init__(
        self,
        config: ResolverConfig | None = None,
        *,
        host: HostPlatform | None = None,
        environ: Mapping[str, str] | None = None,
        quarantine_checker: Callable[[Path], bool] | None = None,
    ) -> None:
        self.config = config or ResolverConfig()
        self.host = host or detect_host_platform()
        self.environ = dict(environ if environ is not None else os.environ)
        self._is_quarantined = quarantine_checker or _default_quarantine_checker

    # ------------------------------------------------------------------ search
    def candidate_paths(self, kind: ToolKind) -> list[tuple[Path, ResolutionSource]]:
        """Every path worth checking for ``kind``, in precedence order, de-duplicated."""
        spec = spec_for(kind)
        found: list[tuple[Path, ResolutionSource]] = []
        seen: set[Path] = set()

        def add(path: Path, source: ResolutionSource) -> None:
            # Absolute, without resolving symlinks: a relative toolbox such as "toolbox" must keep
            # naming the same file after a program is started in an isolated working directory.
            path = _absolute(path)
            if path not in seen:
                seen.add(path)
                found.append((path, source))

        explicit = self.config.explicit_paths.get(kind.value)
        if explicit is not None:
            add(Path(explicit), ResolutionSource.EXPLICIT_CONFIG)

        for name in spec.executable_names:
            if self.config.toolbox_dir is not None:
                add(Path(self.config.toolbox_dir) / name, ResolutionSource.TOOLBOX)

        for directory in self._path_dirs():
            for name in spec.executable_names:
                add(directory / name, ResolutionSource.PATH_ENV)

        if self.config.include_olex2:
            for directory in olex2_candidate_dirs(system=self.host.system):
                for name in spec.executable_names:
                    add(directory / name, ResolutionSource.OLEX2)

        if self.config.include_known_locations:
            for directory in known_install_dirs(system=self.host.system):
                for name in spec.executable_names:
                    add(directory / name, ResolutionSource.KNOWN_LOCATION)

        for directory in self.config.extra_dirs:
            for name in spec.executable_names:
                add(Path(directory) / name, ResolutionSource.KNOWN_LOCATION)

        return found

    def _path_dirs(self) -> list[Path]:
        if self.config.search_path is not None:
            return [Path(p) for p in self.config.search_path]
        raw = self.environ.get("PATH", "")
        dirs = [Path(p) for p in raw.split(os.pathsep) if p]
        # The PLATON-on-Mac guide installs into /usr/local/bin; make sure we look there even when
        # the caller's PATH does not.
        usr_local = Path("/usr/local/bin")
        if self.host.system in ("Darwin", "Linux") and usr_local not in dirs:
            dirs.append(usr_local)
        return dirs

    # ------------------------------------------------------------------ resolution
    def resolve(self, kind: ToolKind) -> ToolResolution:
        """Resolve one tool, returning either an identity or every reason it failed."""
        rejected: list[ToolDiagnostic] = []
        searched: list[Path] = []

        for path, source in self.candidate_paths(kind):
            searched.append(path)
            identity, diagnostic = self._inspect(kind, path, source)
            if identity is not None and diagnostic.kind.is_usable:
                return ToolResolution(
                    kind=kind,
                    identity=identity,
                    diagnostic=diagnostic,
                    rejected=tuple(rejected),
                    searched_paths=tuple(searched),
                )
            if diagnostic.kind is not ToolDiagnosticKind.NOT_FOUND:
                rejected.append(diagnostic)

        message = (
            f"{kind.value} was not found."
            if not rejected
            else f"{kind.value} was found but is not usable: {rejected[0].message}"
        )
        summary_kind = rejected[0].kind if rejected else ToolDiagnosticKind.NOT_FOUND
        remedy = rejected[0].remedy if rejected else (
            f"Install {kind.value}, or place it in the project toolbox directory, "
            "or set an explicit path in the local configuration."
        )
        return ToolResolution(
            kind=kind,
            identity=None,
            diagnostic=ToolDiagnostic(kind=summary_kind, message=message, remedy=remedy),
            rejected=tuple(rejected),
            searched_paths=tuple(searched),
        )

    def resolve_all(self, kinds: Iterable[ToolKind] | None = None) -> ToolAvailabilityReport:
        """Resolve a set of tools (all known kinds by default)."""
        selected = tuple(kinds) if kinds is not None else tuple(ToolKind)
        return ToolAvailabilityReport(
            host=self.host,
            resolutions={kind.value: self.resolve(kind) for kind in selected},
        )

    def resolve_platon(self) -> ToolResolution:
        """Prefer the graphics-free ``platon_special`` build; fall back to ordinary ``platon``.

        The stripped build exists precisely for headless validation jobs such as
        ``platon_special -u name.cif``; the full build tries to open an X11 window.
        """
        last: ToolResolution | None = None
        for kind in PLATON_PREFERENCE:
            resolution = self.resolve(kind)
            if resolution.is_usable:
                return resolution
            last = resolution
        assert last is not None
        return last

    # ------------------------------------------------------------------ inspection
    def inspect_candidate(
        self, kind: ToolKind, path: Path, source: ResolutionSource
    ) -> tuple[ToolIdentity | None, ToolDiagnostic]:
        """Judge one candidate exactly as :meth:`resolve` would, without searching further.

        Lets a tool inventory report every candidate with the same verdict a run would reach.
        """
        return self._inspect(kind, _absolute(path), source)

    def _inspect(
        self, kind: ToolKind, path: Path, source: ResolutionSource
    ) -> tuple[ToolIdentity | None, ToolDiagnostic]:
        if not path.exists():
            return None, ToolDiagnostic(
                kind=ToolDiagnosticKind.NOT_FOUND, message=f"no file at {path}", path=path
            )
        if not path.is_file():
            return None, ToolDiagnostic(
                kind=ToolDiagnosticKind.NOT_A_FILE,
                message=f"{path} exists but is not a regular file",
                path=path,
            )
        if not os.access(path, os.X_OK):
            return None, ToolDiagnostic(
                kind=ToolDiagnosticKind.NOT_EXECUTABLE,
                message=f"{path} is not marked executable",
                path=path,
                remedy=f"chmod +x {path}",
            )

        try:
            stat_result = path.stat()
            digest = sha256_file(path)
        except OSError as exc:
            return None, ToolDiagnostic(
                kind=ToolDiagnosticKind.UNREADABLE,
                message=f"cannot read {path}: {exc}",
                path=path,
            )

        if self.host.is_macos and self._is_quarantined(path):
            return None, ToolDiagnostic(
                kind=ToolDiagnosticKind.QUARANTINED,
                message=f"{path} carries the macOS quarantine attribute and will be refused",
                path=path,
                remedy=f"xattr -d com.apple.quarantine {path}",
            )

        probe = probe_binary(path)
        arch_diagnostic = self._check_arch(path, probe)
        if not arch_diagnostic.kind.is_usable:
            return None, arch_diagnostic
        if self.config.candidate_check is not None:
            deeper = self.config.candidate_check(kind, path)
            if deeper is not None and not deeper.kind.is_usable:
                return None, deeper

        identity = ToolIdentity(
            kind=kind,
            path=path,
            realpath=path.resolve(),
            source=source,
            size_bytes=stat_result.st_size,
            sha256=digest,
            mtime_epoch=stat_result.st_mtime,
            probe=probe,
            version=None,
            version_probe_note=(
                "not probed: running a SHELX program without input is not side-effect free"
            ),
        )
        return identity, arch_diagnostic

    def _check_arch(self, path: Path, probe) -> ToolDiagnostic:
        """Decide whether this host can run this binary, and say what to do if not."""
        unrecognised = probe.arch is BinaryArch.UNKNOWN and probe.fmt is BinaryFormat.UNKNOWN
        if not self.config.allow_translation and (probe.fmt is BinaryFormat.SCRIPT or unrecognised):
            what = (f"a script ({probe.interpreter or 'unknown interpreter'})" if probe.fmt is BinaryFormat.SCRIPT
                    else "a file of unrecognised format")
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.NATIVE_UNVERIFIED,
                message=(
                    f"{path} is {what}. Strict native mode cannot verify what it would run (a wrapper "
                    "may start an x86_64 program through Rosetta 2), so it is not used."
                ),
                path=path,
                remedy=NATIVE_UNVERIFIED_REMEDY,
            )
        if probe.fmt is BinaryFormat.SCRIPT:
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.OK,
                message=f"{path} is a script ({probe.interpreter or 'unknown interpreter'})",
                path=path,
            )
        if probe.arch is BinaryArch.UNKNOWN and probe.fmt is BinaryFormat.UNKNOWN:
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.OK,
                message=f"{path} has an unrecognised header; assuming it is runnable",
                path=path,
                remedy="",
            )
        if probe.supports(self.host.arch):
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.OK,
                message=f"{path} is a native {self.host.arch.value} executable",
                path=path,
            )

        needs_translation = (
            self.host.is_macos
            and self.host.arch is BinaryArch.ARM64
            and (
                probe.arch is BinaryArch.X86_64
                or (probe.arch is BinaryArch.UNIVERSAL and BinaryArch.X86_64 in probe.arches)
            )
        )
        if needs_translation:
            if not self.config.allow_translation:
                return ToolDiagnostic(
                    kind=ToolDiagnosticKind.TRANSLATION_REFUSED,
                    message=(
                        f"{path} contains no arm64 code (x86_64 only). Strict native mode refuses "
                        "translation through Rosetta 2 on this arm64 machine."
                    ),
                    path=path,
                    remedy=STRICT_NATIVE_REMEDY,
                )
            if self.host.rosetta_available:
                return ToolDiagnostic(
                    kind=ToolDiagnosticKind.OK_VIA_TRANSLATION,
                    message=(
                        f"{path} is an x86_64 build and a Rosetta 2 marker is present on this "
                        "arm64 machine. It still needs an opt-in launch check; if that fails, "
                        "install or repair Rosetta 2. Expect translation to be slower than a native build."
                    ),
                    path=path,
                )
            return ToolDiagnostic(
                kind=ToolDiagnosticKind.ROSETTA_MISSING,
                message=(
                    f"{path} is an x86_64 build, this machine is arm64, and Rosetta 2 does not "
                    "appear to be installed. macOS will refuse it with "
                    "'Bad CPU type in executable'."
                ),
                path=path,
                remedy=ROSETTA_REMEDY,
            )

        return ToolDiagnostic(
            kind=ToolDiagnosticKind.ARCH_MISMATCH,
            message=(
                f"{path} is built for {probe.arch.value} but this host is "
                f"{self.host.arch.value}; there is no translation layer for this combination."
            ),
            path=path,
            remedy=f"Obtain a {self.host.arch.value} build of this program.",
        )


def _absolute(path: Path) -> Path:
    """``path`` made absolute against the current directory, symlinks left as they are."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(os.path.abspath(path))


def _default_quarantine_checker(path: Path) -> bool:
    """True when the macOS quarantine attribute is present.

    Uses :func:`crystalx_core.tools.xattr.has_xattr`, which reads the attribute on macOS through
    libc (Python's ``os.getxattr`` exists only on Linux, so the earlier check could never see it).
    An undeterminable answer counts as "not detected".
    """
    return has_xattr(path, _QUARANTINE_XATTR) is True
