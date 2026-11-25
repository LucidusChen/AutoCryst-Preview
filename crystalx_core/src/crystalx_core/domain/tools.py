"""External crystallographic programs: identity, architecture, and failure diagnosis.

SHELX and PLATON are user-supplied, separately licensed programs. This project resolves and invokes
them; it never downloads, compiles, bundles, or redistributes them.

The diagnostic vocabulary here exists because "shelxt not found" is the wrong message when the real
problem is that the executable is an x86_64 binary on an Apple Silicon machine without Rosetta 2
installed -- a situation that is actually present on this project's development machine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ToolKind",
    "BinaryArch",
    "BinaryFormat",
    "ResolutionSource",
    "ToolDiagnosticKind",
    "BinaryProbe",
    "ToolIdentity",
    "ToolDiagnostic",
    "ToolResolution",
    "ToolAvailabilityReport",
    "HostPlatform",
]


class ToolKind(str, enum.Enum):
    """External programs the workflow knows how to talk to."""

    SHELXT = "shelxt"
    SHELXS = "shelxs"
    SHELXD = "shelxd"
    SHELXE = "shelxe"
    SHELXC = "shelxc"
    SHELXL = "shelxl"
    ANODE = "anode"
    PLATON = "platon"
    PLATON_SPECIAL = "platon_special"
    CIFTAB = "ciftab"
    SHREDCIF = "shredcif"


class BinaryArch(str, enum.Enum):
    """CPU architecture of an executable, or of the host."""

    X86_64 = "x86_64"
    ARM64 = "arm64"
    I386 = "i386"
    ARM32 = "arm32"
    UNIVERSAL = "universal"
    """A Mach-O fat binary. ``BinaryProbe.arches`` lists what it actually contains."""

    SCRIPT = "script"
    """A ``#!`` script: architecture is whatever its interpreter is."""

    UNKNOWN = "unknown"


class BinaryFormat(str, enum.Enum):
    MACH_O = "mach_o"
    MACH_O_FAT = "mach_o_fat"
    ELF = "elf"
    PE = "pe"
    SCRIPT = "script"
    UNKNOWN = "unknown"


class ResolutionSource(str, enum.Enum):
    """Where a resolved executable came from. Fixed precedence, highest first."""

    EXPLICIT_CONFIG = "explicit_config"
    TOOLBOX = "toolbox"
    PATH_ENV = "path_env"
    OLEX2 = "olex2"
    KNOWN_LOCATION = "known_location"

    SYNTHETIC_DEMO = "synthetic_demo"
    """Not a program at all: a stand-in from the built-in demo toolchain. The resolver never
    produces this; it exists so a demo run's provenance cannot be mistaken for a real tool's."""


class ToolDiagnosticKind(str, enum.Enum):
    """Why a tool is or is not usable. Never collapse these into a single boolean."""

    OK = "ok"
    OK_VIA_TRANSLATION = "ok_via_translation"
    """Usable, but only through Rosetta 2 (x86_64 binary on an arm64 host). Expect it to be slower."""

    NOT_FOUND = "not_found"
    NOT_A_FILE = "not_a_file"
    NOT_EXECUTABLE = "not_executable"
    """Present and readable, but the executable bit is not set."""

    ARCH_MISMATCH = "arch_mismatch"
    """Built for a CPU this host cannot run, and no translation layer can help."""

    ROSETTA_MISSING = "rosetta_missing"
    """x86_64 binary on Apple Silicon, and Rosetta 2 is not installed."""

    TRANSLATION_REFUSED = "translation_refused"
    """x86_64-only binary on Apple Silicon, refused because strict native (no-Rosetta) mode is on.
    The search continues, so a native arm64 or universal build elsewhere can still be used."""

    QUARANTINED = "quarantined"
    """macOS Gatekeeper quarantine attribute set; the OS will refuse to run it."""

    NATIVE_UNVERIFIED = "native_unverified"
    """Strict native mode: a script or a file of unrecognised format. Its own bytes cannot show that
    native code will run (a wrapper may start an x86_64 program through Rosetta 2), so it is refused."""

    NOT_AN_EXECUTABLE = "not_an_executable"
    """A Mach-O file that is not a main executable (a dynamic library, bundle, object file...)."""

    WRONG_PLATFORM = "wrong_platform"
    """A Mach-O executable built for another Apple platform (iOS, a simulator...), not macOS."""

    MALFORMED_BINARY = "malformed_binary"
    """Structurally broken: truncated or inconsistent load commands, no entry point."""

    UNSIGNED_ARM64 = "unsigned_arm64"
    """arm64 code without a readable code-signature container; macOS refuses to run it. (Presence
    of a container is checked, never its cryptographic validity.)"""

    REQUIRES_NEWER_MACOS = "requires_newer_macos"
    """Built for a newer macOS than this machine runs."""

    LIBRARY_UNAVAILABLE = "library_unavailable"
    """A required (non-weak) linked library is missing or has no slice for this architecture."""

    UNREADABLE = "unreadable"
    TIMEOUT = "timeout"
    NONZERO_EXIT = "nonzero_exit"
    EXPECTED_OUTPUT_MISSING = "expected_output_missing"
    """The program exited successfully but did not produce a file the caller required."""

    VERSION_UNKNOWN = "version_unknown"

    CANCELLED = "cancelled"
    """Stopped on request. The process group was terminated; logs written so far are kept."""

    EXECUTABLE_CHANGED = "executable_changed"
    """The file at the resolved path no longer has the digest recorded at resolution. The program
    is not started: the provenance would otherwise name a binary that did not run."""

    RESOURCE_LIMIT = "resource_limit"
    """The kernel stopped the program for exceeding a declared limit (CPU time or file size).

    Distinct from TIMEOUT, which is wall-clock and enforced by this package: a resource limit is
    enforced by the operating system and leaves a signal in the exit status.
    """

    @property
    def is_usable(self) -> bool:
        return self in (ToolDiagnosticKind.OK, ToolDiagnosticKind.OK_VIA_TRANSLATION)


@dataclass(frozen=True)
class HostPlatform:
    """The machine we are running on, as far as tool compatibility is concerned."""

    system: str
    """``Darwin``, ``Linux``, ``Windows``."""

    arch: BinaryArch
    rosetta_available: bool | None = None
    """``None`` when not applicable (non-macOS) or not determined."""

    @property
    def is_macos(self) -> bool:
        return self.system == "Darwin"


@dataclass(frozen=True)
class BinaryProbe:
    """Result of reading an executable's header bytes.

    Header parsing only: the file is opened, a few dozen bytes are read, and it is closed. The
    program is never executed to find out what it is.
    """

    fmt: BinaryFormat
    arch: BinaryArch
    arches: tuple[BinaryArch, ...] = ()
    interpreter: str = ""
    note: str = ""

    def supports(self, host_arch: BinaryArch) -> bool:
        """True when this binary can run natively on ``host_arch``."""
        if self.fmt is BinaryFormat.SCRIPT:
            return True
        if self.arch is BinaryArch.UNIVERSAL:
            return host_arch in self.arches
        return self.arch is host_arch


@dataclass(frozen=True)
class ToolIdentity:
    """A specific executable, pinned well enough to reproduce a run later."""

    kind: ToolKind
    path: Path
    realpath: Path
    source: ResolutionSource
    size_bytes: int
    sha256: str
    mtime_epoch: float
    probe: BinaryProbe
    version: str | None = None
    version_probe_note: str = ""

    @property
    def arch(self) -> BinaryArch:
        return self.probe.arch


@dataclass(frozen=True)
class ToolDiagnostic:
    """A specific reason a specific path was rejected or accepted, in the user's terms."""

    kind: ToolDiagnosticKind
    message: str
    path: Path | None = None
    remedy: str = ""
    """A concrete, actionable next step, e.g. the exact command that installs Rosetta 2."""


@dataclass(frozen=True)
class ToolResolution:
    """Outcome of resolving one tool: either an identity, or the reasons every candidate failed."""

    kind: ToolKind
    identity: ToolIdentity | None
    diagnostic: ToolDiagnostic
    rejected: tuple[ToolDiagnostic, ...] = ()
    searched_paths: tuple[Path, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.identity is not None and self.diagnostic.kind.is_usable


@dataclass(frozen=True)
class ToolAvailabilityReport:
    """Everything ``crystalx doctor`` found, and what the host is."""

    host: HostPlatform
    resolutions: dict[str, ToolResolution] = field(default_factory=dict)

    def get(self, kind: ToolKind) -> ToolResolution | None:
        return self.resolutions.get(kind.value)

    def usable(self, kind: ToolKind) -> bool:
        resolution = self.get(kind)
        return bool(resolution and resolution.is_usable)

    @property
    def usable_kinds(self) -> tuple[ToolKind, ...]:
        return tuple(
            ToolKind(name) for name, res in sorted(self.resolutions.items()) if res.is_usable
        )
