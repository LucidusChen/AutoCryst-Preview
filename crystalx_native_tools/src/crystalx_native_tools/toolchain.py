"""Compiler toolchain inventory for a possible local arm64 build -- by looking, never by running.

On macOS ``/usr/bin/cc``, ``/usr/bin/clang`` and ``/usr/bin/make`` are ``xcode-select`` shims: run
without the Command Line Tools installed, they open an installer prompt. Nothing here executes
them (or anything else); presence is judged from the real toolchain directories, and compiler
architectures from Mach-O headers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from crystalx_native_tools.host import HostFacts
from crystalx_native_tools.macho import read_macho

__all__ = ["CompilerCandidate", "ToolchainReport", "inspect_toolchain"]

_CLT = Path("/Library/Developer/CommandLineTools")
_XCODE = Path("/Applications/Xcode.app/Contents/Developer")
_FORTRAN_DIRS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/opt/local/bin"), Path("/usr/local/gfortran/bin"))
_X11_LIB = Path("/opt/X11/lib/libX11.6.dylib")


@dataclass(frozen=True)
class CompilerCandidate:
    name: str
    path: str
    realpath: str
    arches: tuple[str, ...]

    @property
    def has_arm64(self) -> bool:
        return any(a in ("arm64", "arm64e") for a in self.arches)

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "path": self.path, "realpath": self.realpath, "arches": list(self.arches)}


@dataclass(frozen=True)
class ToolchainReport:
    system: str
    command_line_tools: str | None
    xcode: str | None
    sdks: tuple[str, ...]
    fortran: tuple[CompilerCandidate, ...]
    c_compiler: str | None
    make: str | None
    xquartz_x11: str | None
    notes: tuple[str, ...]

    @property
    def can_build_fortran_arm64(self) -> bool:
        """An arm64 gfortran plus Apple's linker/SDK (Command Line Tools or Xcode) are present."""
        has_apple_tools = self.command_line_tools is not None or self.xcode is not None
        return self.system == "Darwin" and has_apple_tools and any(c.has_arm64 for c in self.fortran)

    def to_record(self) -> dict[str, object]:
        return {
            "system": self.system,
            "command_line_tools": self.command_line_tools,
            "xcode": self.xcode,
            "sdks": list(self.sdks),
            "fortran": [c.to_record() for c in self.fortran],
            "c_compiler": self.c_compiler,
            "make": self.make,
            "xquartz_x11": self.xquartz_x11,
            "can_build_fortran_arm64": self.can_build_fortran_arm64,
            "notes": list(self.notes),
            "method": "presence and Mach-O headers only; no compiler or shim was executed",
        }


def inspect_toolchain(host: HostFacts, *, clt: Path = _CLT, xcode: Path = _XCODE,
                      fortran_dirs: tuple[Path, ...] = _FORTRAN_DIRS, x11_lib: Path = _X11_LIB) -> ToolchainReport:
    notes: list[str] = []
    if host.system != "Darwin":
        notes.append("toolchain inventory targets macOS; other hosts are reported without details")
        return ToolchainReport(host.system, None, None, (), (), None, None, None, tuple(notes))
    has_clt = (clt / "usr" / "bin" / "clang").is_file()
    has_xcode = xcode.is_dir()
    sdk_root = clt / "SDKs"
    sdks = tuple(sorted(p.name for p in sdk_root.glob("*.sdk"))) if sdk_root.is_dir() else ()
    fortran: list[CompilerCandidate] = []
    seen: set[str] = set()
    for directory in fortran_dirs:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("gfortran*")):
            if not candidate.is_file():
                continue
            real = os.path.realpath(candidate)
            if real in seen:
                continue
            seen.add(real)
            macho = read_macho(Path(real))
            fortran.append(CompilerCandidate(candidate.name, str(candidate), real, macho.arches if macho else ()))
    c_compiler = str(clt / "usr" / "bin" / "clang") if has_clt else (
        str(xcode / "Toolchains/XcodeDefault.xctoolchain/usr/bin/clang") if has_xcode else None
    )
    make = str(clt / "usr" / "bin" / "make") if (clt / "usr" / "bin" / "make").is_file() else (
        str(xcode / "usr/bin/make") if (xcode / "usr/bin/make").is_file() else None
    )
    if not has_clt and not has_xcode:
        notes.append(
            "No Command Line Tools or Xcode found. /usr/bin/cc and /usr/bin/make are installer shims "
            "and are never run here; installing the tools is the owner's decision."
        )
    if not fortran:
        notes.append("No gfortran found in the usual Homebrew, MacPorts or gfortran-installer locations.")
    return ToolchainReport(
        system=host.system,
        command_line_tools=str(clt) if has_clt else None,
        xcode=str(xcode) if has_xcode else None,
        sdks=sdks,
        fortran=tuple(fortran),
        c_compiler=c_compiler,
        make=make,
        xquartz_x11=str(x11_lib) if x11_lib.is_file() else None,
        notes=tuple(notes),
    )
