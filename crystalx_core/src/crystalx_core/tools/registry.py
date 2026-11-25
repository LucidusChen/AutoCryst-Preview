"""Static knowledge about the external programs the workflow can drive.

Executable *names* and the conventional places people install them. No binary is ever shipped,
downloaded, compiled or executed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.domain.tools import ToolKind

__all__ = [
    "ToolSpec",
    "TOOL_SPECS",
    "spec_for",
    "olex2_candidate_dirs",
    "known_install_dirs",
    "PLATON_PREFERENCE",
]


@dataclass(frozen=True)
class ToolSpec:
    """How to recognise one external program."""

    kind: ToolKind
    executable_names: tuple[str, ...]
    purpose: str
    version_argv: tuple[str, ...] = ()
    """Arguments that make the program print its version. Empty when there is no safe way to ask.

    Most SHELX programs print a banner and then wait for input or exit non-zero when run without a
    data file, so probing them is not free of side effects. Version probing is therefore opt-in and
    off by default.
    """

    version_probe_is_safe: bool = False
    reads_stdin_instructions: bool = False
    """PLATON drives most of its analyses (ADDSYM, SQUEEZE, ASYM, LEPAGE) from interactive stdin
    instructions rather than command-line flags, so the runner needs a scripted-stdin mode."""

    needs_x11: bool = False
    notes: str = ""


#: PLATON preference order for headless automation: the stripped, graphics-free ``special`` build
#: first, ordinary ``platon`` as the compatibility fallback.
PLATON_PREFERENCE: tuple[ToolKind, ...] = (ToolKind.PLATON_SPECIAL, ToolKind.PLATON)


TOOL_SPECS: dict[ToolKind, ToolSpec] = {
    ToolKind.SHELXT: ToolSpec(
        kind=ToolKind.SHELXT,
        executable_names=("shelxt",),
        purpose="Dual-space solution with automatic space-group determination.",
        notes="Writes <name>_a.res, <name>_b.res, ... one per candidate space group, plus <name>.lxt.",
    ),
    ToolKind.SHELXS: ToolSpec(
        kind=ToolKind.SHELXS,
        executable_names=("shelxs",),
        purpose="Classical direct methods (TREF) and Patterson interpretation (PATT).",
        notes="PATT is the correct classical route to a dominant heavy atom; it is not a legacy mode.",
    ),
    ToolKind.SHELXD: ToolSpec(
        kind=ToolKind.SHELXD,
        executable_names=("shelxd",),
        purpose="Dual-space direct methods: ab initio at atomic resolution, or substructure search.",
    ),
    ToolKind.SHELXE: ToolSpec(
        kind=ToolKind.SHELXE,
        executable_names=("shelxe",),
        purpose="Density modification, phase extension, hand determination, poly-Ala autotracing.",
        notes="Autotracing assumes polypeptide geometry; keep it off for metal-organic small molecules.",
    ),
    ToolKind.SHELXC: ToolSpec(
        kind=ToolKind.SHELXC,
        executable_names=("shelxc",),
        purpose="Prepare experimental-phasing input and measure the anomalous signal vs resolution.",
        reads_stdin_instructions=True,
        notes="Writes <name>_fa.ins / <name>_fa.hkl for SHELXD and SHELXE.",
    ),
    ToolKind.SHELXL: ToolSpec(
        kind=ToolKind.SHELXL,
        executable_names=("shelxl",),
        purpose="Least-squares refinement; source of R factors, connectivity table and Q peaks.",
    ),
    ToolKind.ANODE: ToolSpec(
        kind=ToolKind.ANODE,
        executable_names=("anode",),
        purpose="Anomalous difference density per site: independent verification of a metal assignment.",
    ),
    ToolKind.PLATON_SPECIAL: ToolSpec(
        kind=ToolKind.PLATON_SPECIAL,
        executable_names=("platon_special",),
        purpose="Stripped, graphics-free PLATON. Preferred for headless validation.",
        reads_stdin_instructions=True,
        needs_x11=False,
        notes="Validation runs as 'platon_special -u name.cif' and writes name.chk.",
    ),
    ToolKind.PLATON: ToolSpec(
        kind=ToolKind.PLATON,
        executable_names=("platon",),
        purpose="Full PLATON. Fallback when the special build is unavailable.",
        reads_stdin_instructions=True,
        needs_x11=True,
        notes=(
            "Opens an X11 window by default; '-o' forces keyboard/TEK4010 mode for headless use. "
            "check.def must be in the working directory or pointed at by $CHECKDEF, and its yymmdd "
            "version must match the executable."
        ),
    ),
    ToolKind.CIFTAB: ToolSpec(
        kind=ToolKind.CIFTAB,
        executable_names=("ciftab",),
        purpose="SHELX CIF tabulation utility.",
    ),
    ToolKind.SHREDCIF: ToolSpec(
        kind=ToolKind.SHREDCIF,
        executable_names=("shredcif",),
        purpose="Split a multi-structure CIF into individual files.",
    ),
}


def spec_for(kind: ToolKind) -> ToolSpec:
    return TOOL_SPECS[kind]


def known_install_dirs(*, system: str) -> tuple[Path, ...]:
    """Conventional locations for locally installed crystallographic programs."""
    if system == "Darwin":
        return (
            Path("/usr/local/bin"),
            Path("/opt/homebrew/bin"),
            Path("/opt/local/bin"),
            Path.home() / "bin",
        )
    if system == "Linux":
        return (
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/opt/shelx"),
            Path.home() / "bin",
        )
    return (Path("C:/Program Files"),) if system == "Windows" else ()


def olex2_candidate_dirs(*, system: str) -> tuple[Path, ...]:
    """Places an Olex2 installation commonly keeps or references SHELX executables.

    Only directories that are safe to *stat*. Nothing here is executed, and a directory that does not
    exist is simply skipped.
    """
    home = Path.home()
    if system == "Darwin":
        fixed = (
            Path("/Applications/olex2.app/Contents/MacOS"),
            Path("/Applications/Olex2.app/Contents/MacOS"),
            home / "Applications/olex2.app/Contents/MacOS",
        )
        # Versioned bundle names (e.g. "Olex2-1.5.app") exist too; list the ones present, sorted so
        # the order -- and therefore which copy wins -- is stable.
        versioned: list[Path] = []
        for root in (Path("/Applications"), home / "Applications"):
            try:
                bundles = sorted(root.glob("[Oo]lex2*.app"))
            except OSError:
                bundles = []
            versioned.extend(b / "Contents" / "MacOS" for b in bundles)
        ordered: list[Path] = []
        for directory in (*fixed, *versioned, home / ".olex2", home / "olex2"):
            if directory not in ordered:
                ordered.append(directory)
        return tuple(ordered)
    if system == "Linux":
        return (
            Path("/opt/olex2"),
            home / "olex2",
            home / ".olex2",
        )
    if system == "Windows":
        return (
            Path("C:/Olex2"),
            Path("C:/Program Files/Olex2"),
        )
    return ()
