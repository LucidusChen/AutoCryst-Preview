"""Plans for an isolated, user-local arm64 build -- from source the owner supplies. Plans only.

Nothing here compiles, copies or downloads anything: :func:`plan_build` returns a
:class:`BuildPlan` with ``executed=False`` and the reasons it could or could not be carried out.
A plan exists for the day the owner provides source whose terms allow a local build; until then
the honest output for SHELX is "no source available", because current SHELX is distributed as
executables. Mach-O executables are never treated as source: they cannot be recompiled.

Recipes are deliberately few and labelled:

* ``platon_full_x11`` -- ``platon.f`` + ``xdrvr.c``, the documented graphics build (gfortran,
  XQuartz X11 headers and library). Used only for the ordinary ``platon``.
* ``single_fortran_file`` -- exactly one Fortran file and no C. A generic compile line, marked
  *unverified*: the special (graphics-free) PLATON recipe is published on a site this project could
  not read, so it is not inferred from the X11 recipe.
* ``makefile`` -- the source's own Makefile; review it before running, since it is the source's code.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_native_tools.toolchain import ToolchainReport

__all__ = ["SourceTree", "BuildStep", "BuildPlan", "scan_source", "plan_build", "DEFAULT_BUILD_ROOT"]

DEFAULT_BUILD_ROOT = Path.home() / ".crystalx" / "native-builds"
_FORTRAN = (".f", ".for", ".f77", ".f90", ".f95", ".f03")
_C = (".c",)
_HEADERS = (".h", ".inc")
_LICENCE_NAMES = ("license", "licence", "copying", "readme", "terms")
_MAX_FILES = 5000
_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class SourceTree:
    root: str
    fortran: tuple[str, ...]
    c: tuple[str, ...]
    headers: tuple[str, ...]
    makefiles: tuple[str, ...]
    licence_files: tuple[str, ...]
    executables_found: tuple[str, ...]
    other_files: int
    digest: str
    problems: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class BuildStep:
    purpose: str
    argv: tuple[str, ...]
    cwd: str

    def to_record(self) -> dict[str, object]:
        return {"purpose": self.purpose, "argv": list(self.argv), "cwd": self.cwd}


@dataclass(frozen=True)
class BuildPlan:
    tool: str
    recipe: str
    source: SourceTree | None
    build_dir: str
    output: str
    target_arch: str
    steps: tuple[BuildStep, ...]
    requirements: tuple[str, ...]
    blockers: tuple[str, ...]
    licence_acknowledgement: str | None
    executed: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_record(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "recipe": self.recipe,
            "source": self.source.to_record() if self.source else None,
            "build_dir": self.build_dir,
            "output": self.output,
            "target_arch": self.target_arch,
            "steps": [s.to_record() for s in self.steps],
            "requirements": list(self.requirements),
            "blockers": list(self.blockers),
            "ready": self.ready,
            "licence_acknowledgement": self.licence_acknowledgement,
            "executed": self.executed,
            "meaning": "a plan: nothing was compiled, copied or downloaded",
            "notes": list(self.notes),
        }


def scan_source(root: Path) -> SourceTree:
    """Classify the files under ``root`` (symlinks are not followed). Reads names and bytes only."""
    root = Path(os.path.abspath(root))
    fortran, c, headers, makefiles, licences, executables = [], [], [], [], [], []
    other = 0
    problems: list[str] = []
    digest = hashlib.sha256()
    total = 0
    count = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        linked = [d for d in dirs if (Path(directory) / d).is_symlink()]
        problems.extend(f"symlink ignored: {(Path(directory) / d).relative_to(root)}" for d in sorted(linked))
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in linked)
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink():
                problems.append(f"symlink ignored: {path.relative_to(root)}")
                continue
            count += 1
            if count > _MAX_FILES:
                problems.append(f"more than {_MAX_FILES} files; scan stopped")
                break
            relative = str(path.relative_to(root))
            try:
                size = path.stat().st_size
                total += size
                if total > _MAX_BYTES:
                    problems.append("source tree larger than the scan limit; scan stopped")
                    break
                with open(path, "rb") as handle:
                    head = handle.read(4)
                    digest.update(relative.encode() + b"\0")
                    digest.update(head)
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
            except OSError as exc:
                problems.append(f"unreadable: {relative} ({exc})")
                continue
            lowered = name.lower()
            if head in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\x7fELF"):
                executables.append(relative)
            elif lowered.endswith(_FORTRAN):
                fortran.append(relative)
            elif lowered.endswith(_C):
                c.append(relative)
            elif lowered.endswith(_HEADERS):
                headers.append(relative)
            elif lowered in ("makefile", "gnumakefile") or lowered.endswith(".mk"):
                makefiles.append(relative)
            elif lowered.startswith(_LICENCE_NAMES):
                licences.append(relative)
            else:
                other += 1
    return SourceTree(
        root=str(root),
        fortran=tuple(fortran),
        c=tuple(c),
        headers=tuple(headers),
        makefiles=tuple(makefiles),
        licence_files=tuple(licences),
        executables_found=tuple(executables),
        other_files=other,
        digest=digest.hexdigest(),
        problems=tuple(problems),
    )


def plan_build(
    tool: str,
    source_root: Path | None,
    *,
    toolchain: ToolchainReport,
    licence_ack: str | None = None,
    build_root: Path = DEFAULT_BUILD_ROOT,
) -> BuildPlan:
    blockers: list[str] = []
    notes: list[str] = []
    requirements: list[str] = []
    is_shelx = tool not in ("platon", "platon_special")
    if source_root is None or not Path(source_root).is_dir():
        if is_shelx:
            notes.append(
                "No SHELX source is visible locally, and current SHELX versions are distributed as "
                "executables (Spek, Acta Cryst. 2026, C82, 595-602). A native build needs source the "
                "owner obtains legitimately and may build; the installed x86_64 programs cannot be recompiled."
            )
        return BuildPlan(tool=tool, recipe="no_source", source=None, build_dir="", output="",
                         target_arch="arm64", steps=(), requirements=(),
                         blockers=("no source directory was given",), licence_acknowledgement=licence_ack,
                         notes=tuple(notes))

    tree = scan_source(Path(source_root))
    build_dir = Path(os.path.abspath(build_root)) / f"{tool}-{tree.digest[:12]}"
    output = build_dir / tool
    names = {Path(f).name.lower() for f in tree.fortran + tree.c}
    if tree.executables_found:
        notes.append(
            "Executables found in the source tree are ignored: compiled programs are not source and "
            "are never rebuilt from."
        )
    if tool == "platon" and {"platon.f", "xdrvr.c"} <= names:
        recipe = "platon_full_x11"
        compile_argv = ("gfortran", "-O2", "-o", str(output), "platon.f", "xdrvr.c",
                        "-I/opt/X11/include", "-L/opt/X11/lib", "-lX11")
        requirements.append("XQuartz (X11 headers and libX11 under /opt/X11)")
        if toolchain.xquartz_x11 is None:
            blockers.append("XQuartz's libX11 was not found; the graphics PLATON cannot be linked")
    elif tree.makefiles:
        recipe = "makefile"
        compile_argv = ("make", "-C", str(build_dir))
        notes.append("Uses the source's own Makefile: review it before running; it is code from the source.")
    elif len(tree.fortran) == 1 and not tree.c:
        recipe = "single_fortran_file"
        compile_argv = ("gfortran", "-O2", "-o", str(output), Path(tree.fortran[0]).name)
        notes.append(
            "Generic single-file Fortran compile: unverified. Check it against the source's own build "
            "instructions before running (the graphics-free PLATON recipe is not inferred from the X11 one)."
        )
    else:
        recipe = "unrecognised"
        compile_argv = ()
        blockers.append("no recognised build layout (one Fortran file, platon.f + xdrvr.c, or a Makefile)")

    requirements.append("an arm64 gfortran with Apple's linker and SDK (Command Line Tools or Xcode)")
    if not toolchain.can_build_fortran_arm64:
        blockers.append("no usable arm64 Fortran toolchain was found (see the toolchain inventory)")
    if not licence_ack or not licence_ack.strip():
        blockers.append(
            "no licence acknowledgement: state that the source's terms allow this local build "
            "(the acknowledgement is recorded in the plan)"
        )
    if is_shelx:
        notes.append("SHELX source is not publicly distributed; build only from source you may legally build.")

    steps: list[BuildStep] = []
    if compile_argv:
        steps = [
            BuildStep("copy the source into an isolated, user-local build directory (never the repository)",
                      ("copy-tree", tree.root, str(build_dir)), str(build_dir.parent)),
            BuildStep("compile for arm64", compile_argv, str(build_dir)),
            BuildStep("record the output's SHA-256 and check that it contains arm64 code",
                      ("inspect", str(output)), str(build_dir)),
        ]
    return BuildPlan(
        tool=tool,
        recipe=recipe,
        source=tree,
        build_dir=str(build_dir),
        output=str(output),
        target_arch="arm64",
        steps=tuple(steps),
        requirements=tuple(requirements),
        blockers=tuple(blockers),
        licence_acknowledgement=licence_ack.strip() if licence_ack else None,
        executed=False,
        notes=tuple(notes),
    )
