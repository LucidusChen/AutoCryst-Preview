"""Run-time resources the PLATON validation mode depends on: ``check.def`` and X11 libraries.

Three facts are kept apart, as the owner asked: whether a file **exists**, whether the program
**can start**, and whether a validation run **actually succeeded**. This module only answers the
first (plus header facts). Starting is the opt-in launch probe; success is a real run whose ``.chk``
was parsed.

``check.def`` (PLATON's documentation, local copies in ``platon_instruction/``): validation needs it
in the working directory or at ``$CHECKDEF``; its ``yymmdd`` version should match the executable;
when PLATON finds no suitable file it writes one into the working directory and uses that. The
version dates below are found heuristically (a six-digit date token) and say so.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from pathlib import Path

from crystalx_core.workspace.hashing import sha256_file

from crystalx_native_tools.inspect import FileInspection

__all__ = ["CheckDefCandidate", "PlatonResources", "inspect_platon_resources", "PIPELINE_CHECKDEF_BEHAVIOUR"]

PIPELINE_CHECKDEF_BEHAVIOUR = (
    "In a run, PLATON starts in a fresh validation directory with CHECKDEF unset. PLATON is "
    "documented to create a matching check.def there when it finds none. That has not yet been "
    "observed with this executable on this machine."
)

_DATE = re.compile(rb"(?<![0-9])([0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(?![0-9])")
_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{6,120}")
#: gfortran's run-time checks embed "At line N of file x.f" / "around line N"; N is no date.
_LINE_NUMBER_BEFORE = re.compile(rb"(?i)\bline\s*$")
_MAX_SCAN_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class CheckDefCandidate:
    path: str
    origin: str
    """``config``, ``environment``, ``beside_executable``, ``toolbox`` or ``usr_local_bin``."""

    exists: bool
    readable: bool = False
    size: int | None = None
    sha256: str | None = None
    version_date: str | None = None
    note: str = ""

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PlatonResources:
    executable: str | None
    embedded_dates: tuple[str, ...]
    """Six-digit ``yymmdd`` tokens found near 'PLATON'/'version' text in the executable (heuristic)."""

    checkdef: tuple[CheckDefCandidate, ...]
    version_match: str
    """``match``, ``mismatch`` or ``unknown`` -- all heuristic."""

    needs_x11: bool | None
    x11_available: bool | None
    pipeline_behaviour: str = PIPELINE_CHECKDEF_BEHAVIOUR
    notes: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        return {
            "executable": self.executable,
            "embedded_dates": list(self.embedded_dates),
            "checkdef": [c.to_record() for c in self.checkdef],
            "version_match": self.version_match,
            "version_match_method": "heuristic six-digit date tokens; not verified against PLATON's own check",
            "needs_x11": self.needs_x11,
            "x11_available": self.x11_available,
            "pipeline_behaviour": self.pipeline_behaviour,
            "notes": list(self.notes),
            "levels": {
                "file_exists": "reported here",
                "can_start": "only the opt-in launch probe can say",
                "validation_succeeded": "only a real validation run with a parsed .chk can say",
            },
        }


def _valid(token: tuple[bytes, bytes, bytes]) -> str | None:
    yy, mm, dd = (int(part) for part in token)
    try:
        _dt.date(2000 + yy, mm, dd)
    except ValueError:
        return None
    return f"{yy:02d}{mm:02d}{dd:02d}"


def embedded_platon_dates(path: Path, *, limit: int = 8) -> tuple[str, ...]:
    """Heuristic: dates in printable strings that also mention PLATON or a version."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            data = handle.read(min(size, _MAX_SCAN_BYTES))
    except OSError:
        return ()
    found: list[str] = []
    for run in _PRINTABLE_RUN.finditer(data):
        text = run.group(0)
        lowered = text.lower()
        if b"platon" not in lowered and b"version" not in lowered:
            continue
        for match in _DATE.finditer(text):
            if _LINE_NUMBER_BEFORE.search(text, 0, match.start()):
                continue  # "At line 130321 of file platon_special.f": a source line, not a date
            date = _valid(match.groups())
            if date and date not in found:
                found.append(date)
                if len(found) >= limit:
                    return tuple(found)
    return tuple(found)


def checkdef_version(path: Path) -> str | None:
    """Heuristic: the first valid ``yymmdd`` token in the first ten lines."""
    try:
        with open(path, "rb") as handle:
            head = b"".join(handle.readline() for _ in range(10))
    except OSError:
        return None
    for match in _DATE.finditer(head):
        date = _valid(match.groups())
        if date:
            return date
    return None


def _candidate(path: Path, origin: str) -> CheckDefCandidate:
    if not path.is_file():
        return CheckDefCandidate(path=str(path), origin=origin, exists=False)
    try:
        return CheckDefCandidate(
            path=str(path),
            origin=origin,
            exists=True,
            readable=True,
            size=path.stat().st_size,
            sha256=sha256_file(path),
            version_date=checkdef_version(path),
        )
    except OSError as exc:
        return CheckDefCandidate(path=str(path), origin=origin, exists=True, note=f"unreadable: {exc}")


def inspect_platon_resources(
    inspection: FileInspection | None,
    *,
    tool: str,
    configured_checkdef: Path | None,
    toolbox: Path | None,
    environ: dict[str, str] | None = None,
    on_macos: bool,
    x11_library: Path = Path("/opt/X11/lib/libX11.6.dylib"),
) -> PlatonResources:
    environ = dict(os.environ if environ is None else environ)
    executable = Path(inspection.path) if inspection is not None and inspection.exists else None
    seen: set[str] = set()
    candidates: list[CheckDefCandidate] = []

    def add(path: Path | None, origin: str) -> None:
        if path is None:
            return
        key = os.path.abspath(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(_candidate(Path(key), origin))

    add(configured_checkdef, "config")
    if environ.get("CHECKDEF"):
        add(Path(environ["CHECKDEF"]), "environment")
    if executable is not None:
        add(Path(os.path.realpath(executable)).parent / "check.def", "beside_executable")
    if toolbox is not None:
        add(toolbox / "check.def", "toolbox")
    if on_macos:
        add(Path("/usr/local/bin/check.def"), "usr_local_bin")

    dates = embedded_platon_dates(executable) if executable is not None else ()
    existing = [c for c in candidates if c.exists and c.version_date]
    if not dates or not existing:
        match = "unknown"
    elif any(c.version_date in dates for c in existing):
        match = "match"
    else:
        match = "mismatch"

    needs_x11 = None
    if inspection is not None and inspection.dependencies:
        needs_x11 = any("libX11" in d.name for d in inspection.dependencies)
    elif inspection is not None and inspection.macho is not None:
        needs_x11 = False
    x11_available = x11_library.is_file() if on_macos else None
    notes: list[str] = []
    if tool == "platon_special" and needs_x11:
        notes.append("this 'special' build links X11 libraries, unlike a graphics-free build")
    if needs_x11 and on_macos and not x11_available:
        notes.append(f"{x11_library} not found: XQuartz appears to be missing, so this build cannot start")
    if not any(c.exists for c in candidates):
        notes.append("no check.def found; PLATON is documented to create one in its working directory")
    return PlatonResources(
        executable=str(executable) if executable else None,
        embedded_dates=dates,
        checkdef=tuple(candidates),
        version_match=match,
        needs_x11=needs_x11,
        x11_available=x11_available,
        notes=tuple(notes),
    )
