"""PLATON structure diagnostics: the invocation contract and the ALERT report.

## Invocation, from PLATON's own documentation

The README shipped with the owner's PLATON distribution (``platon_instruction/README``) states::

    platon -u sk5555.cif        ->  the check-output can be found in: sk5555.chk

and that ``check.def`` must be "present in the current directory" or located through the
``CHECKDEF`` environment variable. Later PLATON versions carry a built-in ``check.def`` and create
one in the current directory when none is found (Spek 2018, 2020). This adapter therefore runs
PLATON in an isolated stage directory with ``CHECKDEF`` *unset*, so a stale ``check.def`` elsewhere
on the machine cannot be picked up silently, and whatever PLATON writes stays inside the run.

``platon_special`` -- the build without X11 graphics, intended for validation -- is preferred, with
the standard ``platon`` as fallback; ``DISPLAY`` is removed either way.

## Prerequisites, made explicit

* ``<stem>.cif`` -- **required**. Written by SHELXL when ``ACTA`` is set.
* ``<stem>.fcf`` -- **required by this workflow**. Spek (Acta Cryst. C82, 2026) notes that when no
  FCF is present PLATON "launches an external SHELXL job to reconstruct the final refinement
  result". On this machine that would be a second, uncontrolled x86_64 process outside the run's
  limits and provenance, so a missing FCF *blocks* the diagnostic with an explanation instead.

## The report

``.chk`` lines carrying ALERTs include prefixed checkCIF codes such as ``PLAT220_ALERT_2_C``
and locally observed three-digit codes such as ``220_ALERT_2_C``: a code, an alert *type* 1-5,
and a *level* A/B/C/G. Selected local patterns have been checked, but overall dialect coverage
remains unverified. The parser keeps an unrecognised ALERT-line sample and sets
``format_verified=False``.

An ALERT is a prompt to look, not a verdict. Zero level-A alerts on a four-cycle isotropic skeleton
is not validation, and the user interface says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.tools.runner import ResourceLimits

__all__ = [
    "PlatonAlert",
    "PlatonReport",
    "PlatonSettings",
    "validation_prerequisites",
    "parse_chk",
    "parse_chk_text",
    "ALERT_LEVEL_MEANING",
]

_ALERT = re.compile(r"^\s*([A-Z][A-Z0-9]{2,7}|[0-9]{3})_ALERT_([1-5])_([ABCG])\b\s*(.*?)\s*$")

#: The conventional checkCIF meanings, for display only.
ALERT_LEVEL_MEANING = {
    "A": "most likely a serious problem; resolve or explain",
    "B": "a potentially serious problem; consider carefully",
    "C": "check; ensure it is not caused by an omission or oversight",
    "G": "general information; check that it is not something unexpected",
}


@dataclass(frozen=True)
class PlatonSettings:
    timeout_s: float = 600.0
    limits: ResourceLimits | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "timeout_s": self.timeout_s,
            "limits": self.limits.to_record() if self.limits else None,
            "argv_shape": ["<platon>", "-u", "<stem>.cif"],
            "env_unset": ["DISPLAY", "CHECKDEF"],
        }


@dataclass(frozen=True)
class PlatonAlert:
    code: str
    alert_type: int
    level: str
    text: str

    @property
    def token(self) -> str:
        return f"{self.code}_ALERT_{self.alert_type}_{self.level}"


@dataclass(frozen=True)
class PlatonReport:
    source: Path | None
    alerts: tuple[PlatonAlert, ...] = ()
    format_verified: bool = False
    lines_read: int = 0
    unrecognised_sample: tuple[str, ...] = field(default_factory=tuple)

    def count(self, level: str) -> int:
        return sum(1 for a in self.alerts if a.level == level)

    @property
    def counts(self) -> dict[str, int]:
        return {level: self.count(level) for level in "ABCG"}

    def to_record(self) -> dict[str, object]:
        return {
            "counts": self.counts,
            "alerts": [
                {"token": a.token, "level": a.level, "type": a.alert_type, "text": a.text}
                for a in self.alerts
            ],
            "format_verified": self.format_verified,
            "lines_read": self.lines_read,
            "unrecognised_sample": list(self.unrecognised_sample),
            "meaning": ALERT_LEVEL_MEANING,
        }


def validation_prerequisites(workdir: Path, stem: str) -> list[str]:
    """Return one human-readable reason per missing prerequisite. Empty means ready."""
    missing: list[str] = []
    if not (workdir / f"{stem}.cif").is_file():
        missing.append(
            f"{stem}.cif is missing. PLATON validates a CIF; SHELXL writes one only when ACTA is "
            "set and the refinement completes."
        )
    if not (workdir / f"{stem}.fcf").is_file():
        missing.append(
            f"{stem}.fcf is missing. Without it PLATON would launch its own external SHELXL job "
            "to regenerate structure factors -- an uncontrolled process outside this run's limits "
            "and provenance -- so the diagnostic is blocked instead. SHELXL writes the FCF when "
            "LIST 4 is set."
        )
    return missing


def parse_chk(path: Path) -> PlatonReport:
    if not path.is_file():
        return PlatonReport(source=path)
    return parse_chk_text(path.read_text(encoding="utf-8", errors="replace"), source=path)


def parse_chk_text(text: str, *, source: Path | None = None) -> PlatonReport:
    alerts: list[PlatonAlert] = []
    unrecognised: list[str] = []
    lines = text.splitlines()
    for line in lines:
        match = _ALERT.match(line)
        if match:
            alerts.append(
                PlatonAlert(
                    code=match.group(1),
                    alert_type=int(match.group(2)),
                    level=match.group(3),
                    text=match.group(4),
                )
            )
        elif "ALERT" in line.upper() and len(unrecognised) < 20:
            unrecognised.append(line.strip())
    return PlatonReport(
        source=source,
        alerts=tuple(alerts),
        lines_read=len(lines),
        unrecognised_sample=tuple(unrecognised),
    )
