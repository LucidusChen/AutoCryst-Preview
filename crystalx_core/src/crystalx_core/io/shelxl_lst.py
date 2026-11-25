"""Reading a SHELXL listing (``.lst``) for the numbers a preliminary refinement can honestly report.

**Format status: unverified against this installation.** The patterns below follow SHELXL's
long-standing listing conventions and selected locally observed output patterns. Overall dialect
coverage remains unverified, so every field is optional and :attr:`ShelxlListing.matched` records exactly which patterns fired. A field
that did not match is ``None`` -- never zero, never a default -- and the user interface shows it as
unknown.

The ``.res`` that SHELXL writes carries the same headline numbers in ``REM`` lines, which
:mod:`crystalx_core.io.shelx_res` already reads. The two are parsed independently and compared;
a disagreement is reported rather than resolved by preferring one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ShelxlListing", "parse_lst", "parse_lst_text"]

_R1 = re.compile(
    r"R1\s*=\s*([0-9.]+)\s+for\s+(\d+)\s+Fo\s*>\s*4sig\(Fo\)\s+and\s+([0-9.]+)\s+for\s+all\s+(\d+)\s+data",
    re.IGNORECASE,
)
_WR2_FINAL = re.compile(
    r"wR2\s*=\s*([0-9.]+)\s*,\s*GooF\s*=\s*S\s*=\s*([0-9.]+)\s*,\s*Restrained\s+GooF\s*=\s*([0-9.]+)\s+for\s+all\s+data",
    re.IGNORECASE,
)
_SHIFT = re.compile(
    r"Mean\s+shift/(?:esd|su)\s*=\s*([-0-9.]+)"
    r"(?:\s+Maximum\s*=\s*([-0-9.]+)(?:\s+for\s+(.+?))?)?\s*$",
    re.IGNORECASE,
)
_CYCLE = re.compile(r"Least-squares\s+cycle\s+(\d+)", re.IGNORECASE)
_PEAK = re.compile(r"Highest\s+peak\s+([-0-9.]+)", re.IGNORECASE)
_HOLE = re.compile(r"Deepest\s+hole\s+([-0-9.]+)", re.IGNORECASE)
_PARAMS = re.compile(r"(\d+)\s*/\s*(\d+)\s+parameters", re.IGNORECASE)
_STARRED = re.compile(r"^\s*\*\*\s*(.+?)\s*\*\*\s*$")
_WARNING = re.compile(r"^\s*(?:Warning[:!]?\s+.+|.*NON-POSITIVE DEFINITE.*)$", re.IGNORECASE)

# SHELXL marks both fatal errors and advice with ``** ... **`` -- for example the familiar
# "** Extinction (EXTI) or solvent water (SWAT) correction may be required **", which is advice.
# Treating every starred line as fatal would fail good refinements; treating none as fatal would
# accept broken ones. The keyword classification below is a heuristic, UNVERIFIED against real
# output from this installation: advice first, then recognised failure words, and anything else
# is kept as an unclassified notice that is shown to the user but neither fails nor passes the
# step on its own. (A SHELXL run that really aborted also leaves its expected output files
# missing, which fails the step independently of this text.)
_ADVISORY_WORDS = re.compile(
    r"\b(?:WARNING|MAY\s+BE\s+REQUIRED|PROBABLY|CONSIDER|SUGGEST\w*|NPD|NON-POSITIVE)\b",
    re.IGNORECASE,
)
_FATAL_WORDS = re.compile(
    r"\b(?:ERROR|FATAL|ABORT\w*|CANNOT|ILLEGAL|MISSING|INSUFFICIENT|UNRECOGNI[SZ]ED|NOT\s+ALLOWED|"
    r"NOT\s+FOUND|INCOMPATIBLE|TOO\s+MANY|ARRAYS\s+TOO\s+SMALL)\b",
    re.IGNORECASE,
)
# This exact SHELXL notice reports inconclusive absolute-hand evidence after a
# completed refinement, not a failed command. Keep it visible for scientific
# review without weakening CANNOT/ERROR classification for any other message.
_EXPLICIT_REVIEW_ONLY = frozenset({"absolute structure cannot be determined reliably"})
_EXPLICIT_FATAL = frozenset({"second damp parameter too small for acta"})

# Convergence is deliberately NOT assessed. A small mean shift/su can hide a large individual
# shift (IUCr checkCIF PLAT080 and the SHFSU01 procedure concern the *maximum* shift/su over all
# refined parameters), and no convergence policy has been agreed or verified against real SHELXL
# output on this project's machines. The raw mean and maximum are reported side by side instead.


@dataclass(frozen=True)
class ShelxlListing:
    source: Path | None
    r1_observed: float | None = None
    n_observed: int | None = None
    r1_all: float | None = None
    n_all: int | None = None
    wr2: float | None = None
    goof: float | None = None
    restrained_goof: float | None = None
    mean_shift_su: float | None = None
    max_shift_su: float | None = None
    max_shift_for: str | None = None
    cycles_run: int | None = None
    highest_peak: float | None = None
    deepest_hole: float | None = None
    parameters_refined: int | None = None
    errors: tuple[str, ...] = ()
    """Starred messages with a recognised failure word. They fail the refinement step."""

    warnings: tuple[str, ...] = ()
    """Advice and warnings (starred or not). Shown; they do not fail the step."""

    notices: tuple[str, ...] = ()
    """Starred messages the heuristic cannot classify. Shown for review; they do not decide."""

    matched: tuple[str, ...] = field(default_factory=tuple)
    format_verified: bool = False
    """``False`` until a real SHELXL listing from this installation has been checked against
    these patterns. Consumers display the values; they do not gate on them."""

    @property
    def converged(self) -> bool | None:
        """Always ``None``: convergence is not assessed (see the module comment).

        Kept as a property so a later, documented policy can be introduced in one place. Until
        then a caller must show :attr:`mean_shift_su` and :attr:`max_shift_su` as raw numbers.
        """
        return None

    @property
    def has_fatal_error(self) -> bool:
        return bool(self.errors)

    def to_record(self) -> dict[str, object]:
        return {
            "r1_observed": self.r1_observed,
            "n_observed": self.n_observed,
            "r1_all": self.r1_all,
            "n_all": self.n_all,
            "wr2": self.wr2,
            "goof": self.goof,
            "restrained_goof": self.restrained_goof,
            "mean_shift_su": self.mean_shift_su,
            "max_shift_su": self.max_shift_su,
            "max_shift_for": self.max_shift_for,
            "cycles_run": self.cycles_run,
            "highest_peak": self.highest_peak,
            "deepest_hole": self.deepest_hole,
            "parameters_refined": self.parameters_refined,
            "convergence": "not_assessed",
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "notices": list(self.notices),
            "message_classification": "keyword heuristic, unverified against real SHELXL output",
            "matched_patterns": list(self.matched),
            "format_verified": self.format_verified,
        }


def parse_lst(path: Path) -> ShelxlListing:
    if not path.is_file():
        return ShelxlListing(source=path, errors=(f"listing not found: {path.name}",))
    return parse_lst_text(path.read_text(encoding="utf-8", errors="replace"), source=path)


def parse_lst_text(text: str, *, source: Path | None = None) -> ShelxlListing:
    """Take the **last** occurrence of each figure, which is the final cycle's value."""
    values: dict[str, object] = {}
    matched: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    notices: list[str] = []
    cycles = 0

    for line in text.splitlines():
        if (m := _R1.search(line)) is not None:
            values.update(
                r1_observed=float(m.group(1)),
                n_observed=int(m.group(2)),
                r1_all=float(m.group(3)),
                n_all=int(m.group(4)),
            )
            matched.append("r1")
        if (m := _WR2_FINAL.search(line)) is not None:
            values.update(
                wr2=float(m.group(1)), goof=float(m.group(2)), restrained_goof=float(m.group(3))
            )
            matched.append("wr2_goof")
        if (m := _SHIFT.search(line)) is not None:
            values.update(
                mean_shift_su=float(m.group(1)),
                max_shift_su=float(m.group(2)) if m.group(2) is not None else None,
                max_shift_for=(m.group(3) or "").strip() or None,
            )
            matched.append("shift")
        if (m := _CYCLE.search(line)) is not None:
            cycles = max(cycles, int(m.group(1)))
        if (m := _PEAK.search(line)) is not None:
            values["highest_peak"] = float(m.group(1))
            matched.append("peak")
        if (m := _HOLE.search(line)) is not None:
            values["deepest_hole"] = float(m.group(1))
            matched.append("hole")
        if (m := _PARAMS.search(line)) is not None:
            active = int(m.group(1))
            # SHELXL prints an additional zero-active evaluation after the last
            # LS cycle (e.g. 0 / 145). It does not erase the parameters refined
            # in the preceding actual cycle. A zero-only listing stays zero.
            if active or "parameters_refined" not in values:
                values["parameters_refined"] = active
        if (m := _STARRED.match(line)) is not None:
            message = m.group(1)
            normalized = " ".join(message.split()).casefold()
            if normalized in _EXPLICIT_FATAL:
                errors.append(message)
            elif normalized in _EXPLICIT_REVIEW_ONLY or _ADVISORY_WORDS.search(message):
                warnings.append(message)
            elif _FATAL_WORDS.search(message):
                errors.append(message)
            else:
                notices.append(message)
        elif _WARNING.match(line):
            warnings.append(line.strip())

    if cycles:
        values["cycles_run"] = cycles
        matched.append("cycles")
    return ShelxlListing(
        source=source,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings))[:50],
        notices=tuple(dict.fromkeys(notices))[:50],
        matched=tuple(dict.fromkeys(matched)),
        **values,  # type: ignore[arg-type]
    )
