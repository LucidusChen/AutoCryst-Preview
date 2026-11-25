"""Parse SHELXT-style solution comments and tolerate an unverified listing dialect.

The solution-comment parser extracts the stated group, R1, Rweak, Alpha, systematic-absence
statistics and formula. Coverage uses newly hand-authored synthetic text, not expert-model
excerpts. Parsed values are observations to review rather than proof of a correct structure.

The ``.lxt`` scanner has not been checked against an authorised real process output. It keeps
unread lines and always reports ``format_verified=False``. Real output dialect verification
belongs in a private, ignored run, with separately invented public regression examples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ShelxtSolutionSummary",
    "ShelxtSpaceGroupTrial",
    "ShelxtListing",
    "parse_solution_rems",
    "scan_lxt",
]

# Exercised by hand-authored synthetic comment records; real dialect verification is separate.
_SOLUTION_RE = re.compile(
    r"SHELXT\s+solution\s+in\s+(?P<sg>\S+)\s*:\s*"
    r"R1\s+(?P<r1>[\d.]+)\s*,\s*Rweak\s+(?P<rweak>[\d.]+)\s*,\s*Alpha\s+(?P<alpha>[\d.]+)",
    re.IGNORECASE,
)
_ABSENCE_RE = re.compile(
    r"<I/s>\s+(?P<ios>[-\d.]+)\s+for\s+(?P<n>\d+)\s+systematic\s+absences", re.IGNORECASE
)
_FORMULA_RE = re.compile(r"Formula\s+found\s+by\s+SHELXT\s*:\s*(?P<formula>.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ShelxtSolutionSummary:
    """SHELXT's own account of the solution it wrote."""

    space_group: str | None = None
    r1: float | None = None
    r_weak: float | None = None
    alpha: float | None = None
    mean_i_over_sigma_absences: float | None = None
    n_systematic_absences: int | None = None
    formula: str | None = None
    raw_lines: tuple[str, ...] = ()

    @property
    def found_anything(self) -> bool:
        return self.space_group is not None or self.formula is not None

    def concerns(self) -> tuple[str, ...]:
        """Things a crystallographer should look at, stated rather than scored.

        No weighted quality number is produced here. These are separate observations; compressing
        them into one figure is what lets a wrong space group pass on a good R1.
        """
        notes: list[str] = []
        if self.r_weak is not None and self.r1 is not None and self.r_weak > 2 * self.r1:
            notes.append(
                f"Rweak ({self.r_weak:.3f}) is more than twice R1 ({self.r1:.3f}). Reflections "
                "this space group expects to be weak are not weak, which is the usual signature "
                "of a wrong space group rather than a bad structure."
            )
        if self.alpha is not None and self.alpha > 0.1:
            notes.append(f"Alpha is {self.alpha:.3f}; SHELXT reports 0.000 for a clean solution.")
        if (
            self.mean_i_over_sigma_absences is not None
            and self.n_systematic_absences
            and self.mean_i_over_sigma_absences > 3.0
        ):
            notes.append(
                f"{self.n_systematic_absences} reflections that this space group forbids have "
                f"<I/sigma> = {self.mean_i_over_sigma_absences:.2f}. A forbidden reflection that "
                "is strong means the space group is wrong."
            )
        return tuple(notes)


@dataclass(frozen=True)
class ShelxtSpaceGroupTrial:
    """One row of the ``.lxt`` scoring table. Field meanings unconfirmed; see the module docstring."""

    space_group: str
    r1: float | None = None
    r_weak: float | None = None
    alpha: float | None = None
    cfom: float | None = None
    raw: str = ""


@dataclass(frozen=True)
class ShelxtListing:
    """What could be read out of a ``.lxt``, and what could not."""

    source: Path
    trials: tuple[ShelxtSpaceGroupTrial, ...] = ()
    format_verified: bool = False
    """Always ``False`` until a real SHELXT listing has been read and this parser checked against
    it. Consumers must treat the contents as indicative, never as a decision input."""

    unparsed_lines: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = ()

    def best(self) -> ShelxtSpaceGroupTrial | None:
        scored = [t for t in self.trials if t.cfom is not None]
        return max(scored, key=lambda t: t.cfom or 0.0) if scored else None


def parse_solution_rems(lines: tuple[str, ...] | list[str]) -> ShelxtSolutionSummary:
    """Extract SHELXT's figures of merit from the ``REM`` records of a ``.res``.

    Accepts the ``rem_lines`` of a parsed file, or any iterable of lines. Anything not recognised is
    ignored rather than guessed at; the recognised lines are kept in ``raw_lines`` so the record
    shows what the numbers were read from.
    """
    summary: dict[str, object] = {}
    raw: list[str] = []

    for line in lines:
        text = line.strip()
        if not text:
            continue
        matched = False

        solution = _SOLUTION_RE.search(text)
        if solution:
            summary["space_group"] = solution.group("sg").rstrip(":")
            summary["r1"] = float(solution.group("r1"))
            summary["r_weak"] = float(solution.group("rweak"))
            summary["alpha"] = float(solution.group("alpha"))
            matched = True

        absence = _ABSENCE_RE.search(text)
        if absence:
            summary["mean_i_over_sigma_absences"] = float(absence.group("ios"))
            summary["n_systematic_absences"] = int(absence.group("n"))
            matched = True

        formula = _FORMULA_RE.search(text)
        if formula:
            summary["formula"] = formula.group("formula").strip()
            matched = True

        if matched:
            raw.append(text)

    return ShelxtSolutionSummary(**summary, raw_lines=tuple(raw))  # type: ignore[arg-type]


def scan_lxt(path: Path | str) -> ShelxtListing:
    """Read a SHELXT ``.lxt`` listing as far as it can be read.

    IDE-FILL(lxt-table): tighten the row parsing **after** the first live SHELXT run, using the
    real listing as the specification.

    What this does today: finds lines that look like a space-group symbol followed by numeric
    columns, and records everything else as unparsed. What it deliberately does not do: assume a
    column order. ``format_verified`` stays ``False``, and a caller must not make a decision on
    this.

    When the real listing is in hand:
      * Confirm the column order and set ``format_verified=True`` **only then**.
      * Keep real listing text in the ignored private run; construct new synthetic rows for
        public regression tests without copying private values or identifiers.
      * Keep the tolerant path as a fallback for other SHELXT versions.
    """
    target = Path(path)
    if not target.is_file():
        return ShelxtListing(
            source=target, warnings=(f"No listing at {target}; SHELXT may not have written one.",)
        )

    trials: list[ShelxtSpaceGroupTrial] = []
    unparsed: list[str] = []
    # A space-group symbol as SHELX writes it -- "P2(1)/m", "P -1", "C 1 2/c 1", "Fdd2" -- followed
    # by numeric columns. Parentheses and embedded spaces are both common and both were missing
    # from the first version of this pattern, which silently dropped every "P2(1)" row.
    row = re.compile(
        r"^\s*(?P<sg>[PABCIFR][A-Za-z0-9\-/()\s]{0,18}?)\s{2,}(?P<numbers>[-\d.]+(?:\s+[-\d.]+)*)\s*$"
    )

    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.rstrip()
        if not text.strip():
            continue
        match = row.match(text)
        if not match:
            unparsed.append(text)
            continue
        numbers = [float(n) for n in match.group("numbers").split() if _is_number(n)]
        if len(numbers) < 2:
            unparsed.append(text)
            continue
        trials.append(
            ShelxtSpaceGroupTrial(
                space_group=match.group("sg").strip(),
                r1=numbers[0] if len(numbers) > 0 else None,
                r_weak=numbers[1] if len(numbers) > 1 else None,
                alpha=numbers[2] if len(numbers) > 2 else None,
                cfom=numbers[3] if len(numbers) > 3 else None,
                raw=text,
            )
        )

    return ShelxtListing(
        source=target,
        trials=tuple(trials),
        format_verified=False,
        unparsed_lines=tuple(unparsed),
        warnings=(
            "The .lxt column layout has not been verified against a real SHELXT run. Treat these "
            "rows as indicative only; do not make a selection decision from them.",
        ),
    )


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
