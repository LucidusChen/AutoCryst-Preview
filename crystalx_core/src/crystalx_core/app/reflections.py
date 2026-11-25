"""Which reflections the bounded SHELXL refinement may use: one contract for the pipeline and verify-run.

SHELXT may change the cell setting of its solution. When it does, the reflections must be
re-indexed to match, and SHELXT writes them beside the solution as ``<stem>_<letter>.hkl``. That
file is the normal source, and it is fully checked: written by SHELXT (recorded in its provenance
with a digest), copied unchanged into the refinement directory, read by SHELXL.

When SHELXT writes no such file, the original reflections fit the solution only if the solution is
still in the input orientation. The original file is then accepted on two conditions, both kept as
evidence:

1. the chosen solution's ``REM`` lines say "Orientation as input". This is SHELXT's own statement,
   read from text whose exact format has **not been verified against real SHELXT output** on this
   project's machines. The synthetic demo writes the phrase itself, so tests built on it prove the
   contract between the pipeline and the checker, not SHELXT's format;
2. the solution's ``CELL`` equals the ``CELL`` SHELXT was given, within rounding, so the statement
   is at least consistent with the files. Agreement of the six cell parameters alone does not
   establish identical indexing or orientation; it never substitutes for the separate statement.

Anything else blocks the refinement. The pipeline calls :func:`decide_from_files` before it
refines and records the decision with digests; ``crystalx verify-run`` calls the same function on
the retained files and compares, so the two cannot disagree about what was legal (Codex recheck
F1, 2026-10-22).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from crystalx_core.errors import CrystalXError
from crystalx_core.io.shelx_reader import read_ins_header
from crystalx_core.io.shelx_res import read_res

__all__ = [
    "CONTRACT",
    "ORIENTATION_MARKER",
    "CELL_TOLERANCE_A",
    "CELL_TOLERANCE_DEG",
    "ReflectionSource",
    "ReflectionDecision",
    "solution_model_name",
    "solution_reflections_name",
    "input_reflections_name",
    "orientation_statement",
    "cell_difference",
    "decide_from_files",
    "required_shelxt_outputs",
]

CONTRACT = "crystalx.reflections/1"

#: Matched case-insensitively within a REM line of the chosen solution. Unverified wording.
ORIENTATION_MARKER = "orientation as input"

#: Rounding allowance when comparing the solution's CELL with the input CELL. SHELXT does not
#: refine the cell, so an unchanged orientation leaves the parameters equal up to formatting.
CELL_TOLERANCE_A = 0.002
CELL_TOLERANCE_DEG = 0.02


class ReflectionSource(str, enum.Enum):
    SHELXT_MATCHING = "shelxt_matching_hkl"
    """SHELXT's own ``<stem>_<letter>.hkl``, in the same setting as the solution."""

    ORIGINAL_ORIENTATION_AS_INPUT = "original_orientation_as_input"
    """The original reflections, because SHELXT wrote none and states the orientation is unchanged."""


def solution_model_name(stem: str, letter: str) -> str:
    return f"{stem}_{letter}.res"


def solution_reflections_name(stem: str, letter: str) -> str:
    return f"{stem}_{letter}.hkl"


def input_reflections_name(stem: str) -> str:
    """SHELXT's own input copy of the reflections, in its branch directory."""
    return f"{stem}.hkl"


def orientation_statement(rem_lines: Iterable[str]) -> str | None:
    """The first REM line carrying :data:`ORIENTATION_MARKER`, or ``None``."""
    for line in rem_lines:
        if ORIENTATION_MARKER in line.lower():
            return line.strip()
    return None


def _cell_tuple(cell) -> tuple[float, float, float, float, float, float] | None:
    if cell is None:
        return None
    return (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)


def cell_difference(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    """Largest differences, using decimal values so a tolerance boundary stays inclusive.

    CELL values originate as decimal text. Binary subtraction can put an exact documented
    boundary (e.g. 10.002 minus 10) just above 0.002. This does not add another tolerance.
    """
    differences = [abs(Decimal(str(a)) - Decimal(str(b))) for a, b in zip(first, second)]
    return float(max(differences[:3])), float(max(differences[3:6]))


@dataclass(frozen=True)
class ReflectionDecision:
    source: ReflectionSource | None
    """``None`` means refused: the refinement must not run."""

    file_name: str
    """The reflection file to refine against, inside SHELXT's branch directory ('' if refused)."""

    reason: str
    orientation_statement: str | None = None
    solution_cell: tuple[float, ...] | None = None
    input_cell: tuple[float, ...] | None = None
    length_difference_a: float | None = None
    angle_difference_deg: float | None = None

    @property
    def allowed(self) -> bool:
        return self.source is not None

    @property
    def is_fallback(self) -> bool:
        return self.source is ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT

    @property
    def note(self) -> str:
        if self.source is ReflectionSource.SHELXT_MATCHING:
            return f"reflections from SHELXT's own {self.file_name}"
        if self.is_fallback:
            return (
                "original reflections (SHELXT states 'Orientation as input' and the cells agree; this "
                "fallback is not yet verified against real SHELXT output)"
            )
        return "refused"

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "contract": CONTRACT,
            "source": self.source.value if self.source else None,
            "allowed": self.allowed,
            "file": self.file_name,
            "evidence": self.reason,
        }
        if self.orientation_statement is not None or self.is_fallback:
            record["orientation_statement"] = self.orientation_statement
            record["statement_format_verified"] = False
        if self.solution_cell is not None or self.input_cell is not None:
            record["cell_check"] = {
                "solution": list(self.solution_cell) if self.solution_cell else None,
                "input": list(self.input_cell) if self.input_cell else None,
                "max_length_difference_a": self.length_difference_a,
                "max_angle_difference_deg": self.angle_difference_deg,
                "tolerance": {"length_a": CELL_TOLERANCE_A, "angle_deg": CELL_TOLERANCE_DEG},
            }
        return record


def _refused(reason: str, **evidence) -> ReflectionDecision:
    return ReflectionDecision(source=None, file_name="", reason=reason, **evidence)


def decide_from_files(branch_dir: Path, stem: str, letter: str) -> ReflectionDecision:
    """Decide from the files in SHELXT's branch directory. Reads; never writes or runs anything."""
    branch_dir = Path(branch_dir)
    matching = solution_reflections_name(stem, letter)
    if (branch_dir / matching).is_file():
        return ReflectionDecision(
            source=ReflectionSource.SHELXT_MATCHING,
            file_name=matching,
            reason=f"SHELXT wrote {matching} beside the chosen solution",
        )

    model = solution_model_name(stem, letter)
    try:
        solution = read_res(branch_dir / model)
    except (OSError, ValueError, CrystalXError) as exc:
        return _refused(f"SHELXT did not write {matching}, and the chosen solution {model} could not be read: {exc}")
    statement = orientation_statement(solution.rem_lines)
    if statement is None:
        return _refused(
            f"SHELXT did not write {matching} and does not state that the orientation is unchanged, "
            "so the original reflections may not match the solution's cell setting. Refinement is "
            "not attempted."
        )
    try:
        given = read_ins_header(branch_dir / f"{stem}.ins")
    except (OSError, ValueError, CrystalXError) as exc:
        return _refused(
            f"SHELXT did not write {matching}; its solution states {statement!r}, but SHELXT's input "
            f"{stem}.ins could not be read to compare the cells: {exc}",
            orientation_statement=statement,
        )
    solution_cell = _cell_tuple(solution.header.cell)
    input_cell = _cell_tuple(given.cell)
    if solution_cell is None or input_cell is None:
        return _refused(
            f"SHELXT did not write {matching}; its solution states {statement!r}, but a CELL record is "
            "missing, so the orientation statement cannot be checked against the files.",
            orientation_statement=statement,
            solution_cell=solution_cell,
            input_cell=input_cell,
        )
    if not all(math.isfinite(value) for value in (*solution_cell, *input_cell)):
        return _refused(
            f"SHELXT did not write {matching}; a CELL parameter is not finite, so the original "
            "reflections are not used.",
            orientation_statement=statement,
        )
    lengths, angles = cell_difference(solution_cell, input_cell)
    evidence = dict(
        orientation_statement=statement,
        solution_cell=solution_cell,
        input_cell=input_cell,
        length_difference_a=round(lengths, 6),
        angle_difference_deg=round(angles, 6),
    )
    if lengths > CELL_TOLERANCE_A or angles > CELL_TOLERANCE_DEG:
        return _refused(
            f"SHELXT did not write {matching}; its solution states {statement!r}, but its CELL differs "
            f"from the input CELL by up to {lengths:.4f} A / {angles:.3f} deg (allowed "
            f"{CELL_TOLERANCE_A} A / {CELL_TOLERANCE_DEG} deg). The original reflections are not used.",
            **evidence,
        )
    return ReflectionDecision(
        source=ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT,
        file_name=input_reflections_name(stem),
        reason=(
            f"SHELXT wrote no {matching}; the chosen solution states {statement!r} and its CELL "
            f"matches the input CELL within {CELL_TOLERANCE_A} A / {CELL_TOLERANCE_DEG} deg"
        ),
        **evidence,
    )


def required_shelxt_outputs(source: ReflectionSource | str | None, stem: str, letter: str) -> tuple[str, ...]:
    """What SHELXT itself must have written for the recorded reflection source.

    Unknown or missing sources get the strict normal-path requirement.
    """
    try:
        resolved = ReflectionSource(source) if source is not None else ReflectionSource.SHELXT_MATCHING
    except ValueError:
        resolved = ReflectionSource.SHELXT_MATCHING
    model = solution_model_name(stem, letter)
    if resolved is ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT:
        return (model,)
    return (model, solution_reflections_name(stem, letter))
