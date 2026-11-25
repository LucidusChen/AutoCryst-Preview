"""Symmetry facts that can be decided exactly, from a SHELX instruction file alone.

This module deliberately does **not** try to be a symmetry engine. It answers one question, and
answers it exactly:

    Does the declared space group contain any improper operation?

That question needs nothing but the ``LATT`` sign and the rotation part of each ``SYMM`` record, and
it decides a hard physical constraint:

    **A homochiral material cannot crystallise in a space group containing an improper operation.**

An inversion centre, a mirror, a glide plane or a rotoinversion axis all map a molecule onto its
mirror image. If every molecule in the crystal has the same handedness -- which is true of any
peptide built from L-amino acids, and of any enantiopure ligand -- there is nothing for those
operations to map onto. The 65 space groups that contain only proper rotations are the **Sohncke**
groups; a homochiral crystal must be in one of them.

A hand-authored synthetic example with ``LATT 1`` is centrosymmetric even when every explicit
``SYMM`` record is a proper rotation: SHELX adds inversion implicitly. A declared homochiral
prior therefore requires review of that input symmetry. This generic constraint does not infer
the correct space group or replace examination of diffraction evidence.

## What ``LATT`` means

The magnitude selects the centring; the **sign** carries the inversion centre:

* positive -- the space group is centrosymmetric, and the inversion centre is *not* listed among
  the ``SYMM`` records because SHELX adds it implicitly.
* negative -- no inversion centre is added.

So a positive ``LATT`` settles the question on its own. A negative ``LATT`` does not: the group may
still contain a mirror or a glide, which will appear as a ``SYMM`` record whose rotation part has
determinant -1.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "SymmetryOperator",
    "ImproperOperation",
    "ChiralityCompatibility",
    "SpaceGroupSymmetry",
    "parse_symm",
    "analyse_symmetry",
    "LATT_CENTRING",
]

#: ``abs(LATT)`` to the centring letter it selects. Only used for reporting.
LATT_CENTRING: dict[int, str] = {
    1: "P", 2: "I", 3: "R (rhombohedral obverse on hexagonal axes)", 4: "F",
    5: "A", 6: "B", 7: "C",
}

_TERM_RE = re.compile(
    r"""(?P<sign>[+-]?)\s*
        (?:
            (?P<num>\d+)\s*/\s*(?P<den>\d+)     # 1/2
          | (?P<dec>\d*\.\d+)                   # 0.5
          | (?P<int>\d+)(?!\s*[/xyzXYZ])        # bare integer translation
        )?
        \s*(?P<axis>[xyzXYZ])?
    """,
    re.VERBOSE,
)


class ImproperOperation(str, enum.Enum):
    """The kind of improper operation found, named the way a crystallographer would name it."""

    INVERSION_CENTRE = "inversion_centre"
    """From a positive ``LATT``. SHELX adds it implicitly; it is never a ``SYMM`` record."""

    MIRROR_OR_GLIDE = "mirror_or_glide"
    """Rotation part has determinant -1 and is an involution: a mirror, or a glide if it
    carries a translation."""

    ROTOINVERSION = "rotoinversion"
    """Determinant -1 but not an involution: a -3, -4 or -6 axis."""


class ChiralityCompatibility(str, enum.Enum):
    """Whether the declared group can hold a homochiral structure."""

    SOHNCKE = "sohncke"
    """Only proper rotations. A homochiral material can crystallise here."""

    CONTAINS_IMPROPER = "contains_improper"
    """At least one improper operation. **Impossible** for a homochiral material."""

    UNKNOWN = "unknown"
    """No ``LATT`` was declared, so the question cannot be decided from this file."""


@dataclass(frozen=True)
class SymmetryOperator:
    """One ``SYMM`` record, split into its rotation matrix and translation vector.

    ``rotation`` is a 3x3 integer matrix in row-major order. SHELX operators are always integral in
    the crystallographic basis, so no floating point is involved and the determinant is exact.
    """

    text: str
    rotation: tuple[tuple[int, int, int], ...]
    translation: tuple[Fraction, Fraction, Fraction]

    @property
    def determinant(self) -> int:
        """+1 for a proper rotation, -1 for an improper one. Exact; never rounded."""
        (a, b, c), (d, e, f), (g, h, i) = self.rotation
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    @property
    def is_proper(self) -> bool:
        return self.determinant == 1

    @property
    def is_identity(self) -> bool:
        return self.rotation == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) and not any(self.translation)

    @property
    def has_translation(self) -> bool:
        return any(t % 1 != 0 for t in self.translation)

    def improper_kind(self) -> ImproperOperation | None:
        """Classify an improper operation, or ``None`` when the operator is proper."""
        if self.is_proper:
            return None
        squared = _matmul(self.rotation, self.rotation)
        if squared == ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            return ImproperOperation.MIRROR_OR_GLIDE
        return ImproperOperation.ROTOINVERSION


@dataclass(frozen=True)
class SpaceGroupSymmetry:
    """What the instruction file's ``LATT``/``SYMM`` records amount to."""

    latt: int | None
    operators: tuple[SymmetryOperator, ...]
    compatibility: ChiralityCompatibility
    improper_operations: tuple[ImproperOperation, ...]
    unparsed: tuple[str, ...] = ()
    """``SYMM`` records that could not be parsed. Their presence forces ``UNKNOWN`` rather than a
    confident ``SOHNCKE``, because an unread operator might be the improper one."""

    @property
    def centring(self) -> str | None:
        if self.latt is None:
            return None
        return LATT_CENTRING.get(abs(self.latt))

    @property
    def is_centrosymmetric(self) -> bool | None:
        """``None`` when no ``LATT`` was declared -- absence is not evidence of absence."""
        if self.latt is None:
            return None
        return self.latt > 0

    @property
    def is_trivially_sohncke(self) -> bool:
        """True for ``P1``: Sohncke because it imposes no symmetry at all.

        This distinction matters when the verdict is used as reassurance. ``P1`` passes a chirality
        check the way an empty list passes "contains no negative numbers" -- correctly, and without
        telling you anything. A reduction left in ``P1`` is usually a decision to *postpone* the
        space group, not a determination of it, so a pass here is not evidence that the symmetry is
        right. It only means nothing has been asserted that is wrong.
        """
        return (
            self.compatibility is ChiralityCompatibility.SOHNCKE
            and not self.operators
            and self.latt is not None
            and abs(self.latt) == 1
        )

    def explain(self) -> str:
        """One sentence a crystallographer can check, naming the evidence."""
        if self.compatibility is ChiralityCompatibility.UNKNOWN:
            if self.latt is None:
                return "No LATT record, so the presence of an inversion centre is undeclared."
            return (
                f"{len(self.unparsed)} SYMM record(s) could not be parsed, so the group cannot be "
                "classified with confidence."
            )
        if self.is_trivially_sohncke:
            return (
                "LATT -1 with no SYMM records is P1, which imposes no symmetry at all. It is "
                "Sohncke trivially, so nothing here contradicts a homochiral structure -- but "
                "nothing here confirms the symmetry either. A reduction left in P1 has postponed "
                "the space group rather than determined it."
            )
        if self.compatibility is ChiralityCompatibility.SOHNCKE:
            return (
                f"LATT {self.latt} is negative (no inversion centre) and all {len(self.operators)} "
                "SYMM operator(s) are proper rotations, so this is a Sohncke group."
            )
        reasons: list[str] = []
        if self.is_centrosymmetric:
            reasons.append(f"LATT {self.latt} is positive, which adds an inversion centre")
        for op in self.operators:
            kind = op.improper_kind()
            if kind is not None:
                reasons.append(f"SYMM {op.text!r} has determinant -1 ({kind.value})")
        return "; ".join(reasons) + "."


def _matmul(
    a: tuple[tuple[int, int, int], ...], b: tuple[tuple[int, int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        tuple(sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)) for r in range(3)
    )


def parse_symm(text: str) -> SymmetryOperator | None:
    """Parse one ``SYMM`` record such as ``-x, y+1/2, -z``.

    Returns ``None`` rather than raising when the record cannot be read, so that an exotic operator
    downgrades the verdict to ``UNKNOWN`` instead of crashing a diagnosis. SHELX writes the three
    components comma-separated; some files use spaces, which is also accepted.
    """
    body = text.strip()
    if body.upper().startswith("SYMM"):
        body = body[4:].strip()
    parts = [p.strip() for p in body.split(",")]
    if len(parts) != 3:
        parts = [p for p in re.split(r"\s+", body) if p]
        if len(parts) != 3:
            return None

    rows: list[tuple[int, int, int]] = []
    shifts: list[Fraction] = []
    for part in parts:
        row, shift = _parse_component(part)
        if row is None:
            return None
        rows.append(row)
        shifts.append(shift)

    return SymmetryOperator(
        text=body,
        rotation=(rows[0], rows[1], rows[2]),
        translation=(shifts[0], shifts[1], shifts[2]),
    )


def _parse_component(part: str) -> tuple[tuple[int, int, int] | None, Fraction]:
    """Split one component (``-x+1/2``) into its axis coefficients and its translation."""
    coefficients = [0, 0, 0]
    shift = Fraction(0)
    cursor = 0
    text = part.replace(" ", "")
    if not text:
        return None, shift

    # Walk the string with an explicit cursor and refuse any gap. Scanning with ``finditer`` and
    # only checking the final end position lets unmatched characters slip through silently:
    # "3x" would match the "x" at position 1, end at 2 == len, and be accepted as "+x" with the
    # coefficient 3 quietly discarded. A wrong operator is worse than an unread one.
    while cursor < len(text):
        match = _TERM_RE.match(text, cursor)
        if match is None or match.end() == match.start():
            return None, shift
        cursor = match.end()
        sign = -1 if match.group("sign") == "-" else 1
        axis = match.group("axis")

        if match.group("num") is not None:
            magnitude = Fraction(int(match.group("num")), int(match.group("den")))
        elif match.group("dec") is not None:
            magnitude = Fraction(match.group("dec")).limit_denominator(24)
        elif match.group("int") is not None:
            magnitude = Fraction(int(match.group("int")))
        else:
            magnitude = Fraction(1)

        if axis is None:
            shift += sign * magnitude
        else:
            index = "xyz".index(axis.lower())
            # A coefficient other than +/-1 is not a SHELX operator; refuse rather than round.
            if magnitude != 1:
                return None, shift
            coefficients[index] += sign

    if all(c == 0 for c in coefficients) and shift == 0:
        return None, shift
    if any(abs(c) > 1 for c in coefficients):
        return None, shift
    return (coefficients[0], coefficients[1], coefficients[2]), shift


def analyse_symmetry(latt: int | None, symm: tuple[str, ...] | list[str]) -> SpaceGroupSymmetry:
    """Decide whether the declared group is Sohncke.

    The identity is implicit in SHELX and is not listed; a group with no ``SYMM`` records at all is
    ``P1`` (``LATT -1``) or ``P-1`` (``LATT 1``), and both are handled correctly by the ``LATT``
    sign alone.
    """
    operators: list[SymmetryOperator] = []
    unparsed: list[str] = []
    for record in symm:
        parsed = parse_symm(record)
        if parsed is None:
            unparsed.append(record)
        else:
            operators.append(parsed)

    improper: list[ImproperOperation] = []
    if latt is not None and latt > 0:
        improper.append(ImproperOperation.INVERSION_CENTRE)
    for op in operators:
        kind = op.improper_kind()
        if kind is not None:
            improper.append(kind)

    if improper:
        # An improper operation that WAS read is decisive even if others were not; you cannot
        # un-find a mirror.
        compatibility = ChiralityCompatibility.CONTAINS_IMPROPER
    elif latt is None or unparsed:
        compatibility = ChiralityCompatibility.UNKNOWN
    else:
        compatibility = ChiralityCompatibility.SOHNCKE

    return SpaceGroupSymmetry(
        latt=latt,
        operators=tuple(operators),
        compatibility=compatibility,
        improper_operations=tuple(dict.fromkeys(improper)),
        unparsed=tuple(unparsed),
    )
