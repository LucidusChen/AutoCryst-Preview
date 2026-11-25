"""Token-anchored readers for SHELX instruction and reflection files.

Deliberately *not* substring matching. The imported engine decides where the atom list starts with
``if "PLAN" in line``, which misfires on a title such as ``TITL ligand PLANar isomer``. Here an
instruction is recognised only when it is the **first whitespace-delimited token** of a line.

Scope: enough to diagnose the data at Stage 0 and to decide the solver portfolio. Writing candidate
instruction files is a later slice.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from crystalx_core.domain.data import (
    AnomalousAvailability,
    HklfFormat,
    InsHeader,
    ReflectionDataSummary,
    ResolutionShellStats,
    SpaceGroupConfidence,
    Unavailable,
    UnitCell,
)
from crystalx_core.errors import ParseError
from crystalx_core.workspace.hashing import sha256_file

#: A SHELX atom label: a letter followed by letters, digits or the label punctuation SHELX allows.
_LABEL_LIKE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_'$]*$")

__all__ = [
    "read_ins_header",
    "scan_hkl",
    "logical_lines",
    "COMPLETENESS_UNAVAILABLE",
]

#: Completeness needs the space-group symmetry to know how many unique reflections were possible.
COMPLETENESS_UNAVAILABLE = Unavailable(
    reason=(
        "Completeness requires space-group symmetry to compute the number of unique reflections "
        "that were possible. crystalx_core has no symmetry engine; register a cctbx-backed provider "
        "to enable this."
    ),
    blocking_capability="space_group_symmetry",
)

_MAX_CONTINUATIONS = 20


@dataclass(frozen=True)
class LogicalLine:
    """One SHELX instruction after continuation lines have been joined."""

    text: str
    first_line_no: int
    raw_lines: tuple[str, ...]

    @property
    def tokens(self) -> list[str]:
        return self.text.split()

    @property
    def instruction(self) -> str:
        toks = self.tokens
        return toks[0].upper() if toks else ""


def logical_lines(path: Path | str) -> Iterator[LogicalLine]:
    """Yield logical SHELX lines, joining ``=``-continued physical lines.

    SHELX continues an instruction when the line ends with ``=``. Ignoring that silently truncates
    long ``SYMM``/``SFAC``/``UNIT`` records, which is exactly the sort of quiet data loss this
    project must not have.
    """
    target = Path(path)
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        buffer: list[str] = []
        raw: list[str] = []
        start_line = 1
        for line_no, physical in enumerate(handle, start=1):
            stripped = physical.rstrip("\n").rstrip("\r")
            if not buffer:
                start_line = line_no
            raw.append(stripped)
            # Strip each fragment before joining so a continued instruction reads as one clean
            # line in error messages and in the record.
            scientific = stripped.split("!", 1)[0]
            content = scientific.strip() if buffer else scientific.rstrip()
            if content.endswith("="):
                buffer.append(content[:-1].rstrip())
                if len(buffer) > _MAX_CONTINUATIONS:
                    raise ParseError(
                        "Too many '=' continuation lines; the file is probably malformed",
                        source=target,
                        line_no=start_line,
                        text=" ".join(buffer)[:200],
                    )
                continue
            buffer.append(content)
            yield LogicalLine(" ".join(buffer).strip(), start_line, tuple(raw))
            buffer = []
            raw = []
        if buffer:
            yield LogicalLine(" ".join(buffer).strip(), start_line, tuple(raw))


def _floats(tokens: Sequence[str], *, source: Path, line_no: int, what: str) -> list[float]:
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ParseError(
                f"{what}: expected a number, got {token!r}",
                source=source,
                line_no=line_no,
                text=" ".join(tokens),
            ) from exc
    return values


def read_ins_header(path: Path | str) -> InsHeader:
    """Read the header of a ``.ins``/``.res`` file.

    Recognised instructions: ``TITL``, ``CELL``, ``ZERR``, ``LATT``, ``SYMM``, ``SFAC``, ``UNIT``,
    ``HKLF``. Reading stops at ``HKLF`` or ``END``, which is where the header ends in every SHELX
    file.
    """
    target = Path(path)
    if not target.is_file():
        raise ParseError(f"Instruction file not found: {target}", source=target)

    title = ""
    wavelength: float | None = None
    cell: UnitCell | None = None
    z_value: float | None = None
    latt: int | None = None
    symm: list[str] = []
    sfac: list[str] = []
    unit: list[float] = []
    hklf: HklfFormat | None = None
    warnings: list[str] = []
    rem_lines: list[str] = []
    has_atom_records = False

    for line in logical_lines(target):
        instruction = line.instruction
        tokens = line.tokens
        if not instruction:
            continue
        if instruction == "REM":
            # Kept, not skipped: the vendor pipelines announce themselves here, and that is how
            # a machine-written placeholder composition is recognised.
            rem_lines.append(line.text[3:].strip())
            continue

        if instruction == "TITL":
            title = line.text[4:].strip()

        elif instruction == "CELL":
            values = _floats(tokens[1:], source=target, line_no=line.first_line_no, what="CELL")
            if len(values) < 7:
                raise ParseError(
                    f"CELL needs wavelength plus six cell parameters, got {len(values)} values",
                    source=target,
                    line_no=line.first_line_no,
                    text=line.text,
                )
            wavelength = values[0]
            try:
                cell = UnitCell(*values[1:7])
            except ValueError as exc:
                raise ParseError(
                    f"CELL is not a valid unit cell: {exc}",
                    source=target,
                    line_no=line.first_line_no,
                    text=line.text,
                ) from exc
            if wavelength <= 0:
                warnings.append(f"CELL wavelength is not positive: {wavelength}")

        elif instruction == "ZERR":
            values = _floats(tokens[1:], source=target, line_no=line.first_line_no, what="ZERR")
            if values:
                z_value = values[0]

        elif instruction == "LATT":
            values = _floats(tokens[1:], source=target, line_no=line.first_line_no, what="LATT")
            if values:
                latt = int(values[0])

        elif instruction == "SYMM":
            operator = line.text[4:].strip()
            if operator:
                symm.append(operator)

        elif instruction == "SFAC":
            symbols, long_form = _parse_sfac(tokens[1:])
            if long_form and sfac:
                sfac.extend(symbols)
            elif long_form:
                sfac.extend(symbols)
            else:
                sfac.extend(symbols)

        elif instruction == "UNIT":
            unit.extend(
                _floats(tokens[1:], source=target, line_no=line.first_line_no, what="UNIT")
            )

        elif instruction == "HKLF":
            values = _floats(tokens[1:], source=target, line_no=line.first_line_no, what="HKLF")
            if values:
                code = int(values[0])
                try:
                    hklf = HklfFormat(code)
                except ValueError:
                    warnings.append(f"Unrecognised HKLF code {code}")
            break

        elif instruction == "END":
            break

        elif not has_atom_records and _looks_like_an_atom(instruction, tokens):
            # Atoms appear between UNIT and HKLF once a structure exists. Their presence is what
            # separates "a human has worked on this file" from "the instrument wrote it and nobody
            # has solved anything yet"; see domain.composition.PRE_SOLUTION_VENDOR_FILE.
            has_atom_records = True

    if sfac and unit and len(sfac) != len(unit):
        warnings.append(
            f"SFAC lists {len(sfac)} elements but UNIT lists {len(unit)} counts; "
            "the composition is inconsistent"
        )

    confidence = (
        SpaceGroupConfidence.FROM_INS
        if latt is not None or symm
        else SpaceGroupConfidence.UNKNOWN
    )

    return InsHeader(
        source=target,
        sha256=sha256_file(target),
        title=title,
        wavelength_a=wavelength,
        cell=cell,
        z=z_value,
        latt=latt,
        symm=tuple(symm),
        sfac=tuple(sfac),
        unit=tuple(unit),
        hklf_format=hklf,
        space_group_confidence=confidence,
        rem_lines=tuple(rem_lines),
        has_atom_records=has_atom_records,
        warnings=tuple(warnings),
    )


def _parse_sfac(tokens: Sequence[str]) -> tuple[list[str], bool]:
    """Return ``(symbols, is_long_form)``.

    ``SFAC`` has two shapes. The short form is a list of element symbols. The long form declares a
    single element followed by its scattering coefficients, which is detectable because the second
    token is numeric.
    """
    if len(tokens) >= 3 and _looks_numeric(tokens[1]):
        return [tokens[0].capitalize()], True
    return [t.capitalize() for t in tokens if not _looks_numeric(t)], False


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_an_atom(instruction: str, tokens: Sequence[str]) -> bool:
    """Is this logical line an atom record rather than an instruction?

    A SHELX atom is ``label sfac_index x y z [occupancy] [U...]`` -- a label that is not a known
    keyword, followed by an integer SFAC index and at least three numbers. Q peaks count as atoms
    for this purpose: they still mean the file has been through a solution program.
    """
    from crystalx_core.io.shelx_res import SHELX_INSTRUCTIONS

    if not tokens or instruction in SHELX_INSTRUCTIONS:
        return False
    if not _LABEL_LIKE_RE.match(tokens[0]):
        return False
    if len(tokens) < 5:
        return False
    try:
        int(tokens[1])
        float(tokens[2]), float(tokens[3]), float(tokens[4])
    except ValueError:
        return False
    return True


def scan_hkl(
    path: Path | str,
    *,
    cell: UnitCell | None = None,
    hklf_hint: HklfFormat | None = None,
    outer_shell_fraction: float = 0.1,
    max_reflections: int | None = None,
) -> ReflectionDataSummary:
    """Scan a SHELX reflection file for the facts the strategy selector needs.

    The important one is **whether unmerged Friedel pairs are present**: both ``h k l`` and
    ``-h -k -l`` occurring as separate records is a hard prerequisite for the SHELXC-D-E route. It is
    a purely structural property of the user's own file, so it can be decided with certainty and
    without any physical constants.

    Resolution limits are computed only when a unit cell is supplied. Completeness is *not* computed;
    see :data:`COMPLETENESS_UNAVAILABLE`.
    """
    target = Path(path)
    if not target.is_file():
        raise ParseError(f"Reflection file not found: {target}", source=target)

    reflections: set[tuple[int, int, int]] = set()
    n_reflections = 0
    n_zero_sigma = 0
    h_lo = k_lo = l_lo = 0
    h_hi = k_hi = l_hi = 0
    first = True
    hit_terminator = False
    warnings: list[str] = []
    per_reflection: list[tuple[float, float | None]] = []
    has_extra_column = False
    saw_negative_extra = False
    n_fields_max = 0

    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parsed = _parse_hkl_record(line)
            if parsed is None:
                if n_reflections == 0 and line_no <= 5:
                    continue  # tolerate a leading title line
                warnings.append(f"line {line_no}: unparsable reflection record, skipped")
                continue
            h, k, l = parsed.h, parsed.k, parsed.l
            intensity, sigma, extra = parsed.intensity, parsed.sigma, parsed.extra
            if h == 0 and k == 0 and l == 0:
                hit_terminator = True
                break
            n_fields_max = max(n_fields_max, parsed.n_fields)
            if extra is not None:
                has_extra_column = True
                if extra < 0:
                    saw_negative_extra = True

            n_reflections += 1
            reflections.add((h, k, l))
            if sigma is not None and sigma == 0.0:
                n_zero_sigma += 1
            if first:
                h_lo = h_hi = h
                k_lo = k_hi = k
                l_lo = l_hi = l
                first = False
            else:
                h_lo, h_hi = min(h_lo, h), max(h_hi, h)
                k_lo, k_hi = min(k_lo, k), max(k_hi, k)
                l_lo, l_hi = min(l_lo, l), max(l_hi, l)

            if cell is not None:
                try:
                    d = cell.d_spacing(h, k, l)
                except ValueError:
                    continue
                ratio = (intensity / sigma) if (sigma not in (None, 0.0)) else None
                per_reflection.append((d, ratio))

            if max_reflections is not None and n_reflections >= max_reflections:
                warnings.append(f"scan stopped after {max_reflections} reflections (caller limit)")
                break

    if n_reflections == 0:
        raise ParseError(
            "No reflections could be read from this file",
            source=target,
            context={"hint": "expected SHELX HKLF format: h k l F^2 sigma"},
        )

    n_pairs = _count_friedel_pairs(reflections)
    anomalous, friedel_coverage, sparse_warning = _classify_anomalous(
        n_pairs=n_pairs, n_reflections=n_reflections
    )
    if sparse_warning:
        warnings.append(sparse_warning)

    hklf = hklf_hint
    hklf_was_declared = hklf_hint is not None
    if hklf is None:
        hklf, hklf_warning = _infer_hklf(
            n_fields_max=n_fields_max,
            saw_negative_extra=saw_negative_extra,
            saw_extra=has_extra_column,
        )
        if hklf_warning:
            warnings.append(hklf_warning)

    d_min = d_max = None
    mean_ratio = None
    outer: ResolutionShellStats | None = None
    if per_reflection:
        ds = [d for d, _ in per_reflection]
        d_min, d_max = min(ds), max(ds)
        ratios = [r for _, r in per_reflection if r is not None]
        if ratios:
            mean_ratio = sum(ratios) / len(ratios)
        outer = _outer_shell(per_reflection, fraction=outer_shell_fraction)

    return ReflectionDataSummary(
        source=target,
        sha256=sha256_file(target),
        n_reflections=n_reflections,
        hklf_format=hklf,
        h_range=(h_lo, h_hi),
        k_range=(k_lo, k_hi),
        l_range=(l_lo, l_hi),
        anomalous=anomalous,
        n_friedel_pairs=n_pairs,
        friedel_coverage=friedel_coverage,
        hklf_was_declared=hklf_was_declared,
        d_min_a=d_min,
        d_max_a=d_max,
        mean_i_over_sigma=mean_ratio,
        outer_shell=outer,
        completeness=None,
        completeness_unavailable=COMPLETENESS_UNAVAILABLE,
        n_zero_sigma=n_zero_sigma,
        truncated_at_terminator=hit_terminator,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class _HklRecord:
    """One reflection record, plus enough shape information to identify the file's dialect."""

    h: int
    k: int
    l: int
    intensity: float
    sigma: float | None
    extra: float | None
    """Sixth column, if present: a twin component in HKLF 5, a batch number otherwise."""
    n_fields: int
    """Total numeric fields on the record. Distinguishes HKLF 5 from the extended formats."""


def _parse_hkl_record(line: str) -> _HklRecord | None:
    """Parse one reflection record.

    Tries the SHELX fixed-format layout ``3I4,2F8.2`` first, because that is what the programs
    themselves write and it survives fields that have run together. Falls back to whitespace
    splitting for the free-format files many pipelines emit.
    """
    if len(line) >= 28:
        try:
            h = int(line[0:4])
            k = int(line[4:8])
            l = int(line[8:12])
            intensity = float(line[12:20])
            sigma = float(line[20:28])
            extra_text = line[28:36].strip()
            extra = float(extra_text) if extra_text else None
            tail = line[36:].strip()
            n_fields = 5 + (1 if extra is not None else 0) + len(tail.split())
            return _HklRecord(h, k, l, intensity, sigma, extra, n_fields)
        except ValueError:
            pass

    tokens = line.split()
    if len(tokens) < 5:
        return None
    try:
        h, k, l = int(tokens[0]), int(tokens[1]), int(tokens[2])
        intensity = float(tokens[3])
        sigma = float(tokens[4])
    except ValueError:
        return None
    extra = None
    if len(tokens) >= 6:
        try:
            extra = float(tokens[5])
        except ValueError:
            extra = None
    return _HklRecord(h, k, l, intensity, sigma, extra, len(tokens))


# --------------------------------------------------------------------------------------
# HKLF dialect inference
# --------------------------------------------------------------------------------------

#: Records with more fields than this are not plain HKLF; the surplus is direction cosines.
#:
#: PLATON's absorption and psi-scan inputs are written as ``h k l F^2 sigma batch`` followed by
#: six direction cosines -- twelve fields in total. Counting only "is there a sixth column?"
#: mistakes every one of those files for an HKLF 5 twin, which would put a perfectly ordinary
#: dataset down the twinning route. Verified against ``PSICOR_EXAMPLE/s588a.hkl`` in a real
#: PLATON distribution.
MAX_PLAIN_HKLF_FIELDS = 6


def _infer_hklf(
    *, n_fields_max: int, saw_negative_extra: bool, saw_extra: bool
) -> tuple[HklfFormat, str | None]:
    """Guess the HKLF dialect from the shape of the records, and say so out loud.

    Only ever called when the instruction file did not declare ``HKLF``. Every branch that
    guesses returns a warning naming the guess, because a wrong dialect changes what the data
    mean, and a silent guess is the dangerous kind.
    """
    if not saw_extra:
        return HklfFormat.HKLF4, None

    if n_fields_max > MAX_PLAIN_HKLF_FIELDS:
        return HklfFormat.HKLF4, (
            f"Records carry {n_fields_max} numeric fields. That is the extended layout used for "
            "absorption and psi-scan corrections -- h k l F^2 sigma batch followed by direction "
            "cosines -- not HKLF 5. Reading as HKLF 4 and ignoring the trailing columns. "
            "Declare HKLF explicitly in the instruction file to remove the guess."
        )

    if saw_negative_extra:
        return HklfFormat.HKLF5, (
            "Records carry a sixth column and some values are negative, which is the HKLF 5 "
            "convention for the non-final members of an overlap group. Assuming HKLF 5 "
            "(twinned). Declare HKLF explicitly in the instruction file to remove the guess."
        )

    return HklfFormat.HKLF4, (
        "Records carry a sixth column but every value is non-negative, so it reads as a batch "
        "number rather than an HKLF 5 twin component. Assuming HKLF 4. Declare HKLF explicitly "
        "in the instruction file to remove the guess."
    )


# --------------------------------------------------------------------------------------
# Friedel-pair coverage
# --------------------------------------------------------------------------------------

#: A file must have at least this many Friedel pairs before the anomalous route is unlocked.
MIN_FRIEDEL_PAIRS = 20

#: ...and the mates must cover at least this fraction of the records.
#:
#: A merged file often still contains a handful of ``(h,k,l)``/``(-h,-k,-l)`` coincidences --
#: re-measured standards, a duplicated batch, centric reflections that are their own mates. Two
#: such pairs in 3566 records is not anomalous data, but the old test (``n_pairs > 0``) called it
#: ``FRIEDEL_PAIRS_PRESENT``, which is the hard prerequisite gating SHELXC/D/E. A genuinely
#: unmerged dataset has mates for a large fraction of its records, so a 5 % floor rejects the
#: noise without coming anywhere near a real measurement. It is deliberately far below what
#: usable anomalous data looks like: this test exists to catch accidents, not to judge quality.
MIN_FRIEDEL_COVERAGE = 0.05


def _count_friedel_pairs(reflections: set[tuple[int, int, int]]) -> int:
    """Number of ``(h,k,l)`` / ``(-h,-k,-l)`` pairs present as separate records.

    ``(0,0,0)`` is excluded, and each pair is counted once.
    """
    counted = 0
    for h, k, l in reflections:
        if (h, k, l) == (0, 0, 0):
            continue
        mate = (-h, -k, -l)
        if mate in reflections and (h, k, l) < mate:
            counted += 1
    return counted


def _classify_anomalous(
    *,
    n_pairs: int,
    n_reflections: int,
    min_pairs: int = MIN_FRIEDEL_PAIRS,
    min_coverage: float = MIN_FRIEDEL_COVERAGE,
) -> tuple[AnomalousAvailability, float, str | None]:
    """Decide whether this file can support experimental phasing, and return the coverage.

    Returns the availability, the fraction of records that participate in a pair, and a warning
    when mates exist but are too sparse to be believed.
    """
    if n_reflections <= 0 or n_pairs == 0:
        return AnomalousAvailability.MERGED_NO_PAIRS, 0.0, None

    coverage = min(1.0, (2 * n_pairs) / n_reflections)
    if n_pairs >= min_pairs and coverage >= min_coverage:
        return AnomalousAvailability.FRIEDEL_PAIRS_PRESENT, coverage, None

    # Say which condition failed. "Looks like a merged file with duplicates" is the right reading
    # of 2 pairs in 3566 records and the wrong reading of 5 pairs in 13, and a message that cannot
    # tell those apart sends the crystallographer looking for the wrong problem.
    if coverage < min_coverage:
        diagnosis = (
            f"Only {coverage:.2%} of the records have a mate, below the {min_coverage:.0%} floor. "
            "That is the signature of re-measured or duplicated reflections in an otherwise "
            "merged file, not of unmerged anomalous data."
        )
    else:
        diagnosis = (
            f"The mates cover {coverage:.2%} of the records, but {n_pairs} pair(s) is below the "
            f"{min_pairs}-pair floor. The file is too small for the anomalous differences to mean "
            "anything, whatever fraction of it is paired."
        )

    return (
        AnomalousAvailability.FRIEDEL_PAIRS_SPARSE,
        coverage,
        (
            f"{n_pairs} Friedel pair(s) found in {n_reflections} records. {diagnosis} The "
            "experimental-phasing route stays closed; supply the full unmerged dataset if you "
            "intend to phase on the anomalous signal."
        ),
    )


def _outer_shell(
    per_reflection: list[tuple[float, float | None]], *, fraction: float
) -> ResolutionShellStats | None:
    """Statistics for the highest-resolution ``fraction`` of the data."""
    if not per_reflection or not 0.0 < fraction <= 1.0:
        return None
    ordered = sorted(per_reflection, key=lambda item: item[0])  # smallest d = highest resolution
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    shell = ordered[:count]
    ratios = [r for _, r in shell if r is not None]
    return ResolutionShellStats(
        d_min_a=shell[0][0],
        d_max_a=shell[-1][0],
        n_reflections=len(shell),
        mean_i_over_sigma=(sum(ratios) / len(ratios)) if ratios else None,
    )
