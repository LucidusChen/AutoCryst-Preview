"""Audited, dependency-free preparation of *already parsed* initial solutions.

This module reads no files and imports no numerical or symmetry extension. Call
``crystalx_core.io.shelx_res.read_res`` at the application boundary, then supply a
complete set of affine symmetry operators explicitly. Absence never means P1.
The supplied set is checked against the declared SHELX generators, including
centring and the LATT inversion, and against the cell metric.

Atoms retain SFAC descriptors. Q peaks have no chemical identity: each needs an
explicit caller-supplied computational descriptor. Neither UNIT ratios nor atom
labels can rewrite descriptors. No original atom or peak is dropped or sorted.
Periodic images are complete within the requested Cartesian radius. That radius
is an explicit preprocessing policy, not a claim of upstream preprocessing parity;
a multilayer model can need a radius exceeding its single-edge cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
import math
import re

from crystalx_core.domain.chemistry import atomic_number
from crystalx_core.domain.inference import ElementInferenceRequest
from crystalx_core.domain.symmetry import SymmetryOperator, parse_symm
from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix
from crystalx_core.io.shelx_res import ResFile


class PreprocessingBlocked(ValueError):
    """The input cannot be prepared without missing or ambiguous information."""


@dataclass(frozen=True)
class CompleteSymmetry:
    """Explicit operators in fractional coordinates and their provenance.

    The finite group includes identity, inversion and all centring translations.
    Operators differing by an integer translation represent the same operation.
    This is a caller boundary, not an automatic provider registration/import.
    """

    operators: tuple[SymmetryOperator, ...]
    source: str


@dataclass(frozen=True)
class SiteProvenance:
    site_id: str
    source_site_id: str
    source_index: int
    kind: str
    label: str
    fractional: tuple[float, float, float]
    descriptor_atomic_number: int
    sfac_index: int
    occupancy_code: float
    u_values: tuple[float, ...]
    part: int | None
    residue: int | None
    afix: int | None
    peak_rank: int | None
    peak_height: float | None
    is_main: bool
    symmetry_operation_index: int
    lattice_translation: tuple[int, int, int]


@dataclass(frozen=True)
class PreparedInference:
    request: ElementInferenceRequest
    sites: tuple[SiteProvenance, ...]
    source_sha256: str
    cell_parameters: tuple[float, ...]
    symmetry: CompleteSymmetry
    context_radius_angstrom: float
    special_position_aliases: int
    warnings: tuple[str, ...]
    ordering_policy: str = "atoms in parsed order, then all Q peaks in parsed order; context follows"
    descriptor_policy: str = "atoms: SFAC as written; Q peaks: explicit caller computational descriptors"
    upstream_preprocessing_parity_verified: bool = False

    @property
    def main_mask(self) -> tuple[bool, ...]:
        return tuple(site.is_main for site in self.sites)


_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_INVERSION = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
_ZERO = (Fraction(0),) * 3
_CENTRING = {
    1: (),
    2: ((Fraction(1, 2),) * 3,),
    3: ((Fraction(2, 3), Fraction(1, 3), Fraction(1, 3)),
        (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3))),
    4: ((0, Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), 0, Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2), 0)),
    5: ((0, Fraction(1, 2), Fraction(1, 2)),),
    6: ((Fraction(1, 2), 0, Fraction(1, 2)),),
    7: ((Fraction(1, 2), Fraction(1, 2), 0),),
}


def _key(op: SymmetryOperator) -> tuple:
    return (op.rotation, tuple(Fraction(t) % 1 for t in op.translation))


def _validated_operator(op: SymmetryOperator) -> tuple:
    if not isinstance(op, SymmetryOperator):
        raise PreprocessingBlocked("Supply SymmetryOperator objects from a trusted explicit source")
    rotation = tuple(tuple(row) for row in op.rotation)
    if (len(rotation) != 3 or any(len(row) != 3 for row in rotation)
            or any(type(v) is not int or abs(v) > 1 for row in rotation for v in row)):
        raise PreprocessingBlocked("Symmetry rotations require 3x3 integer coefficients in [-1, 1]")
    if len(op.translation) != 3 or any(type(t) not in (int, Fraction) for t in op.translation):
        raise PreprocessingBlocked("Symmetry translations require three exact rational numbers")
    canonical = SymmetryOperator(op.text, rotation, tuple(Fraction(t) % 1 for t in op.translation))
    if abs(canonical.determinant) != 1:
        raise PreprocessingBlocked("A symmetry rotation must have determinant +1 or -1")
    return _key(canonical)


def _compose(left: tuple, right: tuple) -> tuple:
    a, u = left
    b, v = right
    return (
        tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
              for i in range(3)),
        tuple((sum(a[i][k] * v[k] for k in range(3)) + u[i]) % 1 for i in range(3)),
    )


def _check_symmetry(res: ResFile, supplied: CompleteSymmetry | None, matrix: tuple) -> CompleteSymmetry:
    if supplied is None:
        raise PreprocessingBlocked("Symmetry computation unavailable: explicit complete operators are required; no P1 fallback")
    if not isinstance(supplied, CompleteSymmetry) or not isinstance(supplied.source, str) or not supplied.source.strip():
        raise PreprocessingBlocked("Complete symmetry requires a stated provenance source")
    if not 1 <= len(supplied.operators) <= 192:
        raise PreprocessingBlocked("Supply between 1 and 192 complete symmetry operations")
    keys = tuple(_validated_operator(op) for op in supplied.operators)
    if len(set(keys)) != len(keys):
        raise PreprocessingBlocked("Duplicate symmetry operations modulo lattice translations")
    latt = res.header.latt
    if type(latt) is not int or abs(latt) not in _CENTRING:
        raise PreprocessingBlocked("An explicit valid SHELX LATT declaration is required")
    identity = (_IDENTITY, _ZERO)
    generators = [identity]
    if latt > 0:
        generators.append((_INVERSION, _ZERO))
    generators.extend((_IDENTITY, tuple(Fraction(t) for t in shift)) for shift in _CENTRING[abs(latt)])
    for text in res.header.symm:
        # The core chirality parser rounds decimal shifts to denominator 24. Do
        # not silently carry that reporting convention into numerical geometry.
        if re.search(r"\d*\.\d", text):
            raise PreprocessingBlocked("Decimal SYMM translations are unsupported here; exact fractional records are required")
        try:
            op = parse_symm(text)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise PreprocessingBlocked("Malformed declared SYMM operator") from exc
        if op is None:
            raise PreprocessingBlocked("Declared SYMM operator cannot be parsed safely")
        generators.append(_validated_operator(op))
    # Check equality to the generated group, not mere subset membership, so an
    # accidental P1 set or an unrelated supergroup cannot pass the boundary.
    expected = {identity}
    pending = [identity]
    while pending:
        current = pending.pop()
        for generator in generators:
            value = _compose(generator, current)
            if value not in expected:
                expected.add(value)
                pending.append(value)
                if len(expected) > 192:
                    raise PreprocessingBlocked("Declared symmetry does not close to a supported finite crystallographic group")
    if set(keys) != expected:
        raise PreprocessingBlocked("Supplied complete symmetry differs from declared LATT/SYMM group")
    metric = tuple(tuple(sum(matrix[k][i] * matrix[k][j] for k in range(3)) for j in range(3))
                   for i in range(3))
    if any(not math.isfinite(v) for row in metric for v in row):
        raise PreprocessingBlocked("CELL metric overflows finite geometry")
    for rotation, _ in keys:
        for i in range(3):
            for j in range(3):
                transformed = sum(rotation[k][i] * metric[k][l] * rotation[l][j]
                                  for k in range(3) for l in range(3))
                tolerance = math.sqrt(metric[i][i]) * math.sqrt(metric[j][j]) * 1e-6
                if not math.isfinite(transformed) or abs(transformed - metric[i][j]) > tolerance:
                    raise PreprocessingBlocked("Declared symmetry is incompatible with the supplied cell metric (relative tolerance 1e-6)")
    ordered = (identity,) + tuple(sorted(expected - {identity}))
    return CompleteSymmetry(tuple(SymmetryOperator("validated complete operator", r, t) for r, t in ordered), supplied.source)


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def prepare_initial_solution(
    res: ResFile,
    *,
    symmetry: CompleteSymmetry | None = None,
    context_radius_angstrom: float,
    q_peak_atomic_numbers: tuple[int, ...] | None = None,
    candidate_atomic_numbers: tuple[int, ...] | None = None,
    max_sites: int = 20_000,
    max_image_candidates: int = 1_000_000,
) -> PreparedInference:
    """Prepare a neutral request without changing scientific input.

    All atoms and Q peaks are main sites, including hydrogen; the caller must
    select a model appropriate for them. No UNIT-based trimming is supported.
    Parsed-file warnings that could conceal skipped rows are blocking. SHELX
    coordinate free-variable codes (absolute coordinate >= 10), disorder PARTs,
    coincident distinct sites and nonfinite fields are explicitly unsupported.
    Occupancy/U/peak height are retained provenance, not model features or
    calibrated confidence. ``max_sites``/``max_image_candidates`` block rather
    than truncate an incomplete environment.
    """
    if not isinstance(res, ResFile):
        raise PreprocessingBlocked("Expected an already safely parsed ResFile")
    if res.warnings:
        raise PreprocessingBlocked("Parsed-file warnings require review before inference: " + "; ".join(res.warnings))
    if (not re.fullmatch(r"[0-9a-fA-F]{64}", res.sha256)
            or res.header.sha256 != res.sha256 or res.header.source != res.source):
        raise PreprocessingBlocked("Parsed header and site provenance must describe the same SHA-256/source snapshot")
    if not _finite(context_radius_angstrom) or context_radius_angstrom <= 0:
        raise PreprocessingBlocked("context_radius_angstrom must be finite and positive")
    if any(type(v) is not int or v < 1 for v in (max_sites, max_image_candidates)):
        raise PreprocessingBlocked("Resource limits must be positive integers")
    cell = res.header.cell
    if cell is None:
        raise PreprocessingBlocked("CELL is required for Cartesian geometry")
    parameters = (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
    if (not all(_finite(v) for v in parameters) or any(v <= 0 for v in parameters[:3])
            or any(not 0 < v < 180 for v in parameters[3:])):
        raise PreprocessingBlocked("CELL must contain finite positive lengths and angles strictly between 0 and 180")
    try:
        matrix = orthogonalisation_matrix(parameters)
        reciprocal_lengths = cell.reciprocal()[:3]
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise PreprocessingBlocked("CELL has a degenerate metric") from exc
    if any(not math.isfinite(v) for row in matrix for v in row) or any(not math.isfinite(v) for v in reciprocal_lengths):
        raise PreprocessingBlocked("CELL metric cannot be represented finitely")
    complete = _check_symmetry(res, symmetry, matrix)
    peak_z = tuple(q_peak_atomic_numbers) if q_peak_atomic_numbers is not None else ()
    if len(peak_z) != len(res.q_peaks) or any(type(z) is not int or not 1 <= z <= 118 for z in peak_z):
        raise PreprocessingBlocked("Every Q peak requires one explicit computational atomic-number descriptor, in parsed peak order")
    main: list[SiteProvenance] = []
    for kind, rows in (("atom", res.atoms), ("peak", res.q_peaks)):
        for index, site in enumerate(rows):
            frac = (site.x, site.y, site.z)
            if (not isinstance(site.label, str) or not site.label.strip()
                    or not all(_finite(v) for v in frac) or not _finite(site.occupancy_code)):
                raise PreprocessingBlocked("All site labels, fractional coordinates and occupancy codes must be valid and finite")
            if any(abs(v) >= 10 for v in frac):
                raise PreprocessingBlocked("Encoded/free-variable or extreme coordinates require explicit resolution before inference")
            if kind == "atom":
                if site.part not in (None, 0):
                    raise PreprocessingBlocked("Disorder PART sites require a separate explicit model; no conformer is selected silently")
                if any(not _finite(v) for v in site.u_values):
                    raise PreprocessingBlocked("Nonfinite displacement parameters require review")
                if type(site.sfac_index) is not int or not 1 <= site.sfac_index <= len(res.header.sfac):
                    raise PreprocessingBlocked("Atom SFAC index is invalid; label-prefix fallback is forbidden")
                try:
                    z = atomic_number(res.header.sfac[site.sfac_index - 1])
                except (KeyError, AttributeError) as exc:
                    raise PreprocessingBlocked("Unknown SFAC element; no element substitution is allowed") from exc
                part, residue, rank, height = site.part, site.residue, None, None
                u_values, afix = site.u_values, site.afix
            else:
                z = peak_z[index]
                if not _finite(site.height) or not _finite(site.u_iso):
                    raise PreprocessingBlocked("Q peak height and displacement parameter must be finite")
                part, residue, rank, height = None, None, site.rank, site.height
                u_values, afix = (site.u_iso,), None
            site_id = f"{kind}/{index + 1:06d}/{site.label}"
            main.append(SiteProvenance(site_id, site_id, len(main), kind, site.label, frac, z,
                                       site.sfac_index, site.occupancy_code, u_values,
                                       part, residue, afix, rank, height, True, 0, (0, 0, 0)))
    if not main:
        raise PreprocessingBlocked("An initial solution must contain at least one atom or Q peak")
    if len(main) > max_sites:
        raise PreprocessingBlocked("Main-site count exceeds max_sites; no peaks were removed")
    positions = [frac_to_cart(matrix, site.fractional) for site in main]
    if any(not math.isfinite(v) for pos in positions for v in pos):
        raise PreprocessingBlocked("Cartesian site coordinates overflow")
    sites = list(main)
    # Coincidence is a numerical identity check, not a bond/chemical-distance
    # threshold. Original distinct sites are never deduplicated.
    epsilon = 1e-7
    for i, point in enumerate(positions):
        if any(math.dist(point, other) <= epsilon for other in positions[:i]):
            raise PreprocessingBlocked("Distinct main sites coincide; resolve duplicates/disorder without dropping peaks")
    aliases = 0
    candidates = 0
    lower = tuple(min(s.fractional[i] for s in main) for i in range(3))
    upper = tuple(max(s.fractional[i] for s in main) for i in range(3))
    radius = float(context_radius_angstrom)
    for source in main:
        for op_index, op in enumerate(complete.operators):
            transformed = tuple(sum(op.rotation[i][k] * source.fractional[k] for k in range(3))
                                + float(op.translation[i]) for i in range(3))
            spreads = tuple(radius * length for length in reciprocal_lengths)
            raw_bounds = tuple((lower[i] - transformed[i] - spreads[i] - epsilon,
                                upper[i] - transformed[i] + spreads[i] + epsilon) for i in range(3))
            if any(not math.isfinite(v) for pair in raw_bounds for v in pair):
                raise PreprocessingBlocked("Periodic search bounds overflow; reduce the explicit radius or review CELL")
            bounds = tuple((math.ceil(lo), math.floor(hi)) for lo, hi in raw_bounds)
            count = math.prod(max(0, hi - lo + 1) for lo, hi in bounds)
            candidates += count
            if candidates > max_image_candidates:
                raise PreprocessingBlocked("Periodic search exceeds max_image_candidates; context was not truncated")
            for shift in product(*(range(lo, hi + 1) for lo, hi in bounds)):
                frac = tuple(transformed[i] + shift[i] for i in range(3))
                point = frac_to_cart(matrix, frac)
                if min(math.dist(point, p) for p in positions[:len(main)]) > radius + epsilon:
                    continue
                overlaps = [i for i, p in enumerate(positions) if math.dist(point, p) <= epsilon]
                if any(sites[i].source_site_id != source.site_id for i in overlaps):
                    raise PreprocessingBlocked("Different source sites coincide through symmetry; no site was removed")
                if overlaps:
                    if op_index != 0 or shift != (0, 0, 0):
                        aliases += 1
                    continue
                site_id = f"{source.site_id}/op{op_index:03d}/t{shift[0]},{shift[1]},{shift[2]}"
                sites.append(replace(source, site_id=site_id, fractional=frac, is_main=False,
                                     symmetry_operation_index=op_index, lattice_translation=shift))
                positions.append(point)
                if len(sites) > max_sites:
                    raise PreprocessingBlocked("Periodic neighborhood exceeds max_sites; context was not truncated")
    request = ElementInferenceRequest(tuple(s.site_id for s in sites),
                                      tuple(s.descriptor_atomic_number for s in sites),
                                      tuple(positions), candidate_atomic_numbers, len(main))
    notes = tuple(res.header.warnings) + (
        "Experimental descriptors and symmetry context; no scientific validation or confidence calibration.",
        "Occupancy, displacement parameters and Q heights are provenance only, not model features.",
        "Requested context radius is explicit; upstream and model receptive-field coverage have not been validated.",
    )
    if peak_z:
        notes += ("Q descriptors are caller hypotheses, not assigned chemical elements.",)
    return PreparedInference(request, tuple(sites), res.sha256, parameters, complete, radius, aliases, notes)
