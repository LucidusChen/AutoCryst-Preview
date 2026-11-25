"""Independent post-seal RES comparison, with a bounded supported geometry scope.

Only this grading entry reads the expert, after the candidate seal is verified.
Nothing returns an expert structure to a solver. Same-cell settings are supported
with declared symmetry, origin candidates, periodic distances and one-to-one site
assignment. General change-of-basis/disorder matching and framework topology are
explicitly not evaluated. Measurements are not a scientific PASS.

Optional APIs: https://gemmi.readthedocs.io/en/latest/symmetry.html and
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
Gemmi's fractional-nearest helper is deliberately not used as a Cartesian metric
minimum in skew cells; translation enumeration below is metric-bounded instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import product
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import time

from crystalx_core._serde import to_jsonable
from crystalx_core.app.checkcif import _fresh_private_directory
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.symmetry import parse_symm
from crystalx_core.io.shelx_res import ResFile, read_res
from crystalx_core.io.shelx_reader import logical_lines
from crystalx_core.private_acceptance.sealing import SealError, read_private_reference, read_sealed_output, verify_seal


@dataclass(frozen=True)
class ComparisonPolicy:
    distance_tolerance_angstrom: float
    metal_elements: tuple[str, ...]
    framework_candidate_site_ids: tuple[str, ...] = ()
    framework_expert_site_ids: tuple[str, ...] = ()
    cell_length_tolerance_angstrom: float = 0.002
    cell_angle_tolerance_deg: float = 0.02
    symmetry_fractional_tolerance: float = 1e-6
    threshold_approval_reference: str = ""
    candidate_reflection_sha256: str = ""
    expert_reflection_sha256: str = ""
    max_sites: int = 1200
    max_origin_candidates: int = 128
    max_pair_evaluations: int = 2_000_000
    max_periodic_images: int = 4096
    max_wall_seconds: float = 60.0
    coordinate_identity_mode: str = "strict_flat"
    origin_selection: str = "all_sites"

    def __post_init__(self):
        if self.coordinate_identity_mode not in ("strict_flat","scoped_residues"):
            raise ValueError("Unsupported coordinate identity mode")
        if self.origin_selection not in ("all_sites","metal_anchors"):
            raise ValueError("Unsupported origin selection mode")
        for value in (self.distance_tolerance_angstrom, self.cell_length_tolerance_angstrom,
                      self.cell_angle_tolerance_deg, self.symmetry_fractional_tolerance, self.max_wall_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Comparison tolerances and time bounds must be explicitly finite and positive")
        for values in (self.metal_elements, self.framework_candidate_site_ids, self.framework_expert_site_ids):
            if not isinstance(values, tuple) or len(set(values)) != len(values) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("Comparison role declarations must be distinct explicit tuples")
        if any(element not in ATOMIC_NUMBERS for element in self.metal_elements):
            raise ValueError("Metal categories require declared canonical element symbols")
        for value in (self.max_sites, self.max_origin_candidates, self.max_pair_evaluations, self.max_periodic_images):
            if type(value) is not int or value < 1:
                raise ValueError("Comparison resource bounds must be positive integers")


@dataclass(frozen=True)
class SiteComparison:
    candidate_site_id: str
    expert_site_id: str
    candidate_element: str
    expert_element: str
    distance_angstrom: float
    expert_symmetry_operation: str
    lattice_translation: tuple[int, int, int]


@dataclass(frozen=True)
class CategoryComparison:
    category: str
    candidate_count: int | None
    expert_count: int | None
    recovered_expert_sites: int | None
    missing_expert_sites: int | None
    extra_candidate_sites: int | None
    same_element_matches: int | None
    element_mismatches: int | None
    rmsd_angstrom: float | None
    role_status: str


@dataclass(frozen=True)
class ModelComparisonReport:
    status: str
    reasons: tuple[str, ...]
    candidate_sha256: str | None = None
    expert_sha256: str | None = None
    candidate_seal_sha256: str | None = None
    categories: tuple[CategoryComparison, ...] = ()
    matches: tuple[SiteComparison, ...] = ()
    origin_shift_fractional: tuple[float, float, float] | None = None
    inventory: dict | None = None
    settings: dict | None = None
    r_comparison: dict | None = None
    origin_search_exhaustive: bool = False
    origins_evaluated: int = 0
    output_directory: Path | None = None
    expert_used_for_solver: bool = False
    scientifically_reviewed: bool = False
    framework_topology_status: str = "NOT_EVALUATED"

    def to_record(self) -> dict:
        return to_jsonable(asdict(self))


class _Unsupported(ValueError):
    pass


def _coordinate_identity_view(res):
    """Namespace already resolved sites; never generate atoms or evaluate restraints.

    This optional, post-seal-only view accepts explicit RESI numbers/chain:number
    identities and known class-qualified restraint commands. Raw model bytes and
    coordinates remain untouched. PART/AFIX/coupled sites retain their usual blocks.
    """
    restraints={"DFIX","DANG","SADI","SAME","FLAT","CHIV","DELU","SIMU","ISOR","RIGU"}
    namespace=None;namespaces={};known_warning_lines=set();commands=set()
    for line in logical_lines(res.source):
        fields=line.tokens
        if not fields:continue
        if line.instruction=="RESI":
            args=fields[1:]
            numbers=[value for value in args if re.fullmatch(r"(?:[A-Za-z][A-Za-z0-9]*:)?[0-9]+",value)]
            if (len(numbers)!=1 or len(args)>2 or any(
                    value not in numbers and not re.fullmatch(r"[A-Za-z0-9]{1,8}",value) for value in args)):
                raise _Unsupported("explicit_residue_or_chain_number_required_for_coordinate_identity")
            namespace=None if numbers[0]=="0" else numbers[0].upper()
        instruction=fields[0].upper()
        base,separator,suffix=instruction.partition("_")
        if separator and base in restraints and re.fullmatch(r"[A-Z0-9]{1,8}",suffix):
            known_warning_lines.add(line.first_line_no);commands.add(instruction)
        namespaces[line.first_line_no]=namespace
    warnings=[]
    for warning in res.warnings:
        match=re.match(r"line ([0-9]+): looks like an atom but could not be parsed:",warning)
        if not match or int(match.group(1)) not in known_warning_lines:warnings.append(warning)
    atoms=[];mapping=[]
    for atom in res.atoms:
        scope=namespaces.get(atom.line_no)
        label=f"{scope}/{atom.label}" if scope is not None else atom.label
        if label!=atom.label:mapping.append({"site_id":label,"raw_label":atom.label,"residue_scope":scope,"line":atom.line_no})
        atoms.append(replace(atom,label=label,residue=None))
    return replace(res,atoms=tuple(atoms),warnings=tuple(warnings)),{
        "mode":"scoped_residues","coordinates_modified":False,"raw_sha256":res.sha256,
        "site_identity_mapping":mapping,"unassessed_scoped_restraint_commands":sorted(commands),
        "scope":"resolved_coordinates_only_restraints_topology_and_chemical_completion_not_evaluated"}


_CENTRES = {
    1: (), 2: (("1/2", "1/2", "1/2"),),
    3: (("2/3", "1/3", "1/3"), ("1/3", "2/3", "2/3")),
    4: (("0", "1/2", "1/2"), ("1/2", "0", "1/2"), ("1/2", "1/2", "0")),
    5: (("0", "1/2", "1/2"),), 6: (("1/2", "0", "1/2"),), 7: (("1/2", "1/2", "0"),),
}


def _cell(res):
    c = res.header.cell
    values = (c.a, c.b, c.c, c.alpha, c.beta, c.gamma) if c else None
    if (values is None or any(not math.isfinite(v) for v in values)
            or any(v <= 0 for v in values[:3]) or any(not 0 < v < 180 for v in values[3:])):
        return None
    return values


def _settings(res: ResFile) -> tuple[str, ...]:
    relevant = {"CELL", "LATT", "SYMM", "HKLF", "MERG", "OMIT", "SHEL", "TWIN", "BASF", "WGHT"}
    return tuple(" ".join(line.split()).upper() for line in res.source.read_text().splitlines()
                 if line.split() and line.split()[0].upper() in relevant)


def _inventory(candidate, expert):
    first = Counter(candidate.element_of(atom) for atom in candidate.atoms)
    second = Counter(expert.element_of(atom) for atom in expert.atoms)
    return {"candidate_site_counts": dict(sorted(first.items())), "expert_site_counts": dict(sorted(second.items())),
            "missing_by_element_inventory": dict(sorted((second - first).items())),
            "extra_by_element_inventory": dict(sorted((first - second).items())),
            "composition_completeness_scope": "ASU_site_inventory_only_not_chemical_formula_or_disorder_completion"}


def _r_comparison(candidate, expert, policy):
    a, b = candidate.r_factors, expert.r_factors
    pin = policy.candidate_reflection_sha256
    comparable = (len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)
        and pin == policy.expert_reflection_sha256 and _settings(candidate) == _settings(expert)
        and a.n_all is not None and a.n_all == b.n_all and a.n_parameters is not None
        and a.n_parameters == b.n_parameters and a.n_observed is not None and a.n_observed == b.n_observed)
    safe = lambda metrics: {key: value if not isinstance(value, float) or math.isfinite(value) else None
                            for key, value in asdict(metrics).items()}
    return {"candidate": safe(a), "expert": safe(b), "comparable": comparable,
            "reason": "same_declared_reflection_pin_settings_counts_and_parameter_count" if comparable else
                      "reflection_subset_cutoffs_weighting_or_parameter_compatibility_not_established",
            "r1_observed_delta": a.r1_observed - b.r1_observed if comparable and a.r1_observed is not None and b.r1_observed is not None else None,
            "note": "A preliminary isotropic model is not expected to match an expert final R value"}


def _operators(res, gemmi, cell):
    latt = res.header.latt
    if type(latt) is not int or abs(latt) not in _CENTRES:
        raise _Unsupported("explicit_supported_latt_and_symmetry_required")
    generators = [gemmi.Op("x,y,z")]
    for text in res.header.symm:
        parsed = parse_symm(text)
        if parsed is None or any(24 % Fraction(v).denominator for v in parsed.translation):
            raise _Unsupported("symmetry_translation_not_exactly_supported_by_gemmi_denominator")
        generators.append(gemmi.Op(text))
    if latt > 0:
        generators.append(gemmi.Op("-x,-y,-z"))
    for vector in _CENTRES[abs(latt)]:
        generators.append(gemmi.Op(",".join(axis + "+" + value for axis, value in zip("xyz", vector))))
    group = gemmi.GroupOps(generators)
    group.add_missing_elements()
    operations = list(group)
    if len(operations) > 192:
        raise _Unsupported("symmetry_group_exceeds_supported_order")
    orth = cell.orth.mat.tolist()
    if not math.isfinite(cell.volume) or cell.volume <= 0 or any(not math.isfinite(v) for row in orth for v in row):
        raise _Unsupported("cell_metric_is_degenerate_or_nonfinite")
    metric = [[sum(orth[k][i] * orth[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    result = []
    for operation in operations:
        if any(value % operation.DEN for row in operation.rot for value in row):
            raise _Unsupported("nonintegral_setting_transform_not_supported")
        rotation = tuple(tuple(value // operation.DEN for value in row) for row in operation.rot)
        transformed = [[sum(rotation[k][i] * metric[k][l] * rotation[l][j] for k in range(3) for l in range(3))
                        for j in range(3)] for i in range(3)]
        if any(abs(transformed[i][j] - metric[i][j]) > 1e-6 * max(1, abs(metric[i][j])) for i in range(3) for j in range(3)):
            raise _Unsupported("declared_symmetry_does_not_preserve_cell_metric")
        result.append((rotation, tuple(value / operation.DEN for value in operation.tran), operation.triplet()))
    return tuple(result)


def _translated_groups_equal(first, second, shift, tolerance):
    if len(first) != len(second):
        return False
    for rotation, translation, _ in first:
        moved = tuple((translation[i] + shift[i] - sum(rotation[i][j] * shift[j] for j in range(3))) % 1 for i in range(3))
        if not any(rotation == other_rotation and all(abs((a - b + .5) % 1 - .5) <= tolerance for a, b in zip(moved, other_translation))
                   for other_rotation, other_translation, _ in second):
            return False
    return True


class _Matcher:
    def __init__(self, cell, policy, deadline):
        self.orth = cell.orth.mat.tolist()
        fractional = cell.frac.mat.tolist()
        self.bounds = tuple(policy.distance_tolerance_angstrom * math.sqrt(sum(v * v for v in row)) for row in fractional)
        self.policy, self.deadline, self.evaluations = policy, deadline, 0

    def distance(self, a, b):
        self.evaluations += 1
        if self.evaluations > self.policy.max_pair_evaluations or time.monotonic() > self.deadline:
            raise _Unsupported("comparison_distance_or_time_budget_exhausted")
        delta = tuple(x - y for x, y in zip(a, b))
        ranges = tuple(range(math.ceil(value - bound - 1e-12), math.floor(value + bound + 1e-12) + 1)
                       for value, bound in zip(delta, self.bounds))
        if math.prod(len(values) for values in ranges) > self.policy.max_periodic_images:
            raise _Unsupported("periodic_metric_search_budget_exhausted")
        best = None
        for shift in product(*ranges):
            fractional = tuple(delta[i] - shift[i] for i in range(3))
            cartesian = tuple(sum(self.orth[i][j] * fractional[j] for j in range(3)) for i in range(3))
            distance = math.sqrt(sum(value * value for value in cartesian))
            if distance <= self.policy.distance_tolerance_angstrom + 1e-12 and (best is None or distance < best[0]):
                best = distance, tuple(shift)
        return best


def _fractional(atom):
    return (atom.x, atom.y, atom.z)


def _apply(operation, fractional):
    rotation, translation, _ = operation
    return tuple(sum(rotation[i][j] * fractional[j] for j in range(3)) + translation[i] for i in range(3))


def _origins(candidate, expert, first_ops, second_ops, policy, deadline):
    candidate_metals = [a for a in candidate.atoms if candidate.element_of(a) in policy.metal_elements]
    expert_metals = [a for a in expert.atoms if expert.element_of(a) in policy.metal_elements]
    anchors_a = candidate_metals or [a for a in candidate.atoms if candidate.element_of(a) != "H"]
    anchors_b = expert_metals or [a for a in expert.atoms if expert.element_of(a) != "H"]
    seen = set()
    proposals = [(0., 0., 0.)]
    for first in anchors_a:
        if time.monotonic() > deadline:
            raise _Unsupported("comparison_origin_search_time_budget_exhausted")
        for second in anchors_b:
            for operation in second_ops:
                point = _apply(operation, _fractional(second))
                shift = tuple((x - y) % 1 for x, y in zip(point, _fractional(first)))
                options = tuple(tuple(dict.fromkeys((value, round(value * 24) / 24 % 1))) for value in shift)
                proposals.extend(product(*options))
                # Do not allocate an unbounded Cartesian list of origin hypotheses.
                if len(proposals) > policy.max_origin_candidates * 32:
                    break
            if len(proposals) > policy.max_origin_candidates * 32:
                break
        if len(proposals) > policy.max_origin_candidates * 32:
            break
    valid = []
    exhausted = False
    for shift in proposals:
        if time.monotonic() > deadline:
            raise _Unsupported("comparison_origin_search_time_budget_exhausted")
        key = tuple(round(value % 1, 10) for value in shift)
        if key in seen:
            continue
        seen.add(key)
        if not _translated_groups_equal(first_ops, second_ops, shift, policy.symmetry_fractional_tolerance):
            continue
        if len(valid) >= policy.max_origin_candidates:
            exhausted = True
            break
        valid.append(tuple(value % 1 for value in shift))
    return tuple(valid), not exhausted and len(proposals) <= policy.max_origin_candidates * 32


def _check_distinct_symmetry_sites(res, operations):
    seen = {}
    for index, atom in enumerate(res.atoms):
        for operation in operations:
            key = tuple(round(value % 1, 10) % 1 for value in _apply(operation, _fractional(atom)))
            previous = seen.get(key)
            if previous is not None and previous != index:
                raise _Unsupported("coincident_or_symmetry_duplicate_sites_require_disorder_aware_matching")
            seen[key] = index


def _assignment(candidate, expert, operations, shift, matcher, scipy_optimize):
    n, m = len(candidate.atoms), len(expert.atoms)
    size = n + m
    costs = [[0.0] * size for _ in range(size)]
    distances = {}
    for i, first in enumerate(candidate.atoms):
        position = tuple(value + shift[axis] for axis, value in enumerate(_fractional(first)))
        for j, second in enumerate(expert.atoms):
            best = None
            for operation in operations:
                found = matcher.distance(position, _apply(operation, _fractional(second)))
                if found is not None and (best is None or found[0] < best[0]):
                    best = found[0], operation[2], found[1]
            if best is not None:
                distances[i, j] = best
            costs[i][j] = best[0] ** 2 / matcher.policy.distance_tolerance_angstrom ** 2 / (size + 1) if best else 4 * (size + 1)
        for j in range(m, size):
            costs[i][j] = 1.0
    for i in range(n, size):
        for j in range(m):
            costs[i][j] = 1.0
    rows, columns = scipy_optimize.linear_sum_assignment(costs)
    result = []
    for i, j in zip(rows.tolist(), columns.tolist()):
        if (i, j) not in distances:
            continue
        distance, operation, translation = distances[i, j]
        a, b = candidate.atoms[i], expert.atoms[j]
        result.append(SiteComparison(a.label, b.label, candidate.element_of(a), expert.element_of(b), distance, operation, translation))
    return tuple(result)


def _categories(candidate, expert, matches, policy, *, geometry_evaluated):
    framework_ready = bool(policy.framework_candidate_site_ids and policy.framework_expert_site_ids)
    by_candidate = {a.label: candidate.element_of(a) for a in candidate.atoms}
    by_expert = {a.label: expert.element_of(a) for a in expert.atoms}
    if not set(policy.framework_candidate_site_ids).issubset(by_candidate) or not set(policy.framework_expert_site_ids).issubset(by_expert):
        raise _Unsupported("declared_framework_site_identity_missing")

    def category(label, element, framework):
        if element == "H": return "hydrogen"
        if element in policy.metal_elements: return "metal"
        if framework_ready and label in framework: return "framework"
        return "light"

    result = []
    for name in ("metal", "framework", "light", "hydrogen"):
        if name == "framework" and not framework_ready:
            result.append(CategoryComparison(name, None, None, None, None, None, None, None, None,
                                              "NOT_EVALUATED_explicit_framework_membership_required"))
            continue
        first_rows = [a.label for a in candidate.atoms if category(a.label,candidate.element_of(a),policy.framework_candidate_site_ids)==name]
        second_rows = [a.label for a in expert.atoms if category(a.label,expert.element_of(a),policy.framework_expert_site_ids)==name]
        if not geometry_evaluated:
            result.append(CategoryComparison(name, len(first_rows), len(second_rows), None, None, None, None, None, None,
                                              "GEOMETRY_NOT_EVALUATED"))
            continue
        first,second=set(first_rows),set(second_rows)
        recovered = [match for match in matches if match.expert_site_id in second]
        matched_candidate = {match.candidate_site_id for match in matches}
        same = sum(match.candidate_element == match.expert_element for match in recovered)
        result.append(CategoryComparison(name, len(first), len(second), len(recovered), len(second) - len(recovered),
            len(first - matched_candidate), same, len(recovered) - same,
            math.sqrt(sum(match.distance_angstrom ** 2 for match in recovered) / len(recovered)) if recovered else None,
            "EXPLICIT_ROLE_MEMBERSHIP" if name != "light" or framework_ready else "includes_unclassified_framework_and_solvent"))
    return tuple(result)


def compare_sealed_res(run_root: Path, *, private_root: Path, expected_seal_sha256: str,
        expert_path: Path, expert_root: Path, policy: ComparisonPolicy,
        expected_expert_sha256: str | None = None, output_dir: Path | None = None,
        model_role: str = "model") -> ModelComparisonReport:
    """Read the held-out expert only after an independently pinned candidate exists.

    Candidate selection is finished before this call. This comparator must never
    be used to choose/rewrite a solver candidate. Its settings/origin search is
    bounded to the current cell basis; unsupported settings give partial metadata.
    Scientific files/reports are written only into a fresh private subdirectory.
    """
    try:
        seal = verify_seal(run_root, private_root=private_root, expected_seal_sha256=expected_seal_sha256)
        candidate_bytes = read_sealed_output(seal, private_root=private_root, role=model_role)
        if not candidate_bytes:
            raise SealError("Candidate empty")
    except (SealError, OSError):
        return ModelComparisonReport("NOT_EVALUATED", ("candidate_must_be_sealed_and_independently_pinned_before_expert_access",))
    if seal.track == "component_regression":
        return ModelComparisonReport("NOT_EVALUATED", ("expert_seeded_component_regression_is_not_an_independent_candidate",),
            hashlib.sha256(candidate_bytes).hexdigest(), candidate_seal_sha256=seal.seal_sha256,
            expert_used_for_solver=True)
    # Expert contents are first opened here, after all candidate identity gates.
    try:
        expert_bytes, expert_sha = read_private_reference(expert_path, case_root=expert_root)
        if expected_expert_sha256 is not None and expert_sha != expected_expert_sha256:
            raise SealError("Expert pin mismatch")
        directory = _fresh_private_directory(output_dir or (Path(run_root) / "expert-comparison"), private_root)
        for name, content in (("candidate.res", candidate_bytes), ("expert.res", expert_bytes)):
            with (directory / name).open("xb") as stream:
                stream.write(content)
        candidate, expert = read_res(directory / "candidate.res"), read_res(directory / "expert.res")
        if candidate.sha256 != hashlib.sha256(candidate_bytes).hexdigest() or expert.sha256 != expert_sha:
            raise SealError("Parsed snapshot differs")
        verify_seal(run_root, private_root=private_root, expected_seal_sha256=seal.seal_sha256)
    except (SealError, OSError, ValueError):
        return ModelComparisonReport("NOT_EVALUATED", ("explicit_expert_or_private_snapshot_identity_failed",),
                                      hashlib.sha256(candidate_bytes).hexdigest(), candidate_seal_sha256=seal.seal_sha256)
    inventory = _inventory(candidate, expert)
    settings = {"candidate_cell": _cell(candidate), "expert_cell": _cell(expert),
                "candidate_latt": candidate.header.latt, "expert_latt": expert.header.latt,
                "candidate_symm": candidate.header.symm, "expert_symm": expert.header.symm,
                "refinement_settings_equal": _settings(candidate) == _settings(expert),
                "comparison_policy":to_jsonable(asdict(policy)),
                "metric_frame":"expert_cell_no_general_basis_transform"}
    reasons = []
    matches, origin, exhaustive, evaluated = (), None, False, 0
    try:
        if policy.coordinate_identity_mode=="scoped_residues":
            candidate,first_view=_coordinate_identity_view(candidate)
            expert,second_view=_coordinate_identity_view(expert)
            settings["coordinate_identity_views"]={"candidate":first_view,"expert":second_view}
        if (not candidate.atoms or not expert.atoms or max(len(candidate.atoms), len(expert.atoms)) > policy.max_sites
                or candidate.warnings or expert.warnings):
            raise _Unsupported("missing_sites_parse_warnings_or_site_budget_exhausted")
        for res in (candidate, expert):
            if len({atom.label.casefold() for atom in res.atoms}) != len(res.atoms):
                raise _Unsupported("duplicate_atom_identity_or_disorder_requires_an_audited_matcher")
            if any(atom.part not in (None, 0) or atom.residue is not None or atom.afix not in (None, 0)
                   or not math.isfinite(atom.occupancy_code) or abs(atom.occupancy_code) >= 20
                   or any(not math.isfinite(v) for v in atom.u_values)
                   or any(not math.isfinite(v) or abs(v) >= 5 for v in _fractional(atom)) for atom in res.atoms):
                raise _Unsupported("encoded_coordinates_disorder_or_coupled_sites_not_supported")
        first_cell, second_cell = _cell(candidate), _cell(expert)
        if first_cell is None or second_cell is None:
            raise _Unsupported("cell_metadata_missing")
        if (max(abs(a - b) for a, b in zip(first_cell[:3], second_cell[:3])) > policy.cell_length_tolerance_angstrom
                or max(abs(a - b) for a, b in zip(first_cell[3:], second_cell[3:])) > policy.cell_angle_tolerance_deg):
            raise _Unsupported("different_cell_or_basis_setting_requires_a_change_of_basis_provider")
        try:
            gemmi = importlib.import_module("gemmi")
            scipy_optimize = importlib.import_module("scipy.optimize")
        except ImportError as exc:
            raise _Unsupported("optional_gemmi_or_scipy_matching_provider_unavailable") from exc
        cell = gemmi.UnitCell(*second_cell)
        deadline = time.monotonic() + policy.max_wall_seconds
        first_ops, second_ops = _operators(candidate, gemmi, cell), _operators(expert, gemmi, cell)
        _check_distinct_symmetry_sites(candidate, first_ops)
        _check_distinct_symmetry_sites(expert, second_ops)
        shifts, exhaustive = _origins(candidate, expert, first_ops, second_ops, policy, deadline)
        if not shifts:
            raise _Unsupported("declared_symmetries_are_not_origin_equivalent_in_the_supported_cell_basis")
        matcher = _Matcher(cell, policy, deadline)
        best_key = None
        if policy.origin_selection=="metal_anchors":
            first_metal=replace(candidate,atoms=tuple(a for a in candidate.atoms if candidate.element_of(a) in policy.metal_elements))
            second_metal=replace(expert,atoms=tuple(a for a in expert.atoms if expert.element_of(a) in policy.metal_elements))
            if not first_metal.atoms or not second_metal.atoms:
                raise _Unsupported("metal_origin_selection_requires_declared_metal_sites_on_both_sides")
            selected_origin=None
            for shift in shifts:
                anchors=_assignment(first_metal,second_metal,second_ops,shift,matcher,scipy_optimize)
                evaluated+=1
                key=(-len(anchors),sum(match.distance_angstrom**2 for match in anchors),shift)
                if best_key is None or key<best_key:best_key,selected_origin=key,shift
            matches=_assignment(candidate,expert,second_ops,selected_origin,matcher,scipy_optimize)
            origin=selected_origin
            settings["full_site_assignments_evaluated"]=1
            shifts=()
        for shift in shifts:
            trial = _assignment(candidate, expert, second_ops, shift, matcher, scipy_optimize)
            evaluated += 1
            key = (-sum(match.expert_element in policy.metal_elements for match in trial), -len(trial),
                   sum(match.distance_angstrom ** 2 for match in trial), shift)
            if best_key is None or key < best_key:
                best_key, matches, origin = key, trial, shift
        if not exhaustive:
            reasons.append("origin_candidate_budget_reached_best_of_supported_search_only")
    except (_Unsupported, ValueError, RuntimeError, OverflowError) as exc:
        reasons.append(str(exc) if isinstance(exc, _Unsupported) else "optional_geometric_matching_failed")
    try:
        categories = _categories(candidate, expert, matches, policy, geometry_evaluated=origin is not None)
    except _Unsupported as exc:
        reasons.append(str(exc))
        categories = ()
    status = "MEASURED_REVIEW_REQUIRED" if origin is not None and not reasons else "PARTIAL"
    if not policy.threshold_approval_reference:
        reasons.append("scientific_matching_tolerance_has_not_been_owner_approved")
    reasons.append("framework_topology_chirality_and_scientific_completion_require_independent_review")
    report = ModelComparisonReport(status, tuple(reasons), candidate.sha256, expert_sha, seal.seal_sha256,
        categories, matches, origin, inventory, settings, _r_comparison(candidate, expert, policy),
        exhaustive, evaluated, directory)
    (directory / "comparison.json").write_text(json.dumps(report.to_record(), sort_keys=True, indent=2, allow_nan=False))
    return report
