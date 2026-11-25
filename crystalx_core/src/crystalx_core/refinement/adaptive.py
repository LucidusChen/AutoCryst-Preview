"""Answer-free proposals for stable assignment, reversible pruning and local growth.

All functions are pure with respect to model files and caller state. A controller must
commit the returned state only when its corresponding refinement is retained. Model
probabilities remain uncalibrated observations; no value here is manufactured as confidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
import math
from statistics import median

from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.geometry.model import orthogonalisation_matrix
from crystalx_core.refinement.cycle import METALS, cell_values,displacement_value
from crystalx_core.workflow.reconstruction import ReconstructionBlocked


HALOGENS = frozenset(("F", "Cl", "Br", "I", "At"))


@dataclass(frozen=True)
class AdaptivePolicy:
    """Explicit proposal switches; all adaptive behavior is off for legacy callers."""

    hysteresis_enabled: bool = False
    uiso_checks_enabled: bool = False
    uiso_directional_proposals: bool = True
    pruning_enabled: bool = False
    growth_filter_enabled: bool = False
    confirmations: int = 2
    max_change_fraction: float = 0.05
    freeze_confirmations: int = 3
    thaw_on_uiso_anomaly: bool = True
    u_abs_low: float = 0.005
    u_abs_high: float = 0.30
    u_relative_low: float = 0.5
    u_relative_high: float = 2.0
    reference_min_count: int = 3
    reference_min_unchanged_rounds: int = 2
    prune_high_u: bool = True
    prune_isolated: bool = True
    prune_close: bool = True
    prune_evidence_mode: str = "any"
    prune_confirmations: int = 2
    max_prune_fraction: float = 0.05
    close_distance_angstrom: float = 1.0
    close_survivor_policy: str = "lower_u"
    minimum_bond_distance_angstrom: float = 0.9
    covalent_radius_scale: float = 1.25
    maximum_bond_distance_angstrom: float = 2.4
    maximum_metal_bond_distance_angstrom: float = 3.0
    isolation_exempt_oxygen: bool = True
    growth_relative_peak_height: float = 0.15
    early_growth_rounds: int = 3
    early_peak_limit: int = 24
    growth_bond_distance_angstrom: float = 2.4
    growth_metal_distance_angstrom: float = 3.0
    max_sites: int = 20_000
    max_image_candidates: int = 2_000_000
    max_neighbor_evaluations: int = 2_000_000
    max_neighbor_pairs: int = 1_000_000

    def __post_init__(self):
        for name in ("hysteresis_enabled", "uiso_checks_enabled", "uiso_directional_proposals", "pruning_enabled",
                     "growth_filter_enabled", "thaw_on_uiso_anomaly", "prune_high_u",
                     "prune_isolated", "prune_close", "isolation_exempt_oxygen"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("confirmations", "freeze_confirmations", "reference_min_count",
                     "reference_min_unchanged_rounds", "prune_confirmations", "early_peak_limit",
                     "max_sites", "max_image_candidates", "max_neighbor_evaluations", "max_neighbor_pairs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.early_growth_rounds) is not int or self.early_growth_rounds < 0:
            raise ValueError("early_growth_rounds must be nonnegative")
        if self.prune_evidence_mode not in ("any", "high_u_and_isolated"):
            raise ValueError("invalid pruning evidence mode")
        if self.close_survivor_policy not in ("lower_u", "reference_consistency"):
            raise ValueError("Unsupported close-site survivor policy")
        for name in ("max_change_fraction", "max_prune_fraction", "growth_relative_peak_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("u_abs_low", "u_abs_high", "u_relative_low", "u_relative_high",
                     "close_distance_angstrom", "minimum_bond_distance_angstrom",
                     "covalent_radius_scale", "maximum_bond_distance_angstrom",
                     "maximum_metal_bond_distance_angstrom", "growth_bond_distance_angstrom",
                     "growth_metal_distance_angstrom"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not (self.u_abs_low < self.u_abs_high and self.u_relative_low < 1 < self.u_relative_high):
            raise ValueError("Uiso absolute and relative limits are inconsistent")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unsupported adaptive policy fields")
        return cls(**value)


@dataclass(frozen=True)
class AssignmentProposal:
    overrides: dict[str, str]
    state: dict
    evidence: dict


@dataclass(frozen=True)
class PruningProposal:
    site_ids: tuple[str, ...]
    state: dict
    evidence: dict


def _finite_u(atom,model=None):
    value = displacement_value(model,atom) if model is not None else atom.u_iso
    return value if value is not None and math.isfinite(value) else None


def _references(model, policy, assignment_state):
    values = {}
    for atom in model.atoms:
        element = model.element_of(atom)
        u = _finite_u(atom,model)
        row = assignment_state.get(atom.label, {})
        unchanged = row.get("observed_element") in (None, element)
        count = row.get("unchanged_rounds", 0) if unchanged else 0
        if (u is not None and policy.u_abs_low < u < policy.u_abs_high
                and count >= policy.reference_min_unchanged_rounds):
            values.setdefault(element, []).append(u)
    return {element: {"count": len(rows), "median": median(rows) if len(rows) >= policy.reference_min_count else None}
            for element, rows in values.items()}


def _u_evidence(atom, element, references, policy,*,model=None):
    u = _finite_u(atom,model)
    reference = references.get(element, {"count": 0, "median": None})
    center = reference["median"]
    low = u is not None and (u <= policy.u_abs_low or center is not None and u < center * policy.u_relative_low)
    high = u is not None and (u >= policy.u_abs_high or center is not None and u > center * policy.u_relative_high)
    return {"u_iso": u, "reference_median": center, "reference_count": reference["count"],
            "reference_available": center is not None,
            "u_over_reference_median": u / center if u is not None and center is not None else None,
            "absolute_low_limit": policy.u_abs_low, "absolute_high_limit": policy.u_abs_high,
            "relative_low_limit": center * policy.u_relative_low if center is not None else None,
            "relative_high_limit": center * policy.u_relative_high if center is not None else None,
            "classification": "low" if low else "high" if high else "ordinary" if u is not None else "unavailable"}


def uiso_consistency(model, *, policy: AdaptivePolicy, state=None):
    """Report post-refinement U evidence without advancing any proposal counters."""
    references = _references(model, policy, (state or {}).get("assignment", {}))
    sites = []
    for atom in model.atoms:
        evidence = _u_evidence(atom, model.element_of(atom), references, policy,model=model)
        sites.append({"site_id": atom.label, "element": model.element_of(atom), **evidence,
                      "suggested_atomic_number_direction": "higher" if evidence["classification"] == "low" else
                          "lower_or_prune" if evidence["classification"] == "high" else "none"})
    return {"same_element_u_references": references, "sites": sites,
            "state_advanced": False, "scientific_validation_performed": False,
            "reference_definition": "previously_unchanged_element_and_positive_bounded_U_only"}


def propose_assignments(model, predictions, *, policy: AdaptivePolicy, state=None,
                        protected_site_ids=(), proposed_elements=None):
    """Return auditable element overrides without touching raw probability rows.

    Uiso low/high evidence supports heavier/lighter alternatives respectively.
    When explicitly enabled, a directional top-K alternative may replace a current
    or contradictory protocol choice; both choices and untouched raw probabilities
    remain in the audit. Missing U evidence cannot authorize a checked switch.
    Confirmation counters concern repeated final proposal elements, including
    explicit coverage decisions and U-supported model alternatives.
    """
    result_state = deepcopy(state or {})
    prior = result_state.setdefault("assignment", {})
    references = _references(model, policy, prior)
    by_id = {row.site_id: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ReconstructionBlocked("adaptive_prediction_site_identity_is_not_unique")
    proposed_elements = proposed_elements or {}
    protected = set(protected_site_ids)
    overrides, decisions, eligible = {}, [], []
    for atom in model.atoms:
        key, current = atom.label, model.element_of(atom)
        if current not in ATOMIC_NUMBERS:
            raise ReconstructionBlocked("adaptive_current_element_is_unknown")
        prediction = by_id.get(key)
        if prediction is None or not prediction.topk:
            raise ReconstructionBlocked("adaptive_prediction_site_missing")
        if any(p.atomic_number not in ATOMIC_NUMBERS.values() or not math.isfinite(p.probability)
               or not 0 <= p.probability <= 1 for p in prediction.topk):
            raise ReconstructionBlocked("adaptive_prediction_probability_invalid")
        top = prediction.topk[0]
        symbols = {number: symbol for symbol, number in ATOMIC_NUMBERS.items()}
        proposed = proposed_elements.get(key, symbols.get(top.atomic_number))
        if proposed not in ATOMIC_NUMBERS:
            raise ReconstructionBlocked("adaptive_proposed_element_is_unknown")
        protocol_proposed = proposed
        proposal_source = "explicit_protocol_choice" if key in proposed_elements else "raw_model_top1"
        previous = prior.get(key, {})
        unchanged_rounds = previous.get("unchanged_rounds", 0) + 1 if previous.get("observed_element") == current else 1
        evidence = _u_evidence(atom, current, references, policy,model=model)
        anomaly = evidence["classification"] in ("low", "high")
        frozen = bool(previous.get("frozen", False))
        if frozen and policy.thaw_on_uiso_anomaly and anomaly:
            frozen = False
        direction = "heavier" if evidence["classification"] == "low" else "lighter" if evidence["classification"] == "high" else None
        if policy.uiso_checks_enabled and policy.uiso_directional_proposals and direction is not None:
            direction_supported = (ATOMIC_NUMBERS[proposed] > ATOMIC_NUMBERS[current] if direction == "heavier"
                                   else ATOMIC_NUMBERS[proposed] < ATOMIC_NUMBERS[current])
            if proposed == current or not direction_supported:
                alternative = next((p for p in prediction.topk if
                    (p.atomic_number > ATOMIC_NUMBERS[current] if direction == "heavier"
                     else p.atomic_number < ATOMIC_NUMBERS[current])), None)
                if alternative is not None:
                    proposed = symbols[alternative.atomic_number]
                    proposal_source = "uiso_supported_model_alternative"
        supported = (not policy.uiso_checks_enabled or
                     direction == "heavier" and ATOMIC_NUMBERS[proposed] > ATOMIC_NUMBERS[current] or
                     direction == "lighter" and ATOMIC_NUMBERS[proposed] < ATOMIC_NUMBERS[current])
        alternatives = [{"element": symbols[p.atomic_number], "probability": p.probability}
                        for p in prediction.topk if p.atomic_number in symbols and direction is not None
                        and ((p.atomic_number > ATOMIC_NUMBERS[current]) if direction == "heavier" else (p.atomic_number < ATOMIC_NUMBERS[current]))]
        proposal_streak = (previous.get("proposal_streak", 0) + 1
                           if previous.get("proposed_element") == proposed else 1)
        confirmation_streak = (previous.get("confirmation_streak", 0) + 1
                               if proposed == current and not anomaly else 0)
        if policy.hysteresis_enabled and confirmation_streak >= policy.freeze_confirmations:
            frozen = True
        row = {"site_id": key, "current_element": current, "proposed_element": proposed,
               "protocol_proposed_element": protocol_proposed, "proposal_source": proposal_source,
               "top1_atomic_number": top.atomic_number, "top1_probability": top.probability,
               "probabilities_calibrated": False, "u_evidence": evidence,
               "u_directional_alternatives": alternatives, "proposal_streak": proposal_streak,
               "confirmation_streak": confirmation_streak, "frozen": frozen,
               "decision": "retain", "reason": "same_element" if proposed == current else "pending"}
        overrides[key] = current
        prior[key] = {"observed_element": current, "unchanged_rounds": unchanged_rounds,
                      "proposed_element": proposed, "proposal_streak": proposal_streak,
                      "confirmation_streak": confirmation_streak, "frozen": frozen}
        if key in protected or current == "H":
            row["reason"] = "protected_site" if key in protected else "existing_hydrogen_preserved"
        elif frozen and policy.hysteresis_enabled:
            row["reason"] = "frozen_after_repeated_confirmation"
        elif proposed != current:
            if policy.hysteresis_enabled and proposal_streak < policy.confirmations:
                row["reason"] = "awaiting_repeated_protocol_proposal"
            elif not supported:
                row["reason"] = "uiso_does_not_support_proposed_direction"
            else:
                eligible.append((key, proposed, row))
        decisions.append(row)
    limit = math.floor(len(model.atoms) * policy.max_change_fraction) if policy.hysteresis_enabled else len(model.atoms)
    # A fractional cap is literal: a tiny model can authorize zero changes. No
    # hidden minimum-one exception exceeds the requested percentage.
    eligible.sort(key=lambda item: (-item[2]["proposal_streak"], item[0]))
    for index, (key, proposed, row) in enumerate(eligible):
        if index < limit:
            overrides[key] = proposed
            row.update(decision="change", reason="repeated_proposal_with_uiso_support")
            prior[key]["confirmation_streak"] = 0
        else:
            row["reason"] = "per_round_change_cap"
    # Q peaks are not established sites; the caller's separate growth policy
    # decides which may be accepted, without borrowing prior atom confidence.
    return AssignmentProposal(overrides, result_state,
        {"policy": policy.to_dict(), "same_element_u_references": references,
         "change_limit": limit, "changed_site_count": sum(row["decision"] == "change" for row in decisions),
         "decisions": decisions, "probabilities_are_not_confidence": True})


def _symmetry_fractional(model, gemmi):
    centres = {1: ((0, 0, 0),), 2: ((0, 0, 0), (12, 12, 12)),
               3: ((0, 0, 0), (16, 8, 8), (8, 16, 16)),
               4: ((0, 0, 0), (0, 12, 12), (12, 0, 12), (12, 12, 0)),
               5: ((0, 0, 0), (0, 12, 12)), 6: ((0, 0, 0), (12, 0, 12)),
               7: ((0, 0, 0), (12, 12, 0))}
    latt = model.header.latt
    if type(latt) is not int or abs(latt) not in centres:
        raise ReconstructionBlocked("adaptive_geometry_requires_explicit_supported_latt")
    group = gemmi.GroupOps([gemmi.Op("x,y,z"), *(gemmi.Op(text) for text in model.header.symm)])
    group.cen_ops = [list(value) for value in centres[abs(latt)]]
    group.add_missing_elements()
    if latt > 0:
        group.add_inversion()
    if not 1 <= len(group) <= 192:
        raise ReconstructionBlocked("adaptive_symmetry_budget_exceeded")
    return tuple(group)


def _neighbor_index(model, policy, radius):
    try:
        import gemmi
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise ReconstructionBlocked("adaptive_geometry_provider_unavailable") from error
    n = len(model.atoms)
    if not 1 <= n <= policy.max_sites or model.header.cell is None:
        raise ReconstructionBlocked("adaptive_geometry_site_or_cell_bound_exceeded")
    try:
        matrix = np.asarray(orthogonalisation_matrix(cell_values(model.header.cell)), dtype=float)
        inverse = np.linalg.inv(matrix)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        raise ReconstructionBlocked("adaptive_geometry_metric_invalid") from error
    if not bool(np.isfinite(matrix).all() and np.isfinite(inverse).all()):
        raise ReconstructionBlocked("adaptive_geometry_metric_nonfinite")
    extents = tuple(math.ceil(radius * float(np.linalg.norm(row))) for row in inverse)
    operations = _symmetry_fractional(model, gemmi)
    metric = matrix.T @ matrix
    tolerance = np.sqrt(np.outer(np.diag(metric), np.diag(metric))) * 1e-6
    for operation in operations:
        rotation = np.asarray(operation.rot, dtype=float) / operation.DEN
        if bool(np.any(np.abs(rotation.T @ metric @ rotation - metric) > tolerance + 1e-10)):
            raise ReconstructionBlocked("adaptive_symmetry_incompatible_with_cell_metric")
    translation_count = math.prod(2 * extent + 1 for extent in extents)
    image_count = n * len(operations) * translation_count
    if image_count > policy.max_image_candidates:
        raise ReconstructionBlocked("adaptive_geometry_image_budget_exceeded")
    translations = tuple(product(*(range(-extent, extent + 1) for extent in extents)))
    original = np.asarray([(atom.x, atom.y, atom.z) for atom in model.atoms], dtype=float)
    if not bool(np.isfinite(original).all()):
        raise ReconstructionBlocked("adaptive_geometry_coordinates_nonfinite")
    wrapped = original % 1.
    positions, atom_indices = [], []
    for operation in operations:
        transformed = np.asarray([operation.apply_to_xyz(point.tolist()) for point in wrapped]) % 1.
        for translation in translations:
            positions.append((transformed + np.asarray(translation)) @ matrix.T)
            atom_indices.extend(range(n))
    images = np.concatenate(positions, axis=0)
    return {"np": np, "tree": cKDTree(images), "images": images,
            "atom_indices": atom_indices, "base": wrapped @ matrix.T, "matrix": matrix,
            "operations": operations,
            "image_count": image_count, "symmetry_operation_count": len(operations),
            "translation_extents": extents,
            "radii": [float(gemmi.Element(model.element_of(atom)).covalent_r) for atom in model.atoms]}


def geometry_summary(model, *, policy: AdaptivePolicy):
    """Count contact evidence using bounded Cartesian images, including declared symmetry."""
    radius = max(policy.maximum_bond_distance_angstrom,
                 policy.maximum_metal_bond_distance_angstrom, policy.close_distance_angstrom)
    index = _neighbor_index(model, policy, radius)
    pairs, self_bonds, evaluations = {}, set(), 0
    elements = [model.element_of(atom) for atom in model.atoms]
    np = index["np"]
    for i, position in enumerate(index["base"]):
        for image in index["tree"].query_ball_point(position, radius):
            evaluations += 1
            if evaluations > policy.max_neighbor_evaluations:
                raise ReconstructionBlocked("adaptive_neighbor_evaluation_budget_exceeded")
            j = index["atom_indices"][image]
            distance = float(np.linalg.norm(index["images"][image] - position))
            if i == j:
                if policy.minimum_bond_distance_angstrom <= distance <= min(
                        policy.maximum_metal_bond_distance_angstrom if elements[i] in METALS else policy.maximum_bond_distance_angstrom,
                        2 * index["radii"][i] * policy.covalent_radius_scale):
                    self_bonds.add(i)
                continue
            key = tuple(sorted((i, j)))
            if key not in pairs or distance < pairs[key]:
                pairs[key] = distance
            if len(pairs) > policy.max_neighbor_pairs:
                raise ReconstructionBlocked("adaptive_neighbor_pair_budget_exceeded")
    neighbors = {atom.label: int(i in self_bonds) for i, atom in enumerate(model.atoms)}
    nearest = {atom.label: None for atom in model.atoms}
    close, bond_lengths = [], []
    for (i, j), distance in sorted(pairs.items()):
        first, second = model.atoms[i], model.atoms[j]
        for atom in (first, second):
            old = nearest[atom.label]
            nearest[atom.label] = distance if old is None else min(old, distance)
        if elements[i] != "H" and elements[j] != "H" and distance < policy.close_distance_angstrom:
            close.append({"site_ids": (first.label, second.label), "distance_angstrom": distance})
        cap = (policy.maximum_metal_bond_distance_angstrom if elements[i] in METALS or elements[j] in METALS
               else policy.maximum_bond_distance_angstrom)
        upper = min(cap, (index["radii"][i] + index["radii"][j]) * policy.covalent_radius_scale)
        if policy.minimum_bond_distance_angstrom <= distance <= upper:
            neighbors[first.label] += 1
            neighbors[second.label] += 1
            bond_lengths.append(distance)
    exempt = tuple(atom.label for atom in model.atoms if model.element_of(atom) in METALS | HALOGENS
                   or policy.isolation_exempt_oxygen and model.element_of(atom) == "O"
                   or model.element_of(atom) == "H")
    isolated = tuple(atom.label for atom in model.atoms if neighbors[atom.label] == 0)
    nonexempt = tuple(key for key in isolated if key not in set(exempt))
    return {"site_count": len(model.atoms), "bonded_neighbor_counts": neighbors,
            "nearest_neighbor_distance_angstrom": nearest, "close_pairs": close,
            "close_pair_count": len(close), "close_pair_scope": "nonhydrogen_only",
            "isolated_site_ids": isolated, "isolated_site_count": len(isolated),
            "isolation_exempt_site_ids": exempt, "isolated_nonexempt_site_ids": nonexempt,
            "isolated_nonexempt_site_count": len(nonexempt),
            "bond_length_distribution": {"count": len(bond_lengths),
                "minimum_angstrom": min(bond_lengths) if bond_lengths else None,
                "median_angstrom": median(bond_lengths) if bond_lengths else None,
                "maximum_angstrom": max(bond_lengths) if bond_lengths else None},
            "provider": "bounded_scipy_cartesian_symmetry_and_periodic_images",
            "coordinates_modified": False, "image_count": index["image_count"],
            "symmetry_operation_count": index["symmetry_operation_count"],
            "translation_extents": index["translation_extents"], "neighbor_evaluations": evaluations}


def propose_pruning(model, *, policy: AdaptivePolicy, state=None, protected_site_ids=(), baseline_site_ids=()):
    """Propose limited deletions; retain complete atom fields and reasons in the audit."""
    result_state = deepcopy(state or {})
    prior = result_state.setdefault("pruning", {})
    if not policy.pruning_enabled:
        return PruningProposal((), result_state, {"policy": policy.to_dict(), "enabled": False,
                                                "geometry": None, "decisions": []})
    geometry = geometry_summary(model, policy=policy)
    protected, baseline = set(protected_site_ids), set(baseline_site_ids)
    atoms = {atom.label: atom for atom in model.atoms}
    references = _references(model, policy, result_state.get("assignment", {}))
    reasons = {key: [] for key in atoms}
    for key, atom in atoms.items():
        u = _finite_u(atom,model)
        if policy.prune_high_u and u is not None and u >= policy.u_abs_high:
            reasons[key].append({"kind": "high_u", "u_iso": u, "threshold": policy.u_abs_high})
        if policy.prune_isolated and key in geometry["isolated_nonexempt_site_ids"]:
            reasons[key].append({"kind": "isolated", "bonded_neighbor_count": 0,
                                 "nearest_neighbor_distance_angstrom": geometry["nearest_neighbor_distance_angstrom"][key]})
    if policy.prune_evidence_mode == "high_u_and_isolated":
        for key, rows in reasons.items():
            if {row["kind"] for row in rows} != {"high_u", "isolated"}:
                reasons[key] = []
    def rank(key):
        atom = atoms[key]
        u = _finite_u(atom,model)
        center = references.get(model.element_of(atom), {}).get("median") or 0.05
        badness = abs(math.log(max(u, 0.000001) / center)) if u is not None else 0.
        value = u if u is not None else 0.
        return (key not in protected, u is None,
                value if policy.close_survivor_policy == "lower_u" else badness,
                key not in baseline, key.casefold())
    if policy.prune_close:
        # A deterministic independent set preserves both endpoints of an A-B-C
        # chain when A and C themselves are not too close.
        adjacency = {}
        for pair in geometry["close_pairs"]:
            first, second = pair["site_ids"]
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        seen = set()
        for key in sorted(adjacency):
            if key in seen:
                continue
            pending, cluster = [key], set()
            while pending:
                current = pending.pop()
                if current in cluster:
                    continue
                cluster.add(current)
                pending.extend(adjacency[current] - cluster)
            seen.update(cluster)
            retained = []
            for current in sorted(cluster, key=rank):
                blockers = [key for key in retained if key in adjacency[current]]
                if not blockers or current in protected:
                    retained.append(current)
                    continue
                reasons[current].append({"kind": "too_close", "retained_site_id": blockers[0],
                    "threshold_angstrom": policy.close_distance_angstrom,
                    "cluster_site_ids": tuple(sorted(cluster)), "retention_rank": rank(blockers[0])})
    decisions, eligible = [], []
    for key, atom in atoms.items():
        kinds = tuple(sorted(row["kind"] for row in reasons[key]))
        previous = prior.get(key, {})
        reason_streaks = {kind: previous.get("reason_streaks", {}).get(kind, 0) + 1 for kind in kinds}
        streak = max(reason_streaks.values(), default=0)
        prior[key] = {"streak": streak, "reason_kinds": kinds, "reason_streaks": reason_streaks}
        row = {"site_id": key, "element": model.element_of(atom), "u_iso": _finite_u(atom,model),
               "reasons": reasons[key], "confirmation_streak": streak, "decision": "retain",
               "protected": key in protected, "atom_record": asdict(atom)}
        if kinds and key not in protected and model.element_of(atom) != "H" and streak >= policy.prune_confirmations:
            eligible.append(row)
        decisions.append(row)
    eligible.sort(key=lambda row: (-len(row["reasons"]), -(row["u_iso"] or 0.), row["site_id"]))
    limit = min(math.floor(len(atoms) * policy.max_prune_fraction), max(0, len(atoms) - 1))
    deleted = tuple(row["site_id"] for row in eligible[:limit])
    for row in decisions:
        if row["site_id"] in deleted:
            row["decision"] = "delete_proposal"
        elif row in eligible:
            row["decision"] = "retain_pruning_cap"
    return PruningProposal(deleted, result_state, {"policy": policy.to_dict(), "enabled": True,
        "prune_limit": limit, "proposed_deleted_site_ids": deleted, "decisions": decisions,
        "geometry": geometry, "raw_input_preserved": True, "rollback_requires_previous_model_and_uncommitted_state": True})


def metal_ripple_rejections(model,*,policy:AdaptivePolicy,radius:float,index=None):
    """A geometric exclusion only; no residual peak becomes an atom identity."""
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<0:
        raise ValueError("metal ripple radius must be finite and nonnegative")
    if not radius:return {}
    index=index or _neighbor_index(model,policy,radius)
    result={};np=index['np']
    for peak in model.q_peaks:
        pos=(np.asarray((peak.x,peak.y,peak.z))%1.)@index['matrix'].T
        best=None
        for image in index['tree'].query_ball_point(pos,radius):
            atom=model.atoms[index['atom_indices'][image]]
            if model.element_of(atom) not in METALS:continue
            distance=float(np.linalg.norm(index['images'][image]-pos))
            if distance<radius and (best is None or (distance,atom.label)<best):best=(distance,atom.label)
        if best is not None:result[peak.label]={'reason':'metal_ripple_exclusion','metal_site_id':best[1],
            'metal_distance_angstrom':best[0],'metal_ripple_exclusion_angstrom':radius}
    return result


def select_growth_peaks(model, *, policy: AdaptivePolicy, round_number: int, limit: int,
                        minimum_height: float, minimum_distance: float,metal_ripple_exclusion_angstrom:float=0.):
    """Filter residual observations by relative height and scaffold contact distance."""
    if not policy.growth_filter_enabled:
        raise ValueError("Filtered growth helper requires an explicit enabled policy")
    if type(round_number) is not int or round_number < 1 or type(limit) is not int or limit < 0:
        raise ValueError("Invalid growth round or peak budget")
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in (minimum_height, minimum_distance)):
        raise ValueError("Growth height and distance limits must be finite and positive")
    if len(model.q_peaks) > policy.max_sites:
        raise ReconstructionBlocked("adaptive_growth_peak_budget_exceeded")
    if any(not math.isfinite(value) for peak in model.q_peaks for value in (peak.height,peak.x,peak.y,peak.z)):
        raise ReconstructionBlocked("adaptive_growth_peak_nonfinite")
    actual_limit = max(limit, policy.early_peak_limit) if limit and round_number <= policy.early_growth_rounds else limit
    radius = max(policy.growth_bond_distance_angstrom, policy.growth_metal_distance_angstrom, minimum_distance,metal_ripple_exclusion_angstrom)
    index = _neighbor_index(model, policy, radius)
    excluded=metal_ripple_rejections(model,policy=policy,radius=metal_ripple_exclusion_angstrom,index=index)
    strongest = max((peak.height for peak in model.q_peaks if peak.label not in excluded and math.isfinite(peak.height) and peak.height > 0), default=0.)
    threshold = max(minimum_height, strongest * policy.growth_relative_peak_height)
    selected, decisions, selected_cartesian = [], [], []
    np = index["np"]
    for peak in sorted(model.q_peaks, key=lambda value: (-value.height, value.label)):
        row = {"peak_id": peak.label, "height": peak.height,
               "strongest_residual_peak_height": strongest, "effective_height_threshold": threshold,
               "relative_height": peak.height / strongest if strongest else None, "decision": "reject"}
        if peak.label in excluded:
            row.update(excluded[peak.label])
        elif not math.isfinite(peak.height) or peak.height < threshold:
            row["reason"] = "relative_or_absolute_height"
        else:
            frac = np.asarray((peak.x, peak.y, peak.z), dtype=float)
            if not bool(np.isfinite(frac).all()):
                raise ReconstructionBlocked("adaptive_growth_peak_nonfinite")
            position = (frac % 1.) @ index["matrix"].T
            nearest, contacts = None, []
            for image in index["tree"].query_ball_point(position, radius):
                atom = model.atoms[index["atom_indices"][image]]
                distance = float(np.linalg.norm(index["images"][image] - position))
                nearest = distance if nearest is None else min(nearest, distance)
                upper = policy.growth_metal_distance_angstrom if model.element_of(atom) in METALS else policy.growth_bond_distance_angstrom
                if minimum_distance <= distance <= upper:
                    contacts.append({"site_id": atom.label, "element": model.element_of(atom), "distance_angstrom": distance,
                                     "maximum_distance_angstrom": upper})
            row["nearest_scaffold_distance_angstrom"] = nearest
            row["scaffold_contacts"] = contacts
            if nearest is not None and nearest < minimum_distance:
                row["reason"] = "too_close_to_existing_scaffold"
            elif not contacts:
                row["reason"] = "no_bond_distance_contact_to_existing_scaffold"
            else:
                # Exact metric check between selected peaks includes periodic
                # neighbors; the raw peak coordinates themselves are unchanged.
                from crystalx_core.refinement.cycle import periodic_distance
                duplicate = any(periodic_distance(cell_values(model.header.cell),
                    (peak.x, peak.y, peak.z), operation.apply_to_xyz(other)) < minimum_distance
                    for other in selected_cartesian for operation in index["operations"])
                if duplicate:
                    row["reason"] = "too_close_to_selected_peak"
                elif len(selected) >= actual_limit:
                    row["reason"] = "per_round_peak_budget"
                else:
                    selected.append(peak.label)
                    selected_cartesian.append((peak.x, peak.y, peak.z))
                    row.update(decision="select", reason="height_and_bond_distance_supported")
        decisions.append(row)
    return tuple(selected), {"policy": policy.to_dict(), "round": round_number,
        "metal_ripple_exclusion_angstrom":metal_ripple_exclusion_angstrom,
        "relative_height_reference_scope":"after_metal_ripple_exclusion" if metal_ripple_exclusion_angstrom else "all_listed_peaks",
        "effective_peak_limit": actual_limit, "selected_peak_ids": tuple(selected), "decisions": decisions,
        "density_scope": "observed_partial_Q_list_not_complete_density_or_ligand_topology"}
