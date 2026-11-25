"""Separate, fail-closed scientific grading of sealed private outputs.

There is deliberately no built-in geometry matcher. The existing optional
geometry helpers do not establish equivalence across setting, origin, symmetry
and site permutations. An installed library or provider's self-description is
not an audit. A caller must supply a separately approved implementation identity
and scientific tolerance approval before this module opens held-out answers.
Even valid matching evidence requires explicit human scientific review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
import re
from typing import Protocol

from .sealing import (RunSeal, SealError, read_private_reference,
                      read_sealed_output, verify_seal, write_private_json, _canonical_track)


class GradingStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    BLOCKED = "BLOCKED"
    EVIDENCE_READY_REVIEW_REQUIRED = "EVIDENCE_READY_REVIEW_REQUIRED"


REQUIRED_MATCHING_CAPABILITIES = frozenset({
    "periodic_boundaries", "origin_shifts", "symmetry_equivalents",
    "cell_setting_transforms", "site_permutations", "occupancy_disorder",
})


@dataclass(frozen=True)
class ProviderIdentity:
    name: str
    version: str
    implementation_sha256: str


@dataclass(frozen=True)
class ProviderAudit:
    """Trusted caller approval; never inferred from a provider's own claim."""

    identity: ProviderIdentity
    audit_reference: str
    capabilities: frozenset[str]


@dataclass(frozen=True)
class ScientificThresholds:
    site_match_tolerance_angstrom: float
    heavy_atomic_number_min: int = 20
    owner_approved: bool = False
    approval_reference: str = ""

    def __post_init__(self) -> None:
        value = self.site_match_tolerance_angstrom
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Site-matching tolerance must be finite and positive")
        if type(self.heavy_atomic_number_min) is not int or not 1 <= self.heavy_atomic_number_min <= 118:
            raise ValueError("Heavy-site atomic-number cutoff must be in [1, 118]")
        if type(self.owner_approved) is not bool or not isinstance(self.approval_reference, str):
            raise ValueError("Threshold approval must be explicit")


@dataclass(frozen=True)
class GradingRequest:
    case_id: str
    track: str
    case_root: Path
    answer_path: Path | None = None
    reference_paths: tuple[Path, ...] = ()
    model_role: str = "model"

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("Grading requires an explicit local case identifier")
        object.__setattr__(self, "track", _canonical_track(self.track))
        if self.track == "unsolved" and self.answer_path is not None:
            raise ValueError("An unsolved track has no held-out ground-truth answer")
        object.__setattr__(self, "case_root", Path(self.case_root))
        object.__setattr__(self, "reference_paths", tuple(Path(p) for p in self.reference_paths))
        if self.answer_path is not None:
            object.__setattr__(self, "answer_path", Path(self.answer_path))


@dataclass(frozen=True)
class StructureSnapshot:
    """Bounded immutable bytes; a provider receives no arbitrary file access API."""

    content: bytes
    sha256: str
    format_hint: str


@dataclass(frozen=True)
class MatchingInput:
    produced: StructureSnapshot
    answer: StructureSnapshot
    advisory_references: tuple[StructureSnapshot, ...]
    track: str


@dataclass(frozen=True)
class SiteMatch:
    produced_site_id: str
    reference_site_id: str
    distance_angstrom: float
    produced_element: str | None = None
    reference_element: str | None = None
    occupancy_status: str = "unknown"


@dataclass(frozen=True)
class MatchingEvidence:
    """Provider output, including full partitions so counts can be checked.

    Sites must be stable identities after the provider's crystallographic
    equivalence expansion. Disorder/unknown occupancy remains explicit; atom
    labels alone must never be used as the geometric correspondence criterion.
    """

    reference_site_ids: tuple[str, ...]
    produced_site_ids: tuple[str, ...]
    heavy_reference_site_ids: tuple[str, ...]
    heavy_produced_site_ids: tuple[str, ...]
    matches: tuple[SiteMatch, ...]
    missing_reference_sites: tuple[str, ...]
    extra_produced_sites: tuple[str, ...]
    setting_transform: str
    origin_shift_fractional: tuple[float, float, float]
    symmetry_context: str
    element_assumptions: tuple[str, ...] = ()
    coordination_review: tuple[str, ...] = ()
    chirality_review: str = "NOT_REVIEWED"
    occupancy_disorder_notes: tuple[str, ...] = ()
    refinement_diagnostics: tuple[str, ...] = ()
    r_metrics_comparable: bool = False
    r_comparison_basis: str = ""
    warnings: tuple[str, ...] = ()


class MatchingProvider(Protocol):
    """Implement only in an optional, separately audited crystallographic package."""

    identity: ProviderIdentity

    def compare(self, request: MatchingInput, *,
                thresholds: ScientificThresholds) -> MatchingEvidence: ...


@dataclass(frozen=True)
class GradingReport:
    status: GradingStatus
    case_id: str
    track: str
    seal_sha256: str | None
    reasons: tuple[str, ...]
    provider_identity: ProviderIdentity | None = None
    provider_audit_reference: str = ""
    answer_sha256: str | None = None
    reference_sha256: tuple[str, ...] = ()
    evidence: MatchingEvidence | None = None
    heavy_site_recovered: int | None = None
    heavy_site_missing: int | None = None
    heavy_site_extra: int | None = None
    heavy_site_recovery_fraction: float | None = None
    thresholds_approval_reference: str = ""
    requires_scientific_review: bool = True
    scientifically_reviewed: bool = False
    blind_initial_solution_success_claimed: bool = False
    independent_element_identification_claimed: bool = False

    def to_dict(self) -> dict[str, object]:
        # Output contains private scientific evidence: it belongs only in the
        # designated ignored run directory, never public records or fixtures.
        result = asdict(self)
        result["status"] = self.status.value
        return result


def _approved_provider(provider: MatchingProvider | None,
                       audit: ProviderAudit | None) -> str | None:
    if provider is None:
        return "No audited crystallographic matching provider is configured"
    if audit is None or not isinstance(audit, ProviderAudit):
        return "Provider approval is missing; self-declared capabilities are insufficient"
    if (not isinstance(audit.identity, ProviderIdentity)
            or not isinstance(audit.audit_reference, str) or not audit.audit_reference.strip()):
        return "Provider approval lacks an independent audit reference"
    identity = audit.identity
    if (not isinstance(identity.name, str) or not identity.name.strip()
            or not isinstance(identity.version, str) or not identity.version.strip()
            or not isinstance(identity.implementation_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity.implementation_sha256)):
        return "Audited provider identity is incomplete"
    if getattr(provider, "identity", None) != identity:
        return "Installed provider identity differs from the independently approved implementation"
    if (not isinstance(audit.capabilities, frozenset)
            or not REQUIRED_MATCHING_CAPABILITIES.issubset(audit.capabilities)):
        return "Provider audit does not cover periodicity, origin, symmetry, setting, permutations and disorder"
    return None


def _site_set(values: tuple[str, ...], field: str) -> set[str]:
    if (not isinstance(values, tuple) or any(not isinstance(v, str) or not v.strip() for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"Provider returned invalid {field}")
    return set(values)


def _validate_evidence(evidence: MatchingEvidence, thresholds: ScientificThresholds) -> None:
    if not isinstance(evidence, MatchingEvidence):
        raise ValueError("Provider returned an unsupported evidence object")
    reference = _site_set(evidence.reference_site_ids, "reference sites")
    produced = _site_set(evidence.produced_site_ids, "produced sites")
    heavy_reference = _site_set(evidence.heavy_reference_site_ids, "heavy reference sites")
    heavy_produced = _site_set(evidence.heavy_produced_site_ids, "heavy produced sites")
    missing = _site_set(evidence.missing_reference_sites, "missing sites")
    extra = _site_set(evidence.extra_produced_sites, "extra sites")
    if not heavy_reference.issubset(reference) or not heavy_produced.issubset(produced):
        raise ValueError("Heavy-site identities are not a subset of declared sites")
    matched_ref: set[str] = set()
    matched_prod: set[str] = set()
    if not isinstance(evidence.matches, tuple):
        raise ValueError("Provider matches must be an immutable sequence")
    for item in evidence.matches:
        if (not isinstance(item, SiteMatch) or item.reference_site_id not in reference
                or item.produced_site_id not in produced or item.reference_site_id in matched_ref
                or item.produced_site_id in matched_prod):
            raise ValueError("Provider site matching is not a valid one-to-one correspondence")
        distance = item.distance_angstrom
        if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                or not math.isfinite(distance) or distance < 0
                or distance > thresholds.site_match_tolerance_angstrom):
            raise ValueError("Provider correspondence exceeds the approved geometric tolerance")
        for element in (item.produced_element, item.reference_element):
            if element is not None and (not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element)):
                raise ValueError("Provider element assumptions must be explicit symbols or unknown")
        if item.occupancy_status not in {"unknown", "compatible", "disordered", "incompatible"}:
            raise ValueError("Provider must retain explicit occupancy/disorder status")
        matched_ref.add(item.reference_site_id)
        matched_prod.add(item.produced_site_id)
    if missing != reference - matched_ref or extra != produced - matched_prod:
        raise ValueError("Provider missing/extra partitions do not match its correspondence")
    if (not isinstance(evidence.setting_transform, str) or not evidence.setting_transform.strip()
            or not isinstance(evidence.symmetry_context, str) or not evidence.symmetry_context.strip()
            or not isinstance(evidence.origin_shift_fractional, tuple)
            or len(evidence.origin_shift_fractional) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in evidence.origin_shift_fractional)):
        raise ValueError("Provider must retain setting, origin and symmetry evidence")
    for field in (evidence.element_assumptions, evidence.coordination_review,
                  evidence.occupancy_disorder_notes, evidence.refinement_diagnostics, evidence.warnings):
        if not isinstance(field, tuple) or any(not isinstance(s, str) for s in field):
            raise ValueError("Provider diagnostic evidence must contain explicit text")
    if not isinstance(evidence.chirality_review, str) or not evidence.chirality_review.strip():
        raise ValueError("Provider must retain explicit chirality review status")
    if (type(evidence.r_metrics_comparable) is not bool
            or not isinstance(evidence.r_comparison_basis, str)
            or evidence.r_metrics_comparable and not evidence.r_comparison_basis.strip()):
        raise ValueError("R metrics require compatible reflection/cutoff/model evidence")


def grade_sealed_run(run_root: Path | str, *, private_root: Path | str,
                     specification: GradingRequest,
                     provider: MatchingProvider | None = None,
                     approved_provider: ProviderAudit | None = None,
                     thresholds: ScientificThresholds | None = None,
                     expected_seal_sha256: str | None = None) -> GradingReport:
    """Verify outputs before any answer read; return honest unevaluated states.

    This does not write a report automatically. Use ``save_grading_report`` to
    persist only inside the private run. The caller must run this independently
    of the blind solver. Same-user Python code is not an operating-system jail.
    """
    case_id = specification.case_id
    track = specification.track

    def report(status: GradingStatus, reason: str, seal: RunSeal | None = None) -> GradingReport:
        return GradingReport(status, case_id, track, seal.seal_sha256 if seal else None, (reason,))

    try:
        seal = verify_seal(run_root, private_root=private_root,
                           expected_seal_sha256=expected_seal_sha256)
    except (SealError, OSError):
        # Do not include paths or parsed private exception text in public callers.
        return report(GradingStatus.BLOCKED, "Run output seal is missing, unsafe or fails integrity verification")
    if expected_seal_sha256 is None:
        return report(GradingStatus.NOT_EVALUATED,
                      "An independently retained seal digest is required before held-out answer access", seal)
    if seal.track != track:
        return report(GradingStatus.BLOCKED, "Grading track differs from the sealed run track", seal)
    if seal.case_id != case_id:
        return report(GradingStatus.BLOCKED, "Grading case differs from the sealed run case", seal)
    if track == "unsolved":
        return report(GradingStatus.NOT_EVALUATED,
                      "Unsolved discovery has no ground truth; references cannot establish exact-answer success", seal)
    if specification.answer_path is None:
        return report(GradingStatus.NOT_EVALUATED, "No held-out answer is configured", seal)
    problem = _approved_provider(provider, approved_provider)
    if problem:
        return report(GradingStatus.NOT_EVALUATED, problem, seal)
    if (not isinstance(thresholds, ScientificThresholds) or not thresholds.owner_approved
            or not thresholds.approval_reference.strip()):
        return report(GradingStatus.NOT_EVALUATED,
                      "Scientific matching tolerance requires explicit owner approval with provenance", seal)
    if seal.terminal_state == "blocked":
        return report(GradingStatus.BLOCKED, "The sealed run was blocked before completing its intended stages", seal)
    assert provider is not None and approved_provider is not None
    try:
        model = seal.file_for_role(specification.model_role)
        # Read model first. A corrupt/missing model must never trigger answer I/O.
        model_bytes = read_sealed_output(seal, private_root=private_root,
                                         role=specification.model_role)
        answer_bytes, answer_digest = read_private_reference(
            specification.answer_path, case_root=specification.case_root)
        references = []
        for reference in specification.reference_paths:
            content, digest = read_private_reference(reference, case_root=specification.case_root)
            references.append(StructureSnapshot(content, digest, reference.suffix.lower()))
        # Recheck immediately before provider access in case files changed during
        # answer loading. No answer is passed if this verification fails.
        verify_seal(run_root, private_root=private_root,
                    expected_seal_sha256=expected_seal_sha256)
    except (SealError, OSError):
        return report(GradingStatus.BLOCKED,
                      "Sealed model or explicit private reference cannot be safely verified/read", seal)
    request = MatchingInput(
        StructureSnapshot(model_bytes, hashlib.sha256(model_bytes).hexdigest(), Path(model.path).suffix.lower()),
        StructureSnapshot(answer_bytes, answer_digest, specification.answer_path.suffix.lower()),
        tuple(references), track)
    try:
        evidence = provider.compare(request, thresholds=thresholds)
        _validate_evidence(evidence, thresholds)
        verify_seal(run_root, private_root=private_root,
                    expected_seal_sha256=expected_seal_sha256)
    except Exception:
        # Third-party exceptions can embed answer contents or coordinates; keep
        # exception text out of returned summary/logging surfaces.
        return report(GradingStatus.BLOCKED,
                      "Approved matching provider failed or returned inconsistent evidence", seal)
    matched_reference = {m.reference_site_id for m in evidence.matches}
    recovered = len(matched_reference.intersection(evidence.heavy_reference_site_ids))
    heavy_missing = len(set(evidence.heavy_reference_site_ids) - matched_reference)
    heavy_extra = len(set(evidence.heavy_produced_site_ids).intersection(evidence.extra_produced_sites))
    reasons = ["Matching evidence requires explicit coordination, chirality and disorder review"]
    if track == "component_regression":
        reasons.append("Expert-model component regression is not independent initial structure solution")
    if not evidence.r_metrics_comparable:
        reasons.append("R values are not compared without compatible reflections, cutoffs and refinement settings")
    return GradingReport(
        GradingStatus.EVIDENCE_READY_REVIEW_REQUIRED, case_id, track, seal.seal_sha256, tuple(reasons),
        approved_provider.identity, approved_provider.audit_reference, answer_digest,
        tuple(r.sha256 for r in references), evidence, recovered, heavy_missing, heavy_extra,
        recovered / len(evidence.heavy_reference_site_ids) if evidence.heavy_reference_site_ids else None,
        thresholds.approval_reference)


def save_grading_report(report: GradingReport, run_root: Path | str, *,
                        private_root: Path | str, relative: str = "grading.json") -> Path:
    """Persist private scientific evidence exclusively; existing reports survive."""
    # Recheck seal whenever a report can contain answer-derived information.
    if report.evidence is not None:
        if report.seal_sha256 is None:
            raise SealError("Answer-derived evidence requires its sealed-run identity")
        verify_seal(run_root, private_root=private_root,
                    expected_seal_sha256=report.seal_sha256)
    return write_private_json(run_root, private_root=private_root,
                              relative=relative, payload=report.to_dict())
