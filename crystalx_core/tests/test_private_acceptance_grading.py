"""Synthetic adapter contracts; no real crystallographic match is asserted."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import crystalx_core.private_acceptance.grading as grading
from crystalx_core.private_acceptance.grading import (
    REQUIRED_MATCHING_CAPABILITIES, GradingRequest, GradingStatus, MatchingEvidence,
    ProviderAudit, ProviderIdentity, ScientificThresholds, SiteMatch,
    grade_sealed_run, save_grading_report,
)
from crystalx_core.private_acceptance.sealing import SealError, seal_run


IDENTITY = ProviderIdentity("synthetic-contract-test-only", "0", "a" * 64)
AUDIT = ProviderAudit(IDENTITY, "hand-authored synthetic adapter fixture", REQUIRED_MATCHING_CAPABILITIES)
THRESHOLDS = ScientificThresholds(0.25, owner_approved=True, approval_reference="synthetic test approval")


def synthetic_evidence():
    return MatchingEvidence(
        reference_site_ids=("r-heavy1", "r-heavy2", "r-light"),
        produced_site_ids=("p-heavy1", "p-heavy3", "p-light"),
        heavy_reference_site_ids=("r-heavy1", "r-heavy2"),
        heavy_produced_site_ids=("p-heavy1", "p-heavy3"),
        matches=(SiteMatch("p-heavy1", "r-heavy1", 0.1, "Zn", "Zn", "unknown"),
                 SiteMatch("p-light", "r-light", 0.2, "N", "O", "disordered")),
        missing_reference_sites=("r-heavy2",), extra_produced_sites=("p-heavy3",),
        setting_transform="synthetic fixture only: declared transform",
        origin_shift_fractional=(0.0, 0.5, 0.0), symmetry_context="synthetic test context",
        element_assumptions=("Owner element prior retained; no independent recognition claim",),
        occupancy_disorder_notes=("One site has unknown occupancy; another is disordered",),
    )


class SyntheticProvider:
    """A fixed response exercises contracts, never geometric matching."""

    identity = IDENTITY

    def __init__(self, evidence=None):
        self.evidence = evidence if evidence is not None else synthetic_evidence()
        self.requests = []

    def compare(self, request, *, thresholds):
        self.requests.append(request)
        return self.evidence


@pytest.fixture
def prepared(tmp_path):
    private = tmp_path.resolve() / "ignored-runs"
    run = private / "synthetic-run"
    run.mkdir(parents=True)
    (run / "model.res").write_bytes(b"synthetic model bytes")
    case = tmp_path.resolve() / "private-case"
    case.mkdir()
    (case / "expert.res").write_bytes(b"synthetic held-out bytes")
    (case / "advisory.cif").write_bytes(b"synthetic reference bytes")
    sealed = seal_run(run, private_root=private, outputs={"model": "model.res"}, case_id="synthetic-case", track="de_novo")
    request = GradingRequest("synthetic-case", "de_novo", case, case / "expert.res")
    return private, run, sealed, request


def grade(prepared, **kwargs):
    private, run, sealed, request = prepared
    parameters = {"specification": request, "provider": SyntheticProvider(),
                  "approved_provider": AUDIT, "thresholds": THRESHOLDS,
                  "expected_seal_sha256": sealed.seal_sha256}
    parameters.update(kwargs)
    return grade_sealed_run(run, private_root=private, **parameters)


def test_default_no_audited_provider_is_not_evaluated_without_answer_read(prepared, monkeypatch):
    private, run, sealed, request = prepared

    def trap(*args, **kwargs):
        pytest.fail("No answer may be opened before an approved matching configuration")

    monkeypatch.setattr(grading, "read_private_reference", trap)
    report = grade_sealed_run(run, private_root=private, specification=request,
                             expected_seal_sha256=sealed.seal_sha256)
    assert report.status is GradingStatus.NOT_EVALUATED
    assert report.answer_sha256 is None
    assert report.scientifically_reviewed is False


@pytest.mark.parametrize("kwargs", [
    {"expected_seal_sha256": None},
    {"provider": None}, {"approved_provider": None},
    {"approved_provider": replace(AUDIT, audit_reference="")},
    {"approved_provider": replace(AUDIT, capabilities=frozenset({"periodic_boundaries"}))},
    {"approved_provider": replace(AUDIT, identity=replace(IDENTITY, version="other"))},
    {"thresholds": None},
    {"thresholds": ScientificThresholds(0.25)},
    {"thresholds": ScientificThresholds(0.25, owner_approved=True)},
])
def test_answer_access_trap_for_all_unapproved_configurations(prepared, monkeypatch, kwargs):
    def trap(*args, **kw):
        pytest.fail("Blocked/unevaluated grading must not read the held-out answer")

    monkeypatch.setattr(grading, "read_private_reference", trap)
    result = grade(prepared, **kwargs)
    assert result.status is GradingStatus.NOT_EVALUATED
    assert result.answer_sha256 is None


@pytest.mark.parametrize("change", ["output", "seal", "track", "case", "missing_model_role"])
def test_seal_or_model_fault_never_opens_answers(prepared, monkeypatch, change):
    private, run, sealed, request = prepared
    kwargs = {}
    if change == "output":
        (run / "model.res").write_bytes(b"tampered")
    elif change == "seal":
        kwargs["expected_seal_sha256"] = "f" * 64
    elif change == "track":
        kwargs["specification"] = replace(request, track="component_regression")
    elif change == "case":
        kwargs["specification"] = replace(request, case_id="other-synthetic-case")
    else:
        kwargs["specification"] = replace(request, model_role="absent")

    def trap(*args, **kw):
        pytest.fail("Invalid sealed output must block before answer access")

    monkeypatch.setattr(grading, "read_private_reference", trap)
    assert grade(prepared, **kwargs).status is GradingStatus.BLOCKED


def test_validated_adapter_snapshot_and_metrics_are_review_required(prepared):
    private, run, sealed, request = prepared
    provider = SyntheticProvider()
    report = grade(prepared, provider=provider,
                   specification=replace(request, reference_paths=(request.case_root / "advisory.cif",)))
    assert report.status is GradingStatus.EVIDENCE_READY_REVIEW_REQUIRED
    assert report.heavy_site_recovered == 1
    assert report.heavy_site_missing == 1
    assert report.heavy_site_extra == 1
    assert report.heavy_site_recovery_fraction == 0.5
    assert report.requires_scientific_review
    assert not report.scientifically_reviewed
    assert not report.blind_initial_solution_success_claimed
    assert not report.independent_element_identification_claimed
    assert report.answer_sha256 == hashlib.sha256(b"synthetic held-out bytes").hexdigest()
    matching_input = provider.requests[0]
    assert matching_input.produced.content == b"synthetic model bytes"
    assert matching_input.answer.content == b"synthetic held-out bytes"
    assert matching_input.answer.format_hint == ".res"
    assert matching_input.advisory_references[0].format_hint == ".cif"
    assert not hasattr(matching_input, "answer_path")
    path = save_grading_report(report, run, private_root=private)
    assert path.parent == run
    data = json.loads(path.read_text())
    assert data["status"] == "EVIDENCE_READY_REVIEW_REQUIRED"
    assert data["evidence"]["occupancy_disorder_notes"]
    with pytest.raises(SealError):
        save_grading_report(report, run, private_root=private)


def test_uppercase_spec_track_and_empty_heavy_set_are_not_fabricated_scores(prepared):
    private, run, sealed, request = prepared
    provider = SyntheticProvider(replace(synthetic_evidence(), heavy_reference_site_ids=(),
                                          heavy_produced_site_ids=()))
    report = grade(prepared, provider=provider, specification=replace(request, track="DE_NOVO"))
    assert report.status is GradingStatus.EVIDENCE_READY_REVIEW_REQUIRED
    assert report.heavy_site_recovery_fraction is None
    assert report.heavy_site_recovered == 0


def test_unsolved_cannot_gain_exact_answer_success_from_references(prepared, monkeypatch):
    private, run, sealed, request = prepared
    # A separate run: never reclassify a sealed de novo run.
    unsolved = private / "unsolved-run"
    unsolved.mkdir()
    (unsolved / "model.res").write_text("synthetic exploratory model")
    output = seal_run(unsolved, private_root=private, outputs={"model": "model.res"}, case_id="synthetic-unknown", track="unsolved")
    spec = GradingRequest("synthetic-unknown", "unsolved", request.case_root,
                          reference_paths=(request.case_root / "advisory.cif",))

    def trap(*args, **kwargs):
        pytest.fail("Unsolved references must not be opened as an answer")

    monkeypatch.setattr(grading, "read_private_reference", trap)
    report = grade_sealed_run(unsolved, private_root=private, specification=spec,
                             expected_seal_sha256=output.seal_sha256, provider=SyntheticProvider(),
                             approved_provider=AUDIT, thresholds=THRESHOLDS)
    assert report.status is GradingStatus.NOT_EVALUATED
    assert report.answer_sha256 is None
    with pytest.raises(ValueError):
        replace(spec, answer_path=request.answer_path)


def test_component_regression_is_never_reported_as_blind_solution(prepared):
    private, run, sealed, request = prepared
    component = private / "component-run"
    component.mkdir()
    (component / "model.res").write_text("synthetic component result")
    output = seal_run(component, private_root=private, outputs={"model": "model.res"}, case_id="synthetic-case", track="component_regression")
    result = grade_sealed_run(component, private_root=private,
        specification=replace(request, track="component_regression"), provider=SyntheticProvider(),
        approved_provider=AUDIT, thresholds=THRESHOLDS, expected_seal_sha256=output.seal_sha256)
    assert result.status is GradingStatus.EVIDENCE_READY_REVIEW_REQUIRED
    assert any("not independent" in reason for reason in result.reasons)
    assert not result.blind_initial_solution_success_claimed


@pytest.mark.parametrize("changes", [
    {"missing_reference_sites": ()},
    {"extra_produced_sites": ()},
    {"heavy_reference_site_ids": ("invented-site",)},
    {"matches": (SiteMatch("p-heavy1", "r-heavy1", 0.1), SiteMatch("p-heavy1", "r-light", 0.1))},
    {"matches": (SiteMatch("p-heavy1", "r-heavy1", float("nan")),)},
    {"matches": (SiteMatch("p-heavy1", "r-heavy1", 99.0),)},
    {"matches": (SiteMatch("p-heavy1", "r-heavy1", -0.1),)},
    {"origin_shift_fractional": (0.0, float("inf"), 0.0)},
    {"symmetry_context": ""}, {"setting_transform": ""},
    {"r_metrics_comparable": True, "r_comparison_basis": ""},
])
def test_inconsistent_provider_evidence_blocks(prepared, changes):
    provider = SyntheticProvider(replace(synthetic_evidence(), **changes))
    report = grade(prepared, provider=provider)
    assert report.status is GradingStatus.BLOCKED
    assert report.evidence is None
    assert report.answer_sha256 is None


def test_provider_error_does_not_copy_answer_in_exception_to_report(prepared):
    class ExplodingProvider(SyntheticProvider):
        def compare(self, request, *, thresholds):
            raise ValueError("private coordinates must not leak through exception text")

    report = grade(prepared, provider=ExplodingProvider())
    assert report.status is GradingStatus.BLOCKED
    assert "private coordinates" not in json.dumps(report.to_dict())


def test_provider_modifying_output_is_detected(prepared):
    private, run, sealed, request = prepared

    class TamperingProvider(SyntheticProvider):
        def compare(self, request, *, thresholds):
            (run / "model.res").write_bytes(b"tampered during grading")
            return synthetic_evidence()

    report = grade(prepared, provider=TamperingProvider())
    assert report.status is GradingStatus.BLOCKED
    assert report.evidence is None


@pytest.mark.parametrize("path_kind", ["outside", "symlink"])
def test_answer_path_escape_blocks_before_provider(prepared, path_kind):
    private, run, sealed, request = prepared
    outside = request.case_root.parent / "outside.res"
    outside.write_bytes(b"outside-root answer bytes")
    if path_kind == "symlink":
        target = request.case_root / "linked.res"
        target.symlink_to(outside)
    else:
        target = outside
    provider = SyntheticProvider()
    report = grade(prepared, provider=provider, specification=replace(request, answer_path=target))
    assert report.status is GradingStatus.BLOCKED
    assert not provider.requests


def test_save_rechecks_integrity_and_never_writes_outside_private_root(prepared):
    private, run, sealed, request = prepared
    report = grade(prepared)
    with pytest.raises(SealError):
        save_grading_report(report, run, private_root=private, relative="../public.json")
    (run / "model.res").write_bytes(b"post-grade tamper")
    with pytest.raises(SealError):
        save_grading_report(report, run, private_root=private)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1, 0, True])
def test_thresholds_are_explicit_finite_positive(tolerance):
    with pytest.raises(ValueError):
        ScientificThresholds(tolerance)
