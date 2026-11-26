"""Synthetic mechanics only. No official weights or actual upstream parity pass."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from crystalx_core.domain.inference import ElementInferenceRequest
from crystalx_engine_apple_silicon.backend import AppleSiliconBackend
from crystalx_engine_apple_silicon.parity import (
    PROTOCOL, UPSTREAM_SHA256, ParityCase, _compare_case, _digest, _run_worker,
    _validate_observation, fixed_cases, measure_parity, verify_upstream_sources,
)
from crystalx_engine_apple_silicon.source import SourceRevisionError, load_model_classes


SOURCE = Path(__file__).resolve().parents[2] / "crystalx_engine" / "inference_repo"


@pytest.fixture
def synthetic_weights(tmp_path, source_root):
    with torch.random.fork_rng():
        torch.manual_seed(812)
        classes = load_model_classes(source_root)
        model = classes.TorchMD_Net(
            classes.TorchMD_ET(hidden_channels=16, num_layers=1, num_rbf=4),
            classes.EquivariantScalar(16, num_classes=32),
        )
    checkpoint = tmp_path / "synthetic-only.pth"
    torch.save(model.state_dict(), checkpoint)
    return checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def _observation(case):
    count = case.request.main_site_count or len(case.request.site_ids)
    candidates = sorted(case.request.candidate_atomic_numbers)
    edges = [[i, i] for i in range(len(case.request.site_ids))]
    return {"case_id": case.case_id, "input_sha256": _digest(asdict(case.request)),
            "site_ids": list(case.request.site_ids[:count]),
            "candidate_atomic_numbers": candidates,
            "first_pass_labels_all_sites": [candidates[0]] * len(case.request.site_ids),
            "probabilities": [[.4, .3, .2, .1] for _ in range(count)],
            "graphs": [{"radius_edges": deepcopy(edges), "effective_edges": deepcopy(edges),
                        "eligible_neighbor_counts": [len(edges)] * len(edges)} for _ in range(2)]}


def test_fixed_dense_probe_exercises_real_cap_and_context():
    from crystalx_engine_apple_silicon.ops import radius_graph_with_diagnostics
    sparse, dense = fixed_cases()
    assert sparse == fixed_cases()[0]
    assert len(dense.request.site_ids) == 41
    assert dense.request.main_site_count == 21
    edges, diagnostic = radius_graph_with_diagnostics(torch.tensor(dense.request.positions_angstrom),
                                                     5., loop=True)
    assert diagnostic.truncated_nodes == 41
    assert diagnostic.max_neighbors_observed == 41
    assert edges.shape == (2, 41 * 32)


def test_missing_checkpoint_is_explicit_block_without_worker(source_root, monkeypatch):
    import crystalx_engine_apple_silicon.parity as module
    monkeypatch.setattr(module, "_run_worker", lambda *a, **k: pytest.fail("No weight may execute"))
    report = measure_parity(source_root, provenance="No authorised official local checkpoint yet")
    assert report["status"] == "BLOCKED"
    assert len(report["blocked_reasons"]) == 2
    assert report["comparisons"] == {}
    assert not report["trained_comparison_measured"]
    assert not report["upstream_prediction_parity_verified"]
    assert not report["scientific_validation_performed"]
    assert report["scientific_tolerance"] is None


@pytest.mark.parametrize("kwargs", [
    {"provenance": ""}, {"checkpoint_kind": "download"},
    {"expected_sha256": "bad"}, {"timeout_s": float("inf")}, {"timeout_s": True},
    {"devices": ("auto",)}, {"devices": ("cpu", "cpu")}, {"cases": ()},
    {"cases": (fixed_cases()[0], fixed_cases()[0])},
])
def test_invalid_run_contract_refused(kwargs):
    with pytest.raises((ValueError, TypeError)):
        measure_parity(SOURCE, **({"provenance": "synthetic fixture"} | kwargs))


def test_mismatched_checkpoint_digest_is_not_a_measurement(synthetic_weights):
    with pytest.raises(ValueError, match="SHA-256"):
        measure_parity(SOURCE, synthetic_weights[0], expected_sha256="0" * 64,
                       provenance="synthetic fixture", checkpoint_kind="synthetic_fixture")


@pytest.mark.parametrize("relative", tuple(UPSTREAM_SHA256))
def test_all_upstream_model_and_two_pass_sources_are_pinned(source_root, tmp_path, relative):
    for name in UPSTREAM_SHA256:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, destination)
    (tmp_path / relative).write_bytes((tmp_path / relative).read_bytes() + b"\nraise RuntimeError('must not execute')\n")
    with pytest.raises(SourceRevisionError):
        verify_upstream_sources(tmp_path)


def test_case_budget_refused():
    with pytest.raises(ValueError, match="128"):
        ParityCase("too-large", ElementInferenceRequest(tuple(map(str, range(129))), (6,) * 129,
                                                      tuple((float(i), 0., 0.) for i in range(129)), (6, 8)))


def test_site_top_runner_up_full_probability_and_edge_drift():
    case = fixed_cases()[0]
    reference = _observation(case)
    native = deepcopy(reference)
    native["probabilities"][0] = [.1, .2, .3, .4]
    native["graphs"][0]["radius_edges"][0] = [1, 0]
    native["graphs"][0]["effective_edges"][0] = [1, 0]
    measured = _compare_case(reference, native, case)
    assert measured["sites"][0]["reference_top"] == 6
    assert measured["sites"][0]["native_top"] == 26
    assert measured["sites"][0]["reference_runner_up"] == 7
    assert measured["sites"][0]["native_runner_up"] == 8
    assert measured["top_change_count"] == measured["runner_up_change_count"] == 1
    assert measured["max_absolute_probability_delta"] == pytest.approx(.3)
    assert measured["sites"][0]["l1_probability_delta"] == pytest.approx(.8)
    assert measured["graphs"][0]["radius_edges"]["removed"] == [(0, 0)]
    assert measured["graphs"][0]["radius_edges"]["added"] == [(1, 0)]
    assert measured["graphs"][0]["reference_truncated_sites"] == list(case.request.site_ids)


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(site_ids=["swapped"] * 4),
    lambda x: x.update(candidate_atomic_numbers=[26, 8, 7, 6]),
    lambda x: x.update(input_sha256="0" * 64),
    lambda x: x.update(first_pass_labels_all_sites=[99] * 4),
    lambda x: x["probabilities"][0].__setitem__(0, float("nan")),
    lambda x: x["probabilities"][0].__setitem__(0, .9),
    lambda x: x["probabilities"].pop(),
    lambda x: x["graphs"].pop(),
    lambda x: x["graphs"][0]["radius_edges"].append([0, 0]),
    lambda x: x["graphs"][0]["radius_edges"].__setitem__(0, [True, 0]),
    lambda x: x["graphs"][0]["effective_edges"].append([2, 0]),
    lambda x: x["graphs"][0].update(eligible_neighbor_counts=[1]),
])
def test_corrupt_or_misaligned_observations_refused(mutation):
    case = fixed_cases()[0]
    observed = _observation(case)
    mutation(observed)
    with pytest.raises(ValueError):
        _validate_observation(observed, case)


def test_real_isolated_native_worker_and_missing_actual_reference(synthetic_weights):
    checkpoint, digest = synthetic_weights
    report = measure_parity(SOURCE, checkpoint, expected_sha256=digest,
                            provenance="temporary test-generated random tensors, never trained",
                            checkpoint_kind="synthetic_fixture", devices=("cpu",))
    cpu = report["workers"]["native_cpu"]
    assert cpu["status"] == "MEASURED", cpu
    assert cpu["device_used"] == "cpu" and cpu["fell_back"] is False
    assert cpu["checkpoint_sha256"] == digest
    assert cpu["torch_threads"] == 1
    assert len(cpu["observations"]) == 2
    for observation, case in zip(cpu["observations"], fixed_cases()):
        _validate_observation(observation, case)
    engine = AppleSiliconBackend(SOURCE, checkpoint)
    actual = engine.predict_elements(fixed_cases()[0].request, device="cpu")
    for row, prediction in zip(cpu["observations"][0]["probabilities"], actual.predictions):
        assert {p.atomic_number: p.probability for p in prediction.topk} == pytest.approx(dict(zip((6, 7, 8, 26), row)), abs=1e-6)
    if any(importlib.util.find_spec(name) is None for name in ("torch_geometric", "torch_scatter", "torch_cluster")):
        assert report["status"] == "BLOCKED"
        assert report["workers"]["upstream_cpu"]["status"] == "BLOCKED"
        assert "dependencies unavailable" in report["workers"]["upstream_cpu"]["reason"]
        assert report["comparisons"] == {}
    # Even an environment with the real reference installed has only measured
    # synthetic mechanics; no scientific or trained acceptance can result.
    assert not report["trained_comparison_measured"]
    assert not report["upstream_prediction_parity_verified"]


@pytest.mark.skipif(any(importlib.util.find_spec(name) is None for name in
                        ("torch_geometric", "torch_scatter", "torch_cluster")),
                    reason="Actual upstream PyG/scatter/cluster dependencies are absent; never substitute native graph")
def test_actual_upstream_cpu_graph_with_synthetic_weights_when_available(synthetic_weights):
    report = measure_parity(SOURCE, synthetic_weights[0], expected_sha256=synthetic_weights[1],
                            provenance="temporary random fixture, not official weights",
                            checkpoint_kind="synthetic_fixture", devices=("cpu",))
    assert report["status"] == "MEASURED", report["blocked_reasons"]
    reference = report["workers"]["upstream_cpu"]
    assert "actual torch_cluster" in reference["graph_implementation"]
    assert all(name in reference["dependencies"] for name in ("torch_geometric", "torch_scatter", "torch_cluster"))
    assert len(report["comparisons"]["cpu"]) == 2
    assert not report["trained_comparison_measured"]
    assert report["scientific_tolerance"] is None


def test_reference_json_cannot_be_supplied_as_an_api():
    with pytest.raises(TypeError):
        measure_parity(SOURCE, provenance="fixture", reference_json={})


def test_complete_synthetic_measurement_never_promotes_scientific_flags(synthetic_weights, monkeypatch):
    import crystalx_engine_apple_silicon.parity as module
    def measured(payload, python, timeout):
        return {"status": "MEASURED", "checkpoint_sha256": synthetic_weights[1],
                "source_sha256": dict(UPSTREAM_SHA256), "case_suite_sha256": _digest(payload["cases"]),
                "device_used": "cpu" if payload["mode"] == "upstream_cpu" else payload["mode"][7:],
                "fell_back": False, "observations": [_observation(case) for case in fixed_cases()]}
    monkeypatch.setattr(module, "_run_worker", measured)
    report = measure_parity(SOURCE, synthetic_weights[0], expected_sha256=synthetic_weights[1],
                            provenance="mock worker protocol fixture", checkpoint_kind="synthetic_fixture")
    assert report["status"] == "MEASURED"
    assert set(report["comparisons"]) == {"cpu", "mps"}
    assert all(not report[name] for name in ("probabilities_calibrated", "scientific_validation_performed",
                                           "upstream_prediction_parity_verified", "trained_comparison_measured"))
    assert report["scientific_tolerance"] is None
    assert report["scientific_decision"] == "OWNER_REVIEW_REQUIRED"


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", "0" * 64), ("source_sha256", {}),
    ("case_suite_sha256", "0" * 64), ("device_used", "mps"), ("fell_back", True),
])
def test_worker_identity_or_fallback_cannot_be_used_as_reference(synthetic_weights, monkeypatch, field, value):
    import crystalx_engine_apple_silicon.parity as module
    def invalid(payload, python, timeout):
        result = {"status": "MEASURED", "checkpoint_sha256": synthetic_weights[1],
                  "source_sha256": dict(UPSTREAM_SHA256), "case_suite_sha256": _digest(payload["cases"]),
                  "device_used": "cpu", "fell_back": False,
                  "observations": [_observation(case) for case in fixed_cases()]}
        result[field] = value
        return result
    monkeypatch.setattr(module, "_run_worker", invalid)
    with pytest.raises(ValueError, match="identity"):
        measure_parity(SOURCE, synthetic_weights[0], expected_sha256=synthetic_weights[1],
                       provenance="identity rejection fixture", checkpoint_kind="synthetic_fixture", devices=("cpu",))


def test_timeout_is_explicit_block(synthetic_weights, monkeypatch):
    import crystalx_engine_apple_silicon.parity as module
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("isolated test worker", .1)
    monkeypatch.setattr(module, "_run_worker", timeout)
    report = measure_parity(SOURCE, synthetic_weights[0], expected_sha256=synthetic_weights[1],
                            provenance="test timeout", checkpoint_kind="synthetic_fixture", devices=("cpu",))
    assert report["status"] == "BLOCKED"
    assert len(report["blocked_reasons"]) == 2
    assert report["comparisons"] == {}


def test_wrong_worker_nonce_fails_closed(monkeypatch):
    import crystalx_engine_apple_silicon.parity as module
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, '{"protocol":"crystalx-local-parity-v1","nonce":"stale","mode":"upstream_cpu","status":"MEASURED"}', ""))
    with pytest.raises(ValueError, match="nonce"):
        _run_worker({"mode": "upstream_cpu", "nonce": "current"}, Path(sys.executable), 1.)
