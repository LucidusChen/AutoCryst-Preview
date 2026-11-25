"""Contract tests use fake optional modules and fictional bytes, never weights.

The fabricated official-local reports below exercise validation paths only.
They are not real measured parity evidence or official checkpoint attestations.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import math
import subprocess
import sys

import pytest

import crystalx_core.app.reconstruction_inference as binding
from crystalx_core.app.reconstruction_inference import (
    ApplePreparationPolicy, create_apple_reconstruction_bridge, preprocessing_binding_sha256,
)
from crystalx_core.domain.inference import (
    ClassProbability, ElementInferenceRequest, ElementInferenceResult, ElementPrediction,
    InferenceDevice, ModelIdentity,
)
from crystalx_core.domain.symmetry import parse_symm
from crystalx_core.io.shelx_res import read_res
from crystalx_core.workflow.reconstruction import DensityObservation, DensityPeak, ReconstructionBlocked, read_snapshot


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_observation(observation, case):
    assert observation["input_sha256"] == digest(asdict(case.request))
    if observation["site_ids"] != list(case.request.site_ids):
        raise ValueError("order")
    for row in observation["probabilities"]:
        if any(not math.isfinite(x) for x in row) or not math.isclose(sum(row), 1):
            raise ValueError("distribution")


def compare_case(reference, native, case):
    for row in (reference, native):
        validate_observation(row, case)
    return {"case_id": case.case_id, "max_delta": max(abs(a - b) for left, right in zip(
        reference["probabilities"], native["probabilities"]) for a, b in zip(left, right))}


@pytest.fixture
def configured(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    checkpoint = root / "NOT_REAL_MODEL_BYTES.bin"
    checkpoint.write_bytes(b"Fictional checkpoint sentinel, never torch-loaded")
    source = root / "fictional-source"
    pins = {}
    for name in ("crystalx_infer/models/torchmd_net.py", "crystalx_infer/common/heavy.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic source sentinel: " + name.encode())
        pins[name] = file_sha(path)
    implementation = root / "fictional_preprocessing.py"
    implementation.write_bytes(b"synthetic preprocessing identity, never executed")
    policy = ApplePreparationPolicy((parse_symm("X,Y,Z"),), "hand-authored synthetic identity symmetry",
                                    1.0, (6, 7, 8), (("peak-1", 8),))
    request = ElementInferenceRequest(("neutral-site",), (6,), ((0., 0., 0.),), (6, 8))
    suite = [{"case_id": "fictional-probe", "request": asdict(request)}]
    observation = {"input_sha256": digest(asdict(request)), "site_ids": ["neutral-site"],
                   "probabilities": [[0.75, 0.25]]}
    common = {"protocol": "crystalx-local-parity-v1", "status": "MEASURED", "nonce": "a" * 32,
              "checkpoint_sha256": file_sha(checkpoint), "source_sha256": pins,
              "case_suite_sha256": digest(suite), "device_used": "cpu", "fell_back": False,
              "coverage_adjustment": False, "observations": [observation],
              "torch_threads": 1, "python_executable": "fictional-python", "python_version": "synthetic",
              "model_shape": {"hidden_channels": 16, "num_classes": 32, "num_layers": 1, "num_rbf": 4, "max_z": 32},
              "graph_implementation": "actual torch_cluster.radius_graph / original upstream Distance"}
    workers = {}
    for mode in ("upstream_cpu", "native_cpu"):
        deps = ("torch", "torch_geometric", "torch_scatter", "torch_cluster", "ase") if mode == "upstream_cpu" else ("torch",)
        workers[mode] = {**deepcopy(common), "mode": mode,
            "dependencies": {key: {"version": "synthetic-test-only", "file": "fictional"} for key in deps}}
    report = {"protocol": "crystalx-local-parity-v1", "status": "MEASURED", "blocked_reasons": [],
              "checkpoint_kind": "official_local", "checkpoint_sha256": file_sha(checkpoint),
              "checkpoint_provenance": "SYNTHETIC CONTRACT FIXTURE, NOT AN OFFICIAL ATTESTATION",
              "official_checkpoint_claim": "caller_attested_not_publisher_authenticated",
              "provenance_verified_by_publisher": False, "source_sha256": pins,
              "case_suite_sha256": digest(suite), "cases": suite, "workers": workers,
              "comparisons": {"cpu": [compare_case(observation, observation, SimpleNamespace(case_id="fictional-probe", request=request))]},
              "scientific_validation_performed": False, "probabilities_calibrated": False,
              "scientific_tolerance": None, "scientific_decision": "OWNER_REVIEW_REQUIRED"}
    report_path = root / "fabricated-contract-report.json"
    report_path.write_text(json.dumps(report))
    calls = {"backend_constructed": 0, "predictions": [], "prepared": [], "sources_checked": 0}
    parity = SimpleNamespace(PROTOCOL="crystalx-local-parity-v1", UPSTREAM_SHA256=pins,
        ParityCase=lambda case_id, request: SimpleNamespace(case_id=case_id, request=request),
        _validate_observation=validate_observation, _compare_case=compare_case)

    def verify_sources(path):
        calls["sources_checked"] += 1
        if any(file_sha(path / name) != sha for name, sha in pins.items()):
            raise ValueError("source changed")

    parity.verify_upstream_sources = verify_sources

    def prepare(parsed, **kwargs):
        calls["prepared"].append((parsed, kwargs))
        rows = [(atom.label, (atom.x, atom.y, atom.z), 6 if atom.sfac_index == 1 else 7, "atom")
                for atom in parsed.atoms]
        rows += [(p.label, (p.x, p.y, p.z), z, "peak") for p, z in zip(parsed.q_peaks, kwargs["q_peak_atomic_numbers"])]
        sites = [SimpleNamespace(site_id=f"engine/{i}/{label}", source_site_id=f"engine/{i}/{label}",
                                 source_index=i, kind=kind, label=label, fractional=frac, is_main=True)
                 for i, (label, frac, _, kind) in enumerate(rows)]
        context = SimpleNamespace(**vars(sites[0]))
        context.site_id += "/periodic-image"
        context.is_main = False
        context.fractional = tuple(value + (1 if i == 0 else 0) for i, value in enumerate(sites[0].fractional))
        sites.append(context)
        numbers = tuple(row[2] for row in rows) + (rows[0][2],)
        positions = tuple(tuple(10 * v for v in site.fractional) for site in sites)
        return SimpleNamespace(source_sha256=parsed.sha256, sites=tuple(sites),
            request=ElementInferenceRequest(tuple(site.site_id for site in sites), numbers, positions,
                                            kwargs["candidate_atomic_numbers"], len(rows)))

    preprocessing = SimpleNamespace(__file__=str(implementation), prepare_initial_solution=prepare,
        CompleteSymmetry=lambda operators, source: SimpleNamespace(operators=operators, source=source))

    class FakeBackend:
        def __init__(self, source_root, checkpoint_path, *, expected_sha256):
            calls["backend_constructed"] += 1

        def predict_elements(self, request, *, device, allow_cpu_fallback):
            calls["predictions"].append((request, device, allow_cpu_fallback))
            count = request.main_site_count or len(request.site_ids)
            return ElementInferenceResult(
                tuple(ElementPrediction(key, (ClassProbability(8, .75), ClassProbability(6, .25))) for key in request.site_ids[:count]),
                ModelIdentity(file_sha(checkpoint), (("torchmd_net.py", pins["crystalx_infer/models/torchmd_net.py"]),), 16, 32, 1, 4),
                device, device, False, "", "synthetic test graph", 0, "synthetic probabilities", (), 0,
                digest(asdict(request)), tuple(sorted(request.candidate_atomic_numbers)), "synthetic-test-only")

    modules = {"crystalx_engine_apple_silicon.parity": parity,
               "crystalx_engine_apple_silicon.preprocessing": preprocessing,
               "crystalx_engine_apple_silicon.backend": SimpleNamespace(AppleSiliconBackend=FakeBackend)}
    original_import = binding.importlib.import_module
    monkeypatch.setattr(binding.importlib, "import_module", lambda name: modules.get(name) or original_import(name))
    model_path = root / "hand-authored-model.res"
    model_path.write_text("TITL synthetic\nCELL 0.7 10 10 10 90 90 90\nLATT -1\nSFAC C N O\nUNIT 1 1 0\nFVAR 0.5\nC1 1 0.1 0.2 0.3 11 0.05\nN1 2 0.4 0.5 0.6 11 0.06\nHKLF 4\nEND\n")
    parsed = read_res(model_path)
    model = read_snapshot(model_path, reflection_sha256="f" * 64)
    observation = DensityObservation(model.sha256, model.basis_key, "b" * 64, "Q_PEAK_LIST",
                                     (DensityPeak("peak-1", (.3, .4, .5), 1.0),))
    arguments = dict(source_root=source, source_pins=dict(pins), checkpoint=checkpoint,
        expected_checkpoint_sha256=file_sha(checkpoint), official_provenance_reference=report["checkpoint_provenance"],
        parity_report=report_path, expected_parity_report_sha256=file_sha(report_path),
        preprocessing_policy=policy, expected_preprocessing_policy_sha256=preprocessing_binding_sha256(policy, file_sha(implementation)),
        expected_preprocessing_implementation_sha256=file_sha(implementation),
        parsed_model_provider=lambda snapshot: parsed, device=InferenceDevice.CPU)
    return SimpleNamespace(arguments=arguments, calls=calls, report=report, report_path=report_path,
        parsed=parsed, model=model, observation=observation, preprocessing=preprocessing, modules=modules)


def rewrite_report(configured, edit):
    report = deepcopy(configured.report)
    edit(report)
    configured.report_path.write_text(json.dumps(report))
    configured.arguments["expected_parity_report_sha256"] = file_sha(configured.report_path)


def test_core_import_does_not_import_optional_engine_or_torch():
    code = """import sys
class BlockOptional:
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'torch' or fullname.startswith('torch.') or fullname.startswith('crystalx_engine_apple_silicon'):
   raise RuntimeError('unexpected optional import')
sys.meta_path.insert(0, BlockOptional())
import crystalx_core.app.reconstruction_inference
print('core-only import passed')
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_explicit_factory_loads_only_after_all_pins_and_recomputed_measurements(configured):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    assert configured.calls["backend_constructed"] == 1
    assert configured.calls["predictions"] == []
    assert configured.calls["prepared"] == []
    assert configured.calls["sources_checked"] == 1
    assert bridge.audit.blockers(synthetic=False) == ()
    assert not bridge.scientific_tolerance_approved
    assert not bridge.publisher_provenance_authenticated
    assert not bridge.full_density_map_claimed
    assert not bridge.upstream_preprocessing_parity_verified


@pytest.mark.parametrize("field", ["expected_checkpoint_sha256", "expected_parity_report_sha256",
                                  "expected_preprocessing_policy_sha256", "expected_preprocessing_implementation_sha256"])
def test_wrong_or_missing_pins_prevent_backend_construction(configured, field):
    configured.arguments[field] = "0" * 64
    with pytest.raises(ReconstructionBlocked):
        create_apple_reconstruction_bridge(**configured.arguments)
    assert not configured.calls["backend_constructed"]


@pytest.mark.parametrize("field", ["checkpoint", "parity_report"])
def test_missing_local_files_prevent_backend_load(configured, field):
    configured.arguments[field] = Path(str(configured.arguments[field]) + ".absent")
    with pytest.raises(ReconstructionBlocked):
        create_apple_reconstruction_bridge(**configured.arguments)
    assert not configured.calls["backend_constructed"]


@pytest.mark.parametrize("edit", [
    lambda p: p.update(status="BLOCKED"),
    lambda p: p.update(checkpoint_kind="synthetic_fixture"),
    lambda p: p.update(checkpoint_sha256="0" * 64),
    lambda p: p.update(comparisons={}),
    lambda p: p.update(blocked_reasons=["upstream absent"]),
    lambda p: p.update(scientific_validation_performed=True),
    lambda p: p.update(probabilities_calibrated=True),
    lambda p: p["comparisons"]["cpu"][0].update(max_delta=0.99),
    lambda p: p["workers"]["upstream_cpu"].update(graph_implementation="native replacement"),
    lambda p: p["workers"]["upstream_cpu"].update(dependencies={"torch": {}}),
    lambda p: p["workers"]["native_cpu"].update(fell_back=True),
    lambda p: p["workers"]["native_cpu"].update(nonce="b" * 32),
    lambda p: p["workers"]["native_cpu"].update(source_sha256={}),
    lambda p: p["workers"]["native_cpu"]["observations"][0].update(site_ids=["wrong-site"]),
    lambda p: p["workers"]["native_cpu"]["observations"][0].update(probabilities=[[.9, .9]]),
])
def test_unmeasured_synthetic_inconsistent_or_unsupported_claims_block_before_load(configured, edit):
    rewrite_report(configured, edit)
    with pytest.raises(ReconstructionBlocked):
        create_apple_reconstruction_bridge(**configured.arguments)
    assert configured.calls["backend_constructed"] == 0


def test_arbitrary_parity_true_without_worker_evidence_is_not_accepted(configured):
    rewrite_report(configured, lambda p: p.update(upstream_prediction_parity_verified=True, workers={}))
    with pytest.raises(ReconstructionBlocked):
        create_apple_reconstruction_bridge(**configured.arguments)
    assert configured.calls["backend_constructed"] == 0


def test_preparer_consumes_observed_coordinates_maps_stable_ids_and_preserves_descriptors(configured):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    prepared = bridge.preparer.prepare(configured.model, configured.observation)
    request = prepared.request
    assert request.site_ids[:3] == ("C1", "N1", "peak-1")
    assert request.main_site_count == 3
    assert request.atomic_numbers[:3] == (6, 7, 8)
    assert request.site_ids[3].startswith("C1/context/")
    assert request.positions_angstrom[2] == (3., 4., 5.)
    assert prepared.source_model_sha256 == configured.model.sha256
    assert prepared.preprocessing_policy_sha256 == bridge.audit.preprocessing_policy_sha256
    parsed, kwargs = configured.calls["prepared"][0]
    assert parsed.atoms == configured.parsed.atoms
    assert parsed.q_peaks[0].height == configured.observation.peaks[0].height
    assert (parsed.q_peaks[0].x, parsed.q_peaks[0].y, parsed.q_peaks[0].z) == configured.observation.peaks[0].fractional
    assert kwargs["q_peak_atomic_numbers"] == (8,)
    assert configured.calls["predictions"] == []
    result = bridge.inference.predict_elements(request, allow_cpu_fallback=False)
    assert tuple(p.site_id for p in result.predictions) == ("C1", "N1", "peak-1")
    assert configured.calls["predictions"][0][1:] == (InferenceDevice.CPU, False)


@pytest.mark.parametrize("change", [
    lambda c: replace(c.model, sfac=("O",)),
    lambda c: replace(c.model, sites=tuple(reversed(c.model.sites))),
    lambda c: replace(c.model, sites=(replace(c.model.sites[0], fractional=(.9, .8, .7)),) + c.model.sites[1:]),
])
def test_parsed_model_cannot_be_replaced_by_stale_or_forged_snapshot_fields(configured, change):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    model = change(configured)
    observation = replace(configured.observation, model_sha256=model.sha256)
    with pytest.raises(ReconstructionBlocked):
        bridge.preparer.prepare(model, observation)
    assert configured.calls["prepared"] == []


def test_missing_descriptor_or_mislabelled_density_blocks_without_guessing(configured):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    with pytest.raises(ReconstructionBlocked):
        bridge.preparer.prepare(configured.model, replace(configured.observation,
            peaks=(replace(configured.observation.peaks[0], peak_id="unconfigured-peak"),)))
    with pytest.raises(ReconstructionBlocked):
        bridge.preparer.prepare(configured.model, replace(configured.observation, coverage="full"))
    assert not configured.calls["prepared"]


def test_peak_and_model_id_collision_blocks(configured):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    with pytest.raises(ReconstructionBlocked):
        bridge.preparer.prepare(configured.model, replace(configured.observation,
            peaks=(replace(configured.observation.peaks[0], peak_id="C1"),)))


def test_optional_preparer_must_not_invent_main_coordinates(configured):
    original = configured.preprocessing.prepare_initial_solution

    def invent(*args, **kwargs):
        result = original(*args, **kwargs)
        result.sites[0].fractional = (.99, .99, .99)
        return result

    configured.preprocessing.prepare_initial_solution = invent
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    with pytest.raises(ReconstructionBlocked):
        bridge.preparer.prepare(configured.model, configured.observation)


def test_inference_rechecks_artifact_pins_and_never_falls_back(configured):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    prepared = bridge.preparer.prepare(configured.model, configured.observation)
    with pytest.raises(ReconstructionBlocked):
        bridge.inference.predict_elements(prepared.request, allow_cpu_fallback=True)
    configured.arguments["checkpoint"].write_bytes(b"changed sentinel")
    with pytest.raises(ReconstructionBlocked):
        bridge.inference.predict_elements(prepared.request, allow_cpu_fallback=False)
    assert not configured.calls["predictions"]


def test_source_path_escape_and_unavailable_selected_device_block(configured):
    arguments = {**configured.arguments, "source_pins": {"../escape.py": "a" * 64}}
    with pytest.raises(ReconstructionBlocked):
        create_apple_reconstruction_bridge(**arguments)
    with pytest.raises(ReconstructionBlocked):
        create_apple_reconstruction_bridge(**{**configured.arguments, "device": InferenceDevice.MPS})
    assert not configured.calls["backend_constructed"]


def test_arbitrary_request_cannot_bypass_explicit_preprocessing(configured):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    request = ElementInferenceRequest(("injected",), (6,), ((1., 2., 3.),), (6, 8))
    with pytest.raises(ReconstructionBlocked, match="audited_preparer"):
        bridge.inference.predict_elements(request)
    assert not configured.calls["predictions"]


def test_preprocessing_implementation_pin_is_rechecked_each_preparation(configured):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    Path(configured.preprocessing.__file__).write_bytes(b"modified synthetic implementation sentinel")
    with pytest.raises(ReconstructionBlocked):
        bridge.preparer.prepare(configured.model, configured.observation)
    assert not configured.calls["prepared"]


@pytest.mark.parametrize("change", [
    lambda result: replace(result, fell_back=True),
    lambda result: replace(result, model_identity=replace(result.model_identity, weights_sha256="0" * 64)),
    lambda result: replace(result, model_identity=replace(result.model_identity, source_files=())),
    lambda result: replace(result, predictions=tuple(reversed(result.predictions))),
    lambda result: replace(result, scientific_validation_performed=True),
    lambda result: replace(result, input_sha256="0" * 64),
])
def test_backend_result_cannot_overstate_scope_or_change_bound_identity(configured, change):
    bridge = create_apple_reconstruction_bridge(**configured.arguments)
    prepared = bridge.preparer.prepare(configured.model, configured.observation)
    original = bridge.inference._backend.predict_elements
    bridge.inference._backend.predict_elements = lambda *args, **kwargs: change(original(*args, **kwargs))
    with pytest.raises(ReconstructionBlocked):
        bridge.inference.predict_elements(prepared.request, allow_cpu_fallback=False)


def test_missing_optional_package_is_blocked_before_backend_load(configured, monkeypatch):
    def absent(name):
        raise ImportError("optional package deliberately unavailable")

    monkeypatch.setattr(binding.importlib, "import_module", absent)
    with pytest.raises(ReconstructionBlocked, match="package_missing"):
        create_apple_reconstruction_bridge(**configured.arguments)
    assert not configured.calls["backend_constructed"]
