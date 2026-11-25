"""Explicit, optional Apple inference binding for reconstruction.

Importing this module imports no torch or optional engine. Only the factory may
load a local checkpoint, after independently supplied file pins and a measured
reference report have been checked. Saved reports are integrity-pinned evidence,
not signatures: their consistency does not authenticate a publisher, approve a
scientific tolerance, establish upstream preprocessing parity or calibrate AI.
Real reconstruction dispatch remains separately disabled by the application.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.inference import ElementInferenceRequest, ElementInferenceResult, InferenceDevice
from crystalx_core.domain.symmetry import SymmetryOperator
from crystalx_core.io.shelx_res import QPeak, ResFile
from crystalx_core.workflow.reconstruction import (
    DensityObservation, InferenceAudit, ModelSnapshot, PreparedInference, ReconstructionBlocked,
)


_PROTOCOL = "crystalx-reconstruction-apple-binding-v1"
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _blocked(reason: str) -> ReconstructionBlocked:
    return ReconstructionBlocked(reason)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _pin(value: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise _blocked("inference_binding_requires_explicit_lowercase_sha256_pins")
    return value


def _read_pinned(path: Path, expected: str, *, collect: bool = False,
                  max_bytes: int | None = None) -> bytes:
    _pin(expected)
    fd = None
    try:
        if Path(path).is_symlink():
            raise _blocked("inference_artifact_symlink_is_not_an_explicit_snapshot")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or (max_bytes is not None and before.st_size > max_bytes):
            raise _blocked("inference_artifact_is_not_a_bounded_regular_file")
        digest = hashlib.sha256()
        parts = []
        size = 0
        while True:
            content = os.read(fd, 1024 * 1024)
            if not content:
                break
            size += len(content)
            if max_bytes is not None and size > max_bytes:
                raise _blocked("inference_artifact_exceeds_its_read_bound")
            digest.update(content)
            if collect:
                parts.append(content)
        after = os.fstat(fd)
        retained = os.stat(path, follow_symlinks=False)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if (identity(before) != identity(after) or identity(before) != identity(retained)
                or not stat.S_ISREG(retained.st_mode) or size != before.st_size
                or digest.hexdigest() != expected):
            raise _blocked("inference_artifact_digest_or_snapshot_changed")
        return b"".join(parts)
    except OSError as exc:
        raise _blocked("required_local_inference_artifact_missing_or_unreadable") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _strict_report(data: bytes) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise _blocked("parity_report_has_duplicate_fields")
            value[key] = item
        return value

    def invalid(_):
        raise _blocked("parity_report_has_nonfinite_values")

    try:
        report = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise _blocked("parity_report_is_not_strict_json") from exc
    if not isinstance(report, dict):
        raise _blocked("parity_report_is_not_an_object")
    return report


@dataclass(frozen=True)
class ApplePreparationPolicy:
    symmetry_operators: tuple[SymmetryOperator, ...]
    symmetry_provenance_reference: str
    context_radius_angstrom: float
    candidate_atomic_numbers: tuple[int, ...]
    q_peak_atomic_numbers: tuple[tuple[str, int], ...] = ()
    max_sites: int = 20_000
    max_image_candidates: int = 1_000_000
    allow_anisotropic_metals: bool = False

    def __post_init__(self):
        if type(self.allow_anisotropic_metals) is not bool:
            raise _blocked("anisotropic_metal_policy_must_be_boolean")
        if (not isinstance(self.symmetry_operators, tuple) or not self.symmetry_operators
                or any(not isinstance(op, SymmetryOperator) for op in self.symmetry_operators)
                or not isinstance(self.symmetry_provenance_reference, str)
                or not self.symmetry_provenance_reference.strip()):
            raise _blocked("explicit_complete_symmetry_and_provenance_required")
        if (isinstance(self.context_radius_angstrom, bool)
                or not isinstance(self.context_radius_angstrom, (int, float))
                or not math.isfinite(self.context_radius_angstrom) or self.context_radius_angstrom <= 0):
            raise _blocked("explicit_finite_positive_context_radius_required")
        if (not isinstance(self.candidate_atomic_numbers, tuple) or len(self.candidate_atomic_numbers) < 2
                or len(set(self.candidate_atomic_numbers)) != len(self.candidate_atomic_numbers)
                or any(type(z) is not int or not 2 <= z <= 118 for z in self.candidate_atomic_numbers)):
            raise _blocked("explicit_distinct_heavy_candidate_descriptors_required")
        if (not isinstance(self.q_peak_atomic_numbers, tuple)
                or any(not isinstance(item, tuple) or len(item) != 2
                       or not isinstance(item[0], str) or not item[0]
                       or type(item[1]) is not int or not 1 <= item[1] <= 118
                       for item in self.q_peak_atomic_numbers)
                or len({item[0] for item in self.q_peak_atomic_numbers}) != len(self.q_peak_atomic_numbers)):
            raise _blocked("q_peak_computational_descriptors_must_be_explicit_per_identity")
        if any(type(v) is not int or v < 1 for v in (self.max_sites, self.max_image_candidates)):
            raise _blocked("preprocessing_resource_limits_must_be_positive")

    def payload(self) -> dict:
        return {
            "protocol": _PROTOCOL,
            "symmetry": [{"rotation": op.rotation, "translation": tuple(str(v) for v in op.translation)}
                         for op in self.symmetry_operators],
            "symmetry_provenance_reference": self.symmetry_provenance_reference,
            "context_radius_angstrom": self.context_radius_angstrom,
            "candidate_atomic_numbers": self.candidate_atomic_numbers,
            "q_peak_atomic_numbers": sorted(self.q_peak_atomic_numbers),
            "max_sites": self.max_sites, "max_image_candidates": self.max_image_candidates,
            "site_identity_policy": "original_model_order_then_observation_peak_order_context_follows",
            "density_scope": "observed_peak_list_only_no_full_map_claim",
            **({"anisotropic_metal_displacement":"cell_metric_Ueq_snapshot_raw_tensor_retained"}
               if self.allow_anisotropic_metals else {}),
        }


def preprocessing_binding_sha256(policy: ApplePreparationPolicy, implementation_sha256: str) -> str:
    """Pin both explicit preparation data and actual optional implementation bytes."""
    return _digest({"policy": policy.payload(), "implementation_sha256": _pin(implementation_sha256)})


def _verify_report(report: dict, *, parity, checkpoint_sha: str,
                    source_pins: Mapping[str, str], provenance: str, device: str) -> None:
    """Revalidate worker observations and recompute the engine's comparison data.

    No worker executes here. The externally pinned report is the trusted local
    measurement record; arbitrary booleans and cached summary numbers are not.
    """
    if (report.get("protocol") != parity.PROTOCOL or report.get("status") != "MEASURED"
            or report.get("blocked_reasons") != [] or report.get("checkpoint_kind") != "official_local"
            or report.get("checkpoint_sha256") != checkpoint_sha
            or report.get("checkpoint_provenance") != provenance
            or report.get("official_checkpoint_claim") != "caller_attested_not_publisher_authenticated"
            or report.get("provenance_verified_by_publisher") is not False
            or report.get("source_sha256") != dict(source_pins)
            or report.get("scientific_validation_performed") is not False
            or report.get("probabilities_calibrated") is not False
            or report.get("scientific_tolerance") is not None
            or report.get("scientific_decision") != "OWNER_REVIEW_REQUIRED"):
        raise _blocked("parity_report_is_not_a_complete_official_local_measurement_record")
    suite = report.get("cases")
    if (not isinstance(suite, list) or not 1 <= len(suite) <= 8
            or report.get("case_suite_sha256") != _digest(suite)):
        raise _blocked("parity_case_suite_identity_is_missing_or_inconsistent")
    cases = []
    try:
        for item in suite:
            if not isinstance(item, dict) or set(item) != {"case_id", "request"}:
                raise ValueError("case schema")
            request = ElementInferenceRequest(**item["request"])
            if len(request.site_ids) > 128:
                raise ValueError("case site bound")
            cases.append(parity.ParityCase(item["case_id"], request))
    except (ValueError, TypeError, KeyError) as exc:
        raise _blocked("parity_report_requests_are_invalid") from exc
    if len({case.case_id for case in cases}) != len(cases):
        raise _blocked("parity_report_case_identifiers_repeat")
    workers = report.get("workers")
    comparisons = report.get("comparisons")
    if not isinstance(workers, dict) or not isinstance(comparisons, dict) or device not in comparisons:
        raise _blocked("parity_report_has_no_measurement_for_selected_device")
    expected_modes = {"upstream_cpu"} | {"native_" + d for d in comparisons}
    if (not comparisons or any(d not in {"cpu", "mps"} for d in comparisons)
            or set(workers) != expected_modes):
        raise _blocked("parity_worker_and_comparison_modes_are_inconsistent")
    nonces = set()
    shapes = set()
    for mode, worker in workers.items():
        target = "cpu" if mode == "upstream_cpu" else mode[7:]
        if (not isinstance(worker, dict) or worker.get("protocol") != parity.PROTOCOL
                or worker.get("mode") != mode or worker.get("status") != "MEASURED"
                or worker.get("checkpoint_sha256") != checkpoint_sha
                or worker.get("source_sha256") != dict(source_pins)
                or worker.get("case_suite_sha256") != report["case_suite_sha256"]
                or worker.get("device_used") != target or worker.get("fell_back") is not False
                or worker.get("coverage_adjustment") is not False
                or worker.get("torch_threads") != 1
                or not isinstance(worker.get("python_executable"), str) or not worker["python_executable"]
                or not isinstance(worker.get("python_version"), str) or not worker["python_version"]
                or not isinstance(worker.get("nonce"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", worker["nonce"])):
            raise _blocked("parity_worker_identity_or_execution_mode_is_inconsistent")
        nonces.add(worker["nonce"])
        shape = worker.get("model_shape")
        fields = {"hidden_channels", "num_classes", "num_layers", "num_rbf", "max_z"}
        if (not isinstance(shape, dict) or set(shape) != fields
                or any(type(value) is not int or value < 1 for value in shape.values())):
            raise _blocked("parity_worker_model_shape_identity_is_missing")
        shapes.add(tuple(sorted(shape.items())))
        dependencies = worker.get("dependencies")
        required = {"torch", "torch_geometric", "torch_scatter", "torch_cluster", "ase"} if mode == "upstream_cpu" else {"torch"}
        if (not isinstance(dependencies, dict) or not required.issubset(dependencies)
                or any(not isinstance(dependencies[name], dict)
                       or not isinstance(dependencies[name].get("version"), str)
                       or not isinstance(dependencies[name].get("file"), str)
                       for name in required)):
            raise _blocked("parity_worker_dependency_evidence_is_missing")
        if (mode == "upstream_cpu" and worker.get("graph_implementation")
                != "actual torch_cluster.radius_graph / original upstream Distance"):
            raise _blocked("parity_reference_did_not_record_actual_upstream_graph")
        observations = worker.get("observations")
        if not isinstance(observations, list) or len(observations) != len(cases):
            raise _blocked("parity_worker_observations_are_incomplete")
        try:
            for observation, case in zip(observations, cases):
                parity._validate_observation(observation, case)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise _blocked("parity_worker_observations_fail_engine_protocol") from exc
    if len(nonces) != 1:
        raise _blocked("parity_workers_do_not_share_a_measurement_invocation")
    if len(shapes) != 1:
        raise _blocked("parity_workers_do_not_share_a_model_shape")
    try:
        for target, stored in comparisons.items():
            computed = [parity._compare_case(a, b, case) for a, b, case in zip(
                workers["upstream_cpu"]["observations"], workers["native_" + target]["observations"], cases)]
            # Normalize tuples/list formatting from engine results before comparing.
            if _digest(computed) != _digest(stored):
                raise _blocked("parity_comparison_summary_does_not_match_observations")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise _blocked("parity_comparison_cannot_be_recomputed") from exc


class AppleReconstructionPreparer:
    """Prepare only explicitly observed model sites and peaks, then periodic context."""

    def __init__(self, *, policy: ApplePreparationPolicy, policy_sha256: str,
                 implementation, implementation_sha256: str,
                 parsed_model_provider: Callable[[ModelSnapshot], ResFile], prepared_requests: set[str]):
        self.policy = policy
        self.policy_sha256 = policy_sha256
        self._implementation = implementation
        self._implementation_sha256 = implementation_sha256
        self._parsed_model_provider = parsed_model_provider
        self._prepared_requests = prepared_requests

    def prepare(self, model: ModelSnapshot, observation: DensityObservation) -> PreparedInference:
        _read_pinned(Path(self._implementation.__file__), self._implementation_sha256)
        observation.validate_for(model, synthetic=False)
        try:
            parsed = self._parsed_model_provider(model)
        except Exception as exc:
            raise _blocked("explicit_parsed_model_provider_failed") from exc
        if (not isinstance(parsed, ResFile) or parsed.sha256 != model.sha256
                or parsed.header.sha256 != model.sha256 or parsed.source != parsed.header.source
                or parsed.header.sfac != model.sfac or len(parsed.atoms) != len(model.sites)):
            raise _blocked("parsed_model_snapshot_or_header_identity_differs")
        for atom, site in zip(parsed.atoms, model.sites):
            value=atom.u_iso
            if self.policy.allow_anisotropic_metals and atom.is_anisotropic:
                from crystalx_core.refinement.cycle import METALS,displacement_value
                if parsed.element_of(atom) not in METALS:raise _blocked("anisotropic_nonmetal_outside_explicit_metal_policy")
                value=displacement_value(parsed,atom)
            if (atom.label != site.site_id or parsed.element_of(atom) != site.element
                    or (atom.x, atom.y, atom.z) != site.fractional or atom.occupancy != site.occupancy
                    or len(atom.u_values) not in ((1,6) if self.policy.allow_anisotropic_metals else (1,)) or value != site.u_iso):
                raise _blocked("parsed_model_site_fields_or_order_differ")
        main_ids = tuple(site.site_id for site in model.sites) + tuple(peak.peak_id for peak in observation.peaks)
        if len(set(main_ids)) != len(main_ids):
            raise _blocked("model_and_density_peak_site_identities_collide")
        descriptors = dict(self.policy.q_peak_atomic_numbers)
        if any(peak.peak_id not in descriptors for peak in observation.peaks):
            raise _blocked("observed_peak_lacks_an_explicit_computational_descriptor")
        # A density peak is not an atom: the occupancy/U placeholders are zero
        # and are never model features. No coordinates or element labels arise
        # from AI. Earlier RES peak rows are replaced by this explicit observed
        # snapshot, not merged with potentially stale residual information.
        peaks = tuple(QPeak(peak.peak_id, peak.fractional[0], peak.fractional[1], peak.fractional[2],
                            peak.height, index + 1, occupancy_code=0.0, u_iso=0.0)
                      for index, peak in enumerate(observation.peaks))
        neutral = replace(parsed, q_peaks=peaks)
        try:
            prepared = self._implementation.prepare_initial_solution(
                neutral,
                symmetry=self._implementation.CompleteSymmetry(self.policy.symmetry_operators,
                                                               self.policy.symmetry_provenance_reference),
                context_radius_angstrom=self.policy.context_radius_angstrom,
                q_peak_atomic_numbers=tuple(descriptors[peak.peak_id] for peak in observation.peaks),
                candidate_atomic_numbers=self.policy.candidate_atomic_numbers,
                max_sites=self.policy.max_sites, max_image_candidates=self.policy.max_image_candidates)
        except Exception as exc:
            raise _blocked("explicit_crystallographic_preprocessing_blocked") from exc
        request = prepared.request
        if (not isinstance(request, ElementInferenceRequest) or prepared.source_sha256 != model.sha256
                or request.main_site_count != len(main_ids) or len(prepared.sites) != len(request.site_ids)
                or len(request.site_ids) < len(main_ids)):
            raise _blocked("engine_preprocessing_snapshot_or_main_site_count_differs")
        identities = []
        main_origins = {}
        expected_fractional = tuple(site.fractional for site in model.sites) + tuple(p.fractional for p in observation.peaks)
        expected_descriptors = tuple(ATOMIC_NUMBERS[site.element] for site in model.sites) + tuple(descriptors[p.peak_id] for p in observation.peaks)
        for index, (site, engine_id) in enumerate(zip(prepared.sites, request.site_ids)):
            if site.site_id != engine_id or type(site.source_index) is not int or not 0 <= site.source_index < len(main_ids):
                raise _blocked("engine_preprocessing_site_provenance_is_inconsistent")
            if index < len(main_ids):
                if (site.is_main is not True or site.source_index != index
                        or tuple(site.fractional) != expected_fractional[index]
                        or request.atomic_numbers[index] != expected_descriptors[index]):
                    raise _blocked("engine_preprocessing_changed_a_measured_main_site_or_descriptor")
                main_origins[site.source_site_id] = index
                identities.append(main_ids[index])
            else:
                if (site.is_main is not False or main_origins.get(site.source_site_id) != site.source_index
                        or request.atomic_numbers[index] != expected_descriptors[site.source_index]):
                    raise _blocked("engine_preprocessing_context_has_no_observed_source_site")
                identities.append(main_ids[site.source_index] + "/context/" + engine_id)
        mapped = replace(request, site_ids=tuple(identities))
        if len(self._prepared_requests) >= 128:
            raise _blocked("prepared_inference_request_registry_budget_exhausted")
        self._prepared_requests.add(_digest(asdict(mapped)))
        return PreparedInference(mapped, model.sha256, self.policy_sha256)


class _BoundAppleBackend:
    name = "crystalx-apple-reconstruction-explicit-binding"

    def __init__(self, backend, *, audit: InferenceAudit, device: InferenceDevice,
                 checkpoint: Path, report: Path, source_root: Path, source_pins: Mapping[str, str],
                 prepared_requests: set[str]):
        self._backend = backend
        self.audit = audit
        self.device = device
        self._checkpoint = checkpoint
        self._report = report
        self._source_root = source_root
        self._source_pins = dict(source_pins)
        self._prepared_requests = prepared_requests

    def predict_elements(self, request: ElementInferenceRequest, *, device: InferenceDevice = InferenceDevice.AUTO,
                         allow_cpu_fallback: bool = False) -> ElementInferenceResult:
        if allow_cpu_fallback or device not in {InferenceDevice.AUTO, self.device}:
            raise _blocked("bound_apple_device_or_no_fallback_policy_differs")
        if not isinstance(request, ElementInferenceRequest) or _digest(asdict(request)) not in self._prepared_requests:
            raise _blocked("inference_request_was_not_bound_by_the_audited_preparer")
        _read_pinned(self._checkpoint, self.audit.weights_sha256)
        _read_pinned(self._report, self.audit.reference_comparison_sha256, max_bytes=_MAX_REPORT_BYTES)
        for name, sha in self._source_pins.items():
            _read_pinned(self._source_root / name, sha)
        try:
            result = self._backend.predict_elements(request, device=self.device, allow_cpu_fallback=False)
        except Exception as exc:
            raise _blocked("bound_apple_inference_failed_without_fallback") from exc
        count = request.main_site_count or len(request.site_ids)
        if (not isinstance(result, ElementInferenceResult) or result.fell_back is not False
                or result.device_used != self.device
                or result.model_identity.weights_sha256 != self.audit.weights_sha256
                or result.model_identity.source_files != self.audit.source_files
                or result.input_sha256 != _digest(asdict(request))
                or tuple(p.site_id for p in result.predictions) != request.site_ids[:count]
                or result.scientific_validation_performed or result.probabilities_calibrated):
            raise _blocked("bound_inference_result_identity_or_scope_differs")
        return result


@dataclass(frozen=True)
class AppleReconstructionBridge:
    inference: _BoundAppleBackend
    preparer: AppleReconstructionPreparer
    audit: InferenceAudit
    reference_measurement_consistency_verified: bool = True
    scientific_tolerance_approved: bool = False
    publisher_provenance_authenticated: bool = False
    upstream_preprocessing_parity_verified: bool = False
    full_density_map_claimed: bool = False


def create_apple_reconstruction_bridge(*, source_root: Path, source_pins: Mapping[str, str],
        checkpoint: Path, expected_checkpoint_sha256: str, official_provenance_reference: str,
        parity_report: Path, expected_parity_report_sha256: str,
        preprocessing_policy: ApplePreparationPolicy, expected_preprocessing_policy_sha256: str,
        expected_preprocessing_implementation_sha256: str,
        parsed_model_provider: Callable[[ModelSnapshot], ResFile],
        device: InferenceDevice = InferenceDevice.MPS) -> AppleReconstructionBridge:
    """Explicit local load boundary. Never download, launch parity workers or infer.

    ``official_provenance_reference`` is an externally supplied provenance
    attestation, not a publisher signature. A synthetic measurement record is
    always refused. Even a consistent official-local measurement remains an
    unapproved scientific comparison; real dispatch has independent blockers.
    """
    if (not isinstance(preprocessing_policy, ApplePreparationPolicy)
            or not callable(parsed_model_provider)
            or not isinstance(official_provenance_reference, str)
            or not official_provenance_reference.strip()
            or not isinstance(device, InferenceDevice) or device not in {InferenceDevice.CPU, InferenceDevice.MPS}):
        raise _blocked("explicit_preparation_provenance_parser_and_device_required")
    if preprocessing_binding_sha256(preprocessing_policy, expected_preprocessing_implementation_sha256) != _pin(expected_preprocessing_policy_sha256):
        raise _blocked("preprocessing_binding_pin_differs")
    checkpoint = Path(checkpoint).absolute()
    parity_report = Path(parity_report).absolute()
    source_root = Path(source_root).absolute()
    _read_pinned(checkpoint, expected_checkpoint_sha256)
    report = _strict_report(_read_pinned(parity_report, expected_parity_report_sha256,
                                        collect=True, max_bytes=_MAX_REPORT_BYTES))
    # Reject a blocked/synthetic record before importing any optional engine.
    if report.get("status") != "MEASURED" or report.get("checkpoint_kind") != "official_local":
        raise _blocked("measured_official_local_reference_comparison_required")
    if not isinstance(source_pins, Mapping) or not source_pins:
        raise _blocked("explicit_upstream_source_pins_required")
    for name, sha in source_pins.items():
        if (not isinstance(name, str) or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts or "\\" in name):
            raise _blocked("upstream_source_pin_path_is_unsafe")
        _read_pinned(source_root / name, sha)
    try:
        parity = importlib.import_module("crystalx_engine_apple_silicon.parity")
        preprocessing = importlib.import_module("crystalx_engine_apple_silicon.preprocessing")
    except (ImportError, OSError) as exc:
        raise _blocked("optional_apple_preprocessing_and_parity_package_missing") from exc
    if (any(not hasattr(parity, name) for name in ("PROTOCOL", "UPSTREAM_SHA256", "ParityCase",
                                                  "verify_upstream_sources", "_validate_observation", "_compare_case"))
            or not isinstance(getattr(preprocessing, "__file__", None), str)
            or not callable(getattr(preprocessing, "prepare_initial_solution", None))
            or not callable(getattr(preprocessing, "CompleteSymmetry", None))):
        raise _blocked("optional_apple_package_revision_lacks_required_binding_protocol")
    if dict(source_pins) != dict(parity.UPSTREAM_SHA256):
        raise _blocked("supplied_sources_are_not_the_audited_upstream_revision")
    try:
        parity.verify_upstream_sources(source_root)
    except Exception as exc:
        raise _blocked("upstream_source_revision_verification_failed") from exc
    _read_pinned(Path(preprocessing.__file__), expected_preprocessing_implementation_sha256)
    _verify_report(report, parity=parity, checkpoint_sha=expected_checkpoint_sha256,
                   source_pins=source_pins, provenance=official_provenance_reference, device=device.value)
    model_pins = tuple(sorted((PurePosixPath(name).name, sha) for name, sha in source_pins.items()
                              if str(PurePosixPath(name).parent) == "crystalx_infer/models"))
    if not model_pins:
        raise _blocked("audited_native_model_source_identity_missing")
    audit = InferenceAudit("official_trained", expected_checkpoint_sha256, model_pins,
                           expected_preprocessing_policy_sha256, expected_parity_report_sha256,
                           official_provenance_reference)
    try:
        backend_module = importlib.import_module("crystalx_engine_apple_silicon.backend")
        backend = backend_module.AppleSiliconBackend(source_root, checkpoint,
                                                    expected_sha256=expected_checkpoint_sha256)
    except Exception as exc:
        raise _blocked("optional_apple_backend_or_checkpoint_load_blocked") from exc
    # Catch races around weights-only construction before exposing the binding.
    _read_pinned(checkpoint, expected_checkpoint_sha256)
    _read_pinned(parity_report, expected_parity_report_sha256, max_bytes=_MAX_REPORT_BYTES)
    prepared_requests: set[str] = set()
    return AppleReconstructionBridge(
        _BoundAppleBackend(backend, audit=audit, device=device, checkpoint=checkpoint,
                           report=parity_report, source_root=source_root, source_pins=source_pins,
                           prepared_requests=prepared_requests),
        AppleReconstructionPreparer(policy=preprocessing_policy,
            policy_sha256=expected_preprocessing_policy_sha256, implementation=preprocessing,
            implementation_sha256=expected_preprocessing_implementation_sha256,
            parsed_model_provider=parsed_model_provider, prepared_requests=prepared_requests), audit)
