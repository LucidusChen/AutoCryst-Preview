"""Actual local pretrained inference, explicitly separated from scientific certification.

This experimental entry executes the pinned Apple CPU/MPS implementation without requiring
an upstream numerical-comparison report. The older strict reconstruction bridge is unchanged.
There is no training, download, weight substitution, SFAC rewrite, peak removal or CPU fallback.
Returned coordinates and assignments are private local evidence; callers retain them in runs/.
Importing this module does not import torch, Gemmi or the optional Apple package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib
import math
import re
from pathlib import Path
from typing import Any

from crystalx_core.app.reconstruction_inference import (
    ApplePreparationPolicy, AppleReconstructionPreparer, _read_pinned,
    preprocessing_binding_sha256,
)
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS, atomic_number
from crystalx_core.domain.inference import ElementInferenceRequest, ElementInferenceResult, InferenceDevice
from crystalx_core.domain.symmetry import SymmetryOperator, parse_symm
from crystalx_core.io.shelx_res import ResFile, read_res
from crystalx_core.io.shelx_reader import read_ins_header
from crystalx_core.app.inference_protocols import INPUT_PROTOCOLS, HEAVY_HELPER_SHA256, pinned_upstream_helper
from crystalx_core.workflow.reconstruction import DensityObservation, DensityPeak, ReconstructionBlocked, digest, read_snapshot
from crystalx_core.workspace.hashing import sha256_file

__all__ = ["OFFICIAL_HEAVY_SHA256", "OFFICIAL_HEAVY_REVISION", "OFFICIAL_HYDRO_SHA256",
           "RuntimeAssignment", "RuntimeElementAssigner"]

OFFICIAL_HEAVY_SHA256 = "5d378b5cc44d8eeac57f942b0fe827eb9f9999cb908b7a88d244f3614caba233"
OFFICIAL_HEAVY_REVISION = "2d621dd3d15bfaa87484b238f81446c474005c1f"
OFFICIAL_HYDRO_SHA256 = "5da6137f093474f0af3dd60babe37e1f144a99e28d8be17bdbd92004ad04f728"
_MAX_MODEL_BYTES = 16 * 1024 * 1024


class _RuntimeDensityObservation(DensityObservation):
    """Retain ordinary unwrapped SHELXL peak coordinates without changing legacy policy."""

    def validate_for(self,model,*,synthetic):
        # Validate shared model/source/capability metadata separately from the extended peak domain.
        DensityObservation(self.model_sha256,self.basis_key,self.source_sha256,self.kind,(),self.coverage).validate_for(
            model,synthetic=synthetic)
        if len({p.peak_id for p in self.peaks})!=len(self.peaks):
            raise ReconstructionBlocked("duplicate_density_peak_identity")
        for peak in self.peaks:
            if (re.fullmatch(r"[A-Za-z0-9_-]{1,64}",peak.peak_id) is None
                    or len(peak.fractional)!=3
                    or any(not math.isfinite(v) or not -5<v<5 for v in peak.fractional)
                    or not math.isfinite(peak.height) or peak.height<=0):
                raise ReconstructionBlocked("runtime_inference_q_peaks_outside_supported_fractional_domain")
_CENTRING_24 = {
    1: ((0, 0, 0),),
    2: ((0, 0, 0), (12, 12, 12)),
    3: ((0, 0, 0), (16, 8, 8), (8, 16, 16)),
    4: ((0, 0, 0), (0, 12, 12), (12, 0, 12), (12, 12, 0)),
    5: ((0, 0, 0), (0, 12, 12)),
    6: ((0, 0, 0), (12, 0, 12)),
    7: ((0, 0, 0), (12, 12, 0)),
}


@dataclass(frozen=True)
class RuntimeAssignment:
    request: ElementInferenceRequest
    result: ElementInferenceResult
    model_sha256: str
    site_ids: tuple[str, ...]
    peak_mapping: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]

    @property
    def inference_inputs(self) -> dict[str, Any]:
        return asdict(self.request)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _complete_symmetry(parsed: ResFile) -> tuple[tuple[SymmetryOperator, ...], str]:
    """Close the declared SHELX generators, including its centring and implicit inversion.

    Gemmi GroupOps API: https://gemmi.readthedocs.io/en/latest/symmetry.html
    No space-group name, expert structure or intensity-based assignment is inferred here.
    """
    latt = parsed.header.latt
    if type(latt) is not int or abs(latt) not in _CENTRING_24:
        raise ReconstructionBlocked("runtime_inference_requires_declared_supported_latt")
    try:
        gemmi = importlib.import_module("gemmi")
        texts = tuple(dict.fromkeys(("x,y,z", *parsed.header.symm)))
        if any(parse_symm(text) is None for text in texts):
            raise ReconstructionBlocked("runtime_inference_symmetry_generator_unreadable")
        group = gemmi.GroupOps([gemmi.Op(text) for text in texts])
        group.cen_ops = [list(vector) for vector in _CENTRING_24[abs(latt)]]
        group.add_missing_elements()
        if latt > 0:
            group.add_inversion()
        if not 1 <= len(group) <= 192:
            raise ReconstructionBlocked("runtime_inference_symmetry_order_unsupported")
        operators = tuple(parse_symm(op.triplet()) for op in group)
        if any(op is None for op in operators):
            raise ReconstructionBlocked("runtime_inference_symmetry_rotation_unsupported")
        reference = f"Gemmi {gemmi.__version__} closure of this model's explicit LATT/SYMM"
        return operators, reference
    except ReconstructionBlocked:
        raise
    except Exception as exc:
        raise ReconstructionBlocked("runtime_inference_gemmi_symmetry_unavailable") from exc


class RuntimeElementAssigner:
    """Load an actual local state_dict once and infer with an explicit native device.

    ``context_radius_angstrom`` is an experimental preparation policy. Its default is one
    model-edge cutoff, not a claim of upstream preprocessing or multilayer receptive-field
    equivalence. The full supplied model and every selected Q peak remain main sites.
    """

    def __init__(self, source_root: Path, checkpoint: Path, *,
                 expected_checkpoint_sha256: str = OFFICIAL_HEAVY_SHA256,
                 device: InferenceDevice = InferenceDevice.CPU,
                 context_radius_angstrom: float = 5.0, q_peak_atomic_number: int = 6,
                 max_sites: int = 20_000, max_image_candidates: int = 1_000_000,
                 allow_anisotropic_metals:bool=False):
        if (not isinstance(device, InferenceDevice) or device not in (InferenceDevice.CPU, InferenceDevice.MPS)
                or isinstance(context_radius_angstrom, bool) or not math.isfinite(context_radius_angstrom)
                or context_radius_angstrom <= 0 or type(q_peak_atomic_number) is not int
                or not 1 <= q_peak_atomic_number <= 118
                or any(type(n) is not int or n < 1 for n in (max_sites, max_image_candidates))
                or type(allow_anisotropic_metals) is not bool):
            raise ReconstructionBlocked("runtime_inference_explicit_device_and_bounded_preparation_required")
        self.source_root = Path(source_root).absolute()
        self.checkpoint = Path(checkpoint).absolute()
        self.expected_sha256 = expected_checkpoint_sha256
        _read_pinned(self.checkpoint, expected_checkpoint_sha256, max_bytes=512 * 1024 * 1024)
        try:
            parity = importlib.import_module("crystalx_engine_apple_silicon.parity")
            self.preprocessing = importlib.import_module("crystalx_engine_apple_silicon.preprocessing")
            self.source_pins = dict(parity.UPSTREAM_SHA256)
            for name, expected in self.source_pins.items():
                _read_pinned(self.source_root / name, expected)
            self.preparation_path = Path(self.preprocessing.__file__).absolute()
            self.preparation_sha256 = sha256_file(self.preparation_path)
            backend = importlib.import_module("crystalx_engine_apple_silicon.backend")
            self.backend = backend.AppleSiliconBackend(self.source_root, self.checkpoint,
                                                      expected_sha256=expected_checkpoint_sha256)
        except ReconstructionBlocked:
            raise
        except Exception as exc:
            raise ReconstructionBlocked("runtime_pretrained_engine_or_checkpoint_incompatible") from exc
        self.device = device
        self.radius = context_radius_angstrom
        self.q_descriptor = q_peak_atomic_number
        self.max_sites, self.max_image_candidates = max_sites, max_image_candidates
        self.allow_anisotropic_metals=allow_anisotropic_metals
        self._verify_assets()

    def _verify_assets(self) -> None:
        _read_pinned(self.checkpoint, self.expected_sha256, max_bytes=512 * 1024 * 1024)
        _read_pinned(self.preparation_path, self.preparation_sha256)
        for name, expected in self.source_pins.items():
            _read_pinned(self.source_root / name, expected)

    def assign_upstream_heavy(self, model_path: Path) -> RuntimeAssignment:
        """One opt-in initial SHELXT heavy stage; no Q peak descriptors."""
        from crystalx_core.app.upstream_heavy import infer_upstream_heavy
        return infer_upstream_heavy(self,model_path)

    def assign(self, model_path: Path, *, candidate_elements: tuple[str, ...],
               include_q_peaks: bool = True,input_protocol:str="current_labels",
               original_ins_path:Path|None=None,initial_labels:dict[str,str]|None=None,
               protocol_seed:int=42) -> RuntimeAssignment:
        if (not isinstance(candidate_elements, tuple) or not candidate_elements
                or any(not isinstance(element, str) for element in candidate_elements)
                or type(include_q_peaks) is not bool):
            raise ReconstructionBlocked("runtime_inference_requires_explicit_candidate_elements")
        if input_protocol not in INPUT_PROTOCOLS or type(protocol_seed) is not int or not 0<=protocol_seed<2**32:
            raise ReconstructionBlocked("runtime_inference_input_protocol_or_seed_invalid")
        try:
            supplied = tuple(sorted(set(atomic_number(element) for element in candidate_elements)))
        except (KeyError, ValueError) as exc:
            raise ReconstructionBlocked("runtime_inference_candidate_element_unknown") from exc
        # This is the heavy-element head. H remains an input site/descriptor when present,
        # but is explicitly omitted from the output candidate set; no model file is edited.
        candidates = tuple(z for z in supplied if z > 1)
        if len(candidates) < 2 or max(candidates) >= min(self.backend.shape.num_classes, self.backend.shape.max_z):
            raise ReconstructionBlocked("runtime_inference_candidates_do_not_fit_heavy_checkpoint")
        self._verify_assets()
        path = Path(model_path).absolute()
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_MODEL_BYTES:
            raise ReconstructionBlocked("runtime_inference_model_must_be_a_bounded_regular_snapshot")
        model_sha = sha256_file(path)
        _read_pinned(path, model_sha, max_bytes=_MAX_MODEL_BYTES)
        parsed = read_res(path)
        model = read_snapshot(path, reflection_sha256="runtime-model-only-no-reflection-binding",
                              coordinate_bound=5.0, allow_nonpositive_u=True,
                              allow_anisotropic_metals=self.allow_anisotropic_metals)
        if parsed.sha256 != model_sha or model.sha256 != model_sha:
            raise ReconstructionBlocked("runtime_inference_model_changed_during_parse")
        operators, symmetry_source = _complete_symmetry(parsed)
        peaks = tuple(DensityPeak(p.label, (p.x, p.y, p.z), p.height)
                      for p in parsed.q_peaks) if include_q_peaks else ()
        observation = _RuntimeDensityObservation(model_sha, model.basis_key, model_sha, "Q_PEAK_LIST", peaks, "partial")
        policy = ApplePreparationPolicy(operators, symmetry_source, self.radius, candidates,
            tuple((p.peak_id, self.q_descriptor) for p in peaks), self.max_sites, self.max_image_candidates,
            allow_anisotropic_metals=self.allow_anisotropic_metals)
        policy_sha = preprocessing_binding_sha256(policy, self.preparation_sha256)
        preparer = AppleReconstructionPreparer(policy=policy, policy_sha256=policy_sha,
            implementation=self.preprocessing, implementation_sha256=self.preparation_sha256,
            parsed_model_provider=lambda _model: parsed, prepared_requests=set())
        prepared = preparer.prepare(model, observation)
        request = prepared.request
        request,protocol_record=self._input_protocol(request,input_protocol=input_protocol,
            original_ins_path=original_ins_path,initial_labels=initial_labels,protocol_seed=protocol_seed)
        candidates=request.candidate_atomic_numbers
        _read_pinned(path, model_sha, max_bytes=_MAX_MODEL_BYTES)
        try:
            if input_protocol=="upstream_ins":
                result,choices=self.backend.predict_elements_with_coverage(request,device=self.device,allow_cpu_fallback=False)
            else:
                result = self.backend.predict_elements(request, device=self.device, allow_cpu_fallback=False)
                choices=tuple(prediction.topk[0].atomic_number for prediction in result.predictions)
        except Exception as exc:
            raise ReconstructionBlocked("runtime_pretrained_prediction_failed_without_fallback") from exc
        expected_ids = request.site_ids[:request.main_site_count]
        expected_model_sources = tuple(sorted((Path(name).name, sha) for name, sha in self.source_pins.items()
                                               if str(Path(name).parent) == "crystalx_infer/models"))
        if (not isinstance(result, ElementInferenceResult) or result.input_sha256 != digest(request)
                or result.model_identity.weights_sha256 != self.expected_sha256 or result.fell_back is not False
                or result.model_identity.source_files != expected_model_sources
                or result.candidate_atomic_numbers != candidates
                or result.device_requested != self.device or result.device_used != self.device
                or tuple(p.site_id for p in result.predictions) != expected_ids
                or result.scientific_validation_performed or result.probabilities_calibrated):
            raise ReconstructionBlocked("runtime_pretrained_prediction_identity_or_scope_differs")
        if len(choices)!=len(expected_ids) or any(type(z) is not int or z not in candidates for z in choices):
            raise ReconstructionBlocked("runtime_protocol_element_choices_do_not_match_main_sites")
        for prediction in result.predictions:
            top = prediction.topk
            if (not 2 <= len(top) <= min(5, len(candidates))
                    or len({p.atomic_number for p in top}) != len(top)
                    or any(p.atomic_number not in candidates or not math.isfinite(p.probability)
                           or not 0 <= p.probability <= 1 for p in top)
                    or any(a.probability < b.probability for a, b in zip(top, top[1:]))
                    or sum(p.probability for p in top) > 1.00001):
                raise ReconstructionBlocked("runtime_pretrained_probability_rows_invalid")
        self._verify_assets()
        _read_pinned(path, model_sha, max_bytes=_MAX_MODEL_BYTES)
        if protocol_record["original_ins_path"] is not None:
            _read_pinned(Path(protocol_record["original_ins_path"]),protocol_record["original_ins_sha256"],max_bytes=_MAX_MODEL_BYTES)
        main_atom_count = len(model.sites)
        peak_mapping = tuple({"peak_id": peak.peak_id, "prediction_index": main_atom_count + index,
                              "computational_descriptor_z": self.q_descriptor,
                              "coordinate_source": "retained_model_Q_peak_record"}
                             for index, peak in enumerate(peaks))
        provenance = {
            "protocol": "crystalx-pretrained-runtime/1", "model_path": str(path),
            "weights_path": str(self.checkpoint), "weights_sha256": self.expected_sha256,
            "official_revision": OFFICIAL_HEAVY_REVISION if self.expected_sha256 == OFFICIAL_HEAVY_SHA256 else None,
            "source_sha256": self.source_pins, "preprocessing_sha256": self.preparation_sha256,
            "runtime_implementation_sha256": sha256_file(Path(__file__)),
            "preprocessing_policy_sha256": policy_sha, "symmetry_provenance": symmetry_source,
            "symmetry_operator_count": len(operators), "context_radius_angstrom": self.radius,
            "main_atom_count": main_atom_count, "selected_peak_count": len(peaks),
            "total_graph_sites": len(request.site_ids), "model_only_no_reflection_binding": True,
            "main_coordinates_preserved_without_wrapping": True,
            "ordinary_main_fractional_coordinate_bound": 5.0,
            "nonpositive_u_sites_preserved_for_geometry_only_inference": [s.site_id for s in model.sites if s.u_iso<=0],
            "anisotropic_metals_allowed":self.allow_anisotropic_metals,
            "displacement_scalar":"Uiso_or_cell_metric_Ueq; raw ADPs unchanged",
            "nonpositive_u_scientifically_approved": False,
            "selected_q_peak_fractional_domain": "(-5,5); ordinary unwrapped coordinates preserved; no clipping or wrapping",
            "hydrogen_output_candidate_explicitly_omitted": 1 in supplied,
            "q_peak_descriptor_is_a_computational_input_not_chemical_identity": True,
            "actual_pretrained_weights_executed": True, "model_training_performed": False,
            "upstream_prediction_parity_verified": False, "upstream_preprocessing_parity_verified": False,
            "scientific_validation_performed": False, "probabilities_calibrated": False,
            "no_cpu_fallback": True, "density_scope": "partial_Q_peak_list_no_full_map_claim",
            "input_protocol":input_protocol,"input_protocol_record":protocol_record,
            "protocol_element_choices":{key:next(symbol for symbol,z in ATOMIC_NUMBERS.items() if z==chosen)
                                        for key,chosen in zip(expected_ids,choices)},
            "coverage_adjusted_sites":[key for key,chosen,prediction in zip(expected_ids,choices,result.predictions)
                                       if chosen!=prediction.topk[0].atomic_number],
            "coverage_choices_are_not_ranked_probabilities":True,
        }
        return RuntimeAssignment(request, result, model_sha, expected_ids, peak_mapping, provenance)

    def _input_protocol(self,request,*,input_protocol,original_ins_path,initial_labels,protocol_seed):
        """Replace computational descriptors only; model coordinates/bytes stay unchanged."""
        original=None;original_sha=None;header=None
        if original_ins_path is not None:
            original=Path(original_ins_path).absolute()
            try:
                if original.is_symlink() or not original.is_file():raise OSError("not a regular original INS")
                original_sha=sha256_file(original)
                _read_pinned(original,original_sha,max_bytes=_MAX_MODEL_BYTES)
                header=read_ins_header(original)
            except ReconstructionBlocked:raise
            except Exception as error:
                raise ReconstructionBlocked("input_protocol_original_ins_unreadable") from error
            if header.sha256!=original_sha or header.has_atom_records:
                raise ReconstructionBlocked("input_protocol_requires_atom_free_original_ins")
        main_count=request.main_site_count or len(request.site_ids)
        current=tuple(request.atomic_numbers)
        declared_labels={} if initial_labels is None else dict(initial_labels)
        if (any(not isinstance(key,str) or not key or not isinstance(value,str) or value not in ATOMIC_NUMBERS
                for key,value in declared_labels.items())):
            raise ReconstructionBlocked("initial_label_protocol_requires_known_element_symbols")
        if input_protocol in ("initial_labels","upstream_ins") and not declared_labels:
            raise ReconstructionBlocked("input_protocol_requires_frozen_initial_automatic_labels")
        fallback=[]
        initial=[]
        for key in request.site_ids:
            source=key.split("/context/",1)[0]
            if source in declared_labels:initial.append(atomic_number(declared_labels[source]))
            else:
                initial.append(self.q_descriptor)
                if source not in fallback:fallback.append(source)
        numbers=current
        candidates=request.candidate_atomic_numbers
        ratio_metadata=None
        if input_protocol=="initial_labels":
            numbers=tuple(initial)
        elif input_protocol=="upstream_ins":
            if header is None or not header.sfac or len(header.sfac)!=len(header.unit):
                raise ReconstructionBlocked("upstream_input_protocol_requires_original_sfac_unit")
            if any(not math.isfinite(value) or value<0 for value in header.unit):
                raise ReconstructionBlocked("upstream_input_protocol_unit_counts_invalid")
            if 1 in current[:main_count]:
                raise ReconstructionBlocked("upstream_heavy_protocol_requires_nonhydrogen_main_sites")
            counts=tuple(int(float(value)) for value in header.unit)
            try:
                ratio=pinned_upstream_helper(self.source_root,"build_sorted_init_z_from_ratio")
                main=tuple(ratio(header.sfac,counts,main_count))
            except Exception as error:
                raise ReconstructionBlocked("upstream_ratio_initialization_failed") from error
            numbers=main+tuple(initial[main_count:])
            if len(numbers)!=len(current) or any(z not in candidates for z in numbers):
                raise ReconstructionBlocked("upstream_graph_descriptors_exceed_declared_candidate_scope")
            candidates=tuple(sorted(set(numbers)))
            if len(candidates)<2:
                raise ReconstructionBlocked("upstream_graph_requires_two_effective_element_candidates")
            ratio_metadata={"original_sfac":header.sfac,"original_unit":header.unit,
                "integer_unit_used":counts,"count_conversion":"int(float) as pinned upstream parser",
                "allocation":"pinned largest-remainder then descending atomic number",
                "main_site_count":main_count,"unit_is_declared_not_scientifically_validated":True}
        if max(numbers)>=self.backend.shape.max_z:
            raise ReconstructionBlocked("input_protocol_descriptor_exceeds_checkpoint_embedding")
        mapped=replace(request,atomic_numbers=numbers,candidate_atomic_numbers=candidates)
        record={"protocol":input_protocol,"seed":protocol_seed,
            "randomized_initialization":False,"seed_role":"recorded_only_no_random_draws_in_pinned_upstream_allocation",
            "original_ins_path":str(original) if original is not None else None,"original_ins_sha256":original_sha,
            "initial_labels_sha256":digest(declared_labels) if declared_labels else None,
            "initial_labels":declared_labels,"new_site_fallback_atomic_number":self.q_descriptor,
            "new_site_fallback_ids":fallback if input_protocol!="current_labels" else [],
            "main_input_atomic_numbers":numbers[:main_count],
            "context_input_source":"frozen_initial_labels" if input_protocol!="current_labels" else "current_labels",
            "main_input_source":"original_ins_sfac_unit_ratio" if input_protocol=="upstream_ins" else input_protocol,
            "requested_candidate_atomic_numbers":request.candidate_atomic_numbers,
            "effective_candidate_atomic_numbers":candidates,"ratio_initialization":ratio_metadata,
            "forced_element_coverage":input_protocol=="upstream_ins",
            "heavy_helper_sha256":HEAVY_HELPER_SHA256 if input_protocol=="upstream_ins" else None,
            "second_pass_input":"all_first_pass_graph_predictions",
            "descriptor_request_sha256":digest(mapped),"geometry_request_sha256":digest(request),
            "full_upstream_preprocessing_parity_verified":False}
        return mapped,record
