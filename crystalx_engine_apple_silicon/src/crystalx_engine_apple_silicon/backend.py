"""Experimental two-pass heavy-element inference on native torch CPU/MPS.

No periodic expansion, peak trimming, or SHELX file editing happens here.
Callers supply Cartesian sites and context explicitly. Forced coverage is an
explicit optional protocol returning choices separately from raw probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time

from crystalx_core.domain.inference import (
    ClassProbability, ElementInferenceRequest, ElementInferenceResult,
    ElementPrediction, InferenceDevice, ModelIdentity,
)

from .device import BackendUnavailable, DeviceDecision, select_device
from .source import load_model_classes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelShape:
    hidden_channels: int
    num_classes: int
    num_layers: int
    num_rbf: int
    max_z: int


def infer_model_shape(state) -> ModelShape:
    """Read tensor dimensions, not CLI defaults, and fail on unsupported layouts.

    Attention heads (8), cutoffs (0/5 A), neighbor cap (32) and activation are
    the audited inference defaults; these are not inferable from tensor shapes.
    """
    import torch

    if not isinstance(state, dict) or not state or any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Checkpoint must contain only a tensor state_dict")
    try:
        embedding = state["representation_model.embedding.weight"]
        rbf = state["representation_model.distance_expansion.means"]
        bias = state["output_model.output_network.1.update_net.2.bias"]
        layers = sorted({int(m.group(1)) for key in state
                         if (m := re.match(r"representation_model\.attention_layers\.(\d+)\.", key))})
        if (embedding.ndim != 2 or rbf.ndim != 1 or bias.ndim != 1
                or not layers or layers != list(range(len(layers))) or bias.numel() % 2):
            raise ValueError("Unsupported model tensor dimensions")
        shape = ModelShape(int(embedding.shape[1]), bias.numel() // 2,
                           len(layers), rbf.numel(), int(embedding.shape[0]))
        if shape.hidden_channels % 8 or min(shape.hidden_channels, shape.num_classes,
                                           shape.num_rbf, shape.max_z) < 2:
            raise ValueError("Unsupported model architecture")
        if any(v.dtype != torch.float32 for v in state.values()):
            raise ValueError("This CPU/MPS adapter requires all checkpoint tensors to be float32; implicit casting is refused")
        if any(not bool(torch.isfinite(v).all().item()) for v in state.values()):
            raise ValueError("Checkpoint contains nonfinite parameters")
        return shape
    except KeyError as exc:
        raise ValueError(f"Required inference checkpoint tensor is absent: {exc}") from exc


class AppleSiliconBackend:
    """Local weights-only loader; prediction remains an unvalidated hypothesis."""

    name = "crystalx-apple-experimental"

    def __init__(self, source_root: Path, checkpoint: Path, *, expected_sha256: str | None = None):
        import torch

        self.source_root = Path(source_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        self.weights_sha256 = _sha256(self.checkpoint)
        if expected_sha256 is not None and self.weights_sha256 != expected_sha256.lower():
            raise ValueError("Checkpoint SHA-256 differs from the expected digest")
        # Do not opt into unrestricted pickle even for an old checkpoint format.
        loaded = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self._state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        if _sha256(self.checkpoint) != self.weights_sha256:
            raise ValueError("Checkpoint changed while being loaded")
        self.shape = infer_model_shape(self._state)
        # Verify sources now as well as immediately before each model is built.
        self._source_identity = dict(load_model_classes(self.source_root).identity)

    def _run(self, request, used):
        predictions,observations,_=self._run_protocol(request,used,enforce_element_coverage=False)
        return predictions,observations

    def _run_protocol(self, request, used, *, enforce_element_coverage):
        import torch
        from .ops import radius_graph_with_diagnostics

        # Audited upstream forwards allocate tensors using torch's process-global default.
        # Refuse a caller's incompatible setting instead of mutating global state or allowing
        # implicit dtype conversions in an otherwise float32-attested computation.
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("This adapter requires torch's default dtype to be float32; caller setting was preserved")
        diagnostics = []

        def graph(*args, **kwargs):
            edges, observed = radius_graph_with_diagnostics(*args, **kwargs)
            diagnostics.append(observed)
            return edges

        classes = load_model_classes(self.source_root, radius_graph_fn=graph)
        if dict(classes.identity) != self._source_identity:
            raise ValueError("Model source identity changed")
        cfg = self.shape
        model = classes.TorchMD_Net(
            classes.TorchMD_ET(hidden_channels=cfg.hidden_channels, num_layers=cfg.num_layers,
                               num_rbf=cfg.num_rbf, num_heads=8, max_z=cfg.max_z,
                               max_num_neighbors=32),
            classes.EquivariantScalar(cfg.hidden_channels, num_classes=cfg.num_classes),
        )
        model.load_state_dict(self._state, strict=True)
        model.eval().to(used.value)
        z = torch.tensor(request.atomic_numbers, dtype=torch.long, device=used.value)
        cpu_pos = torch.tensor(request.positions_angstrom, dtype=torch.float32)
        if not bool(torch.isfinite(cpu_pos).all().item()):
            raise ValueError("Coordinates overflow float32")
        if len(set(map(tuple, cpu_pos.tolist()))) != len(request.site_ids):
            raise ValueError("Coincident float32 positions require explicit caller deduplication")
        pos = cpu_pos.to(used.value)
        batch = torch.zeros_like(z)
        candidates = tuple(sorted(request.candidate_atomic_numbers or set(request.atomic_numbers)))
        if len(candidates) < 2:
            raise ValueError("Provide at least two candidate elements to retain a runner-up")
        if max(request.atomic_numbers) >= cfg.max_z:
            raise ValueError("Input descriptor exceeds the checkpoint embedding range")
        if min(candidates) < 2 or max(candidates) >= min(cfg.num_classes, cfg.max_z):
            raise ValueError("Heavy-element candidates must fit both output classes and the embedding (Z>=2)")
        labels = torch.tensor(candidates, dtype=torch.long, device=used.value)
        with torch.inference_mode():
            logits = model(z, pos, batch)
            first = labels[logits[:, labels].argmax(dim=-1)]
            second = model(first, pos, batch)
            if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(second).all().item()):
                raise ValueError("Model produced nonfinite logits")
            probabilities = torch.softmax(second[:, labels], dim=-1).cpu()
        if used == InferenceDevice.MPS:
            torch.mps.synchronize()
        count = request.main_site_count or len(request.site_ids)
        chosen=labels.cpu()[probabilities[:count].argmax(dim=-1)]
        if enforce_element_coverage:
            from crystalx_core.app.inference_protocols import pinned_upstream_helper
            coverage=pinned_upstream_helper(self.source_root,"enforce_coverage_by_prob")
            chosen=coverage(probabilities[:count],labels.cpu(),chosen.clone())
        predictions = []
        for site_id, values in zip(request.site_ids[:count], probabilities[:count].tolist()):
            ranking = sorted(zip(candidates, values), key=lambda item: (-item[1], item[0]))[:5]
            predictions.append(ElementPrediction(site_id, tuple(ClassProbability(z, p) for z, p in ranking)))
        return tuple(predictions), diagnostics,tuple(int(value) for value in chosen.tolist())

    def predict_elements(self, request: ElementInferenceRequest, *, device=InferenceDevice.AUTO,
                         allow_cpu_fallback: bool = True) -> ElementInferenceResult:
        return self._predict_protocol(request,device=device,allow_cpu_fallback=allow_cpu_fallback,
                                      enforce_element_coverage=False)[0]

    def predict_elements_with_coverage(self,request:ElementInferenceRequest,*,device=InferenceDevice.AUTO,
                                      allow_cpu_fallback:bool=True):
        """Return raw ranked probabilities and separately coverage-adjusted choices."""
        return self._predict_protocol(request,device=device,allow_cpu_fallback=allow_cpu_fallback,
                                      enforce_element_coverage=True)

    def _predict_protocol(self,request,*,device,allow_cpu_fallback,enforce_element_coverage):
        started = time.monotonic()
        decision = select_device(device, allow_cpu_fallback=allow_cpu_fallback)
        def run(used):
            if enforce_element_coverage:
                return self._run_protocol(request,used,enforce_element_coverage=True)
            predictions,observations=self._run(request,used)
            return predictions,observations,tuple(p.topk[0].atomic_number for p in predictions)
        try:
            predictions, observations,choices = run(decision.used)
        except (RuntimeError, NotImplementedError) as exc:
            is_backend_failure = isinstance(exc, NotImplementedError) or any(
                token in str(exc).lower() for token in ("mps", "metal"))
            if decision.used != InferenceDevice.MPS or not is_backend_failure:
                raise
            if not allow_cpu_fallback:
                raise BackendUnavailable(f"Full-model MPS execution failed: {exc}") from exc
            decision = DeviceDecision(decision.requested, InferenceDevice.CPU, True,
                                      f"Full-model MPS execution failed; reran whole model on CPU: {exc}")
            predictions, observations,choices = run(decision.used)
        from .ops import NEIGHBOR_SELECTION_POLICY

        warnings = [
            "Experimental hypotheses only; no scientific validation or probability calibration performed.",
            "Nearest-neighbor truncation differs from upstream torch_cluster; trained-checkpoint parity is unverified.",
            "Uses audited inference defaults: 8 heads, 5 Angstrom cutoff, 32 neighbors; shapes cannot identify these settings.",
            ("Two-pass prediction with pinned upstream forced element coverage; adjusted choices are separate from raw rankings."
             if enforce_element_coverage else
             "Two-pass prediction without forced element coverage; caller supplies Cartesian/symmetry context."),
        ]
        truncated = max((item.truncated_nodes for item in observations), default=0)
        if truncated:
            warnings.append(f"Neighbor cap truncated candidates at {truncated} nodes per graph construction.")
        if decision.fell_back:
            warnings.append(decision.reason)
        cfg = self.shape
        import torch
        request_digest = hashlib.sha256(json.dumps(asdict(request), sort_keys=True,
                                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        result=ElementInferenceResult(
            predictions, ModelIdentity(self.weights_sha256, tuple(sorted(self._source_identity.items())),
                                       cfg.hidden_channels, cfg.num_classes, cfg.num_layers, cfg.num_rbf),
            decision.requested, decision.used, decision.fell_back, decision.reason,
            NEIGHBOR_SELECTION_POLICY, truncated, "softmax_conditioned_on_candidate_elements",
            tuple(warnings), time.monotonic() - started,
            request_digest, tuple(sorted(request.candidate_atomic_numbers or set(request.atomic_numbers))),
            str(torch.__version__),
        )
        return result,choices
