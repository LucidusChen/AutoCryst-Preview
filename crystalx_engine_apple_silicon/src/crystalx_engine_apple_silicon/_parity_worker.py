"""Private isolated worker; invoke only through :mod:`.parity`.

The upstream branch compiles the original verified source bytes without AST
rewriting and uses the installed torch_cluster, torch_scatter and PyG. Observing
its local graph alias and Distance output does not replace graph selection.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import platform
import sys
from types import ModuleType


def _upstream_modules(contents: dict[str, bytes], root: Path) -> dict:
    # Empty namespace parents avoid executing any unpinned __init__.py. The six
    # audited module bodies and all their dependency imports remain unchanged.
    for name in ("crystalx_infer", "crystalx_infer.models", "crystalx_infer.common"):
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    modules = {}
    for relative in contents:
        name = relative.removesuffix(".py").replace("/", ".")
        module = ModuleType(name)
        module.__file__ = str(root / relative)
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
        exec(compile(contents[relative], module.__file__, "exec"), module.__dict__)
        modules[name] = module
    return modules


def _edges(edge_index) -> list[list[int]]:
    return edge_index.detach().cpu().t().tolist()


def _eligible_counts(torch, positions, cutoff, batch):
    offsets = positions[:, None, :] - positions[None, :, :]
    mask = (offsets * offsets).sum(-1) < cutoff * cutoff
    if batch is not None:
        mask &= batch[:, None] == batch[None, :]
    return mask.sum(-1).cpu().tolist()


def _measure(payload):
    from crystalx_core.domain.inference import ElementInferenceRequest
    from crystalx_engine_apple_silicon.backend import infer_model_shape
    from crystalx_engine_apple_silicon.parity import (
        PROTOCOL, UPSTREAM_SHA256, ParityCase, _digest, _file_digest, verify_upstream_sources,
    )

    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported parity protocol")
    mode = payload["mode"]
    if mode not in ("upstream_cpu", "native_cpu", "native_mps"):
        raise ValueError("Unsupported worker mode")
    source_root = Path(payload["source_root"])
    contents = verify_upstream_sources(source_root)
    dependencies = {}
    names = ("torch", "torch_geometric", "torch_scatter", "torch_cluster", "ase") if mode == "upstream_cpu" else ("torch",)
    missing = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as exc:
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            dependencies[name] = {"version": str(getattr(module, "__version__", "unreported")),
                                  "file": str(getattr(module, "__file__", "unreported"))}
    if missing:
        return {"status": "BLOCKED", "reason": "Actual upstream CPU dependencies unavailable: " + "; ".join(missing),
                "dependencies": dependencies}
    import torch
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Isolated parity worker requires the float32 torch default dtype")
    torch.set_num_threads(1)
    checkpoint = Path(payload["checkpoint"])
    before = _file_digest(checkpoint)
    if before != payload["expected_sha256"]:
        raise ValueError("Checkpoint digest differs inside isolated worker")
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
    cfg = infer_model_shape(state)
    if _file_digest(checkpoint) != before:
        raise ValueError("Checkpoint changed while loading")
    graphs = []

    def observe_graph(fn, *args, **kwargs):
        positions = args[0] if args else kwargs["x"]
        edges = fn(*args, **kwargs)
        graphs.append({
            "radius_edges": _edges(edges),
            "eligible_neighbor_counts": _eligible_counts(
                torch, positions, kwargs.get("r", args[1] if len(args) > 1 else 5.), kwargs.get("batch")),
        })
        return edges

    if mode == "upstream_cpu":
        # Actual imports, not native replacements. All sources were checked
        # above before even the first upstream module executes.
        modules = _upstream_modules(contents, source_root)
        utils = modules["crystalx_infer.models.torchmd_utils"]
        actual_radius_graph = utils.radius_graph
        if not actual_radius_graph.__module__.startswith("torch_cluster"):
            raise ValueError("Upstream radius graph is not from torch_cluster")
        utils.radius_graph = lambda *a, **k: observe_graph(actual_radius_graph, *a, **k)
        representation_type = modules["crystalx_infer.models.torchmd_et"].TorchMD_ET
        network_type = modules["crystalx_infer.models.torchmd_net"].TorchMD_Net
        output_type = modules["crystalx_infer.models.noise_output_model"].EquivariantScalar
        two_pass = modules["crystalx_infer.common.heavy"].run_two_pass_heavy_prediction
        device = "cpu"
        graph_implementation = "actual torch_cluster.radius_graph / original upstream Distance"
    else:
        from crystalx_engine_apple_silicon.device import select_device
        from crystalx_engine_apple_silicon.ops import NEIGHBOR_SELECTION_POLICY, radius_graph
        from crystalx_engine_apple_silicon.source import load_model_classes
        decision = select_device(mode[7:], allow_cpu_fallback=False)
        device = decision.used.value
        classes = load_model_classes(source_root, radius_graph_fn=lambda *a, **k: observe_graph(radius_graph, *a, **k))
        representation_type, network_type, output_type = classes.TorchMD_ET, classes.TorchMD_Net, classes.EquivariantScalar
        graph_implementation = NEIGHBOR_SELECTION_POLICY

        def two_pass(model, input_z, pos, batch, mask, candidate_elements, enforce_element_coverage):
            if enforce_element_coverage:
                raise ValueError("Element coverage adjustment is excluded from model parity")
            logits = model(input_z, pos, batch)
            first = candidate_elements[logits[:, candidate_elements].argmax(-1)]
            second = model(first, pos, batch)[mask]
            if not torch.isfinite(logits).all() or not torch.isfinite(second).all():
                raise ValueError("Model produced nonfinite logits")
            probabilities = torch.softmax(second[:, candidate_elements], -1)
            return probabilities, candidate_elements[probabilities.argmax(-1)], first

    model = network_type(
        representation_type(hidden_channels=cfg.hidden_channels, num_layers=cfg.num_layers,
                            num_rbf=cfg.num_rbf, num_heads=8, max_z=cfg.max_z, max_num_neighbors=32),
        output_type(cfg.hidden_channels, num_classes=cfg.num_classes),
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    def effective_graph(_module, _input, output):
        if not graphs or "effective_edges" in graphs[-1]:
            raise ValueError("Unexpected upstream/native Distance graph invocation")
        graphs[-1]["effective_edges"] = _edges(output[0])

    model.representation_model.distance.register_forward_hook(effective_graph)
    # A finite logit check observes the original upstream call unchanged.
    def finite_output(_module, _input, output):
        if not torch.isfinite(output).all():
            raise ValueError("Model produced nonfinite logits")
    model.register_forward_hook(finite_output)
    observations = []
    case_ids = set()
    if not 1 <= len(payload["cases"]) <= 8:
        raise ValueError("Worker case count is out of bounds")
    for item in payload["cases"]:
        case = ParityCase(item["case_id"], ElementInferenceRequest(**item["request"]))
        if case.case_id in case_ids:
            raise ValueError("Worker case IDs must be unique")
        case_ids.add(case.case_id)
        request = case.request
        candidates = sorted(request.candidate_atomic_numbers or set(request.atomic_numbers))
        if (len(candidates) < 2 or min(candidates) < 2
                or max(candidates) >= min(cfg.num_classes, cfg.max_z)
                or max(request.atomic_numbers) >= cfg.max_z):
            raise ValueError("Case descriptors/candidates exceed supported heavy model dimensions")
        cpu_pos = torch.tensor(request.positions_angstrom, dtype=torch.float32)
        if not torch.isfinite(cpu_pos).all() or len(set(map(tuple, cpu_pos.tolist()))) != len(request.site_ids):
            raise ValueError("Case has nonfinite or coincident float32 positions")
        pos = cpu_pos.to(device)
        z = torch.tensor(request.atomic_numbers, dtype=torch.long, device=device)
        labels = torch.tensor(candidates, dtype=torch.long, device=device)
        mask = torch.arange(len(z), device=device) < (request.main_site_count or len(z))
        graphs.clear()
        with torch.inference_mode():
            probabilities, _, first = two_pass(
                model=model, input_z=z, pos=pos, batch=torch.zeros_like(z), mask=mask,
                candidate_elements=labels, enforce_element_coverage=False,
            )
        if device == "mps":
            torch.mps.synchronize()
        observations.append({
            "case_id": case.case_id, "input_sha256": _digest(asdict(request)),
            "site_ids": list(request.site_ids[:request.main_site_count]),
            "candidate_atomic_numbers": candidates,
            "first_pass_labels_all_sites": first.detach().cpu().tolist(),
            "probabilities": probabilities.detach().cpu().tolist(),
            "graphs": list(graphs),
        })
    verify_upstream_sources(source_root)
    return {"status": "MEASURED", "device_used": device, "fell_back": False,
            "checkpoint_sha256": before, "source_sha256": dict(UPSTREAM_SHA256),
            "case_suite_sha256": _digest(payload["cases"]), "dependencies": dependencies,
            "graph_implementation": graph_implementation,
            "model_shape": asdict(cfg), "observations": observations,
            "python_executable": sys.executable, "python_version": platform.python_version(),
            "torch_threads": torch.get_num_threads(), "coverage_adjustment": False}


def main():
    payload = json.load(sys.stdin)
    # Dependencies may print banners. Reserve stdout for our single JSON object.
    try:
        with redirect_stdout(sys.stderr):
            result = _measure(payload)
    except Exception as exc:
        result = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    result.update({"protocol": "crystalx-local-parity-v1", "nonce": payload.get("nonce"), "mode": payload.get("mode")})
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
