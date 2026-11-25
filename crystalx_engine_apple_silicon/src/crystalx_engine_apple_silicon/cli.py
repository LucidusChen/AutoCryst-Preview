"""Local JSON interface and a clearly synthetic CPU/MPS smoke test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import tempfile


def _parser():
    parser = argparse.ArgumentParser(description="Experimental CrystalX native-torch CPU/MPS adapter")
    from . import __version__
    parser.add_argument("--version", action="version", version=__version__)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("doctor", help="Probe torch/MPS without changing system settings")
    prepare = actions.add_parser("prepare", help="Experimental, explicitly configured crystallographic preprocessing")
    prepare.add_argument("--input", type=Path, required=True, help="Initial solution RES, never an answer structure")
    prepare.add_argument("--symmetry-config", type=Path, required=True,
                         help="JSON with provenance source and complete exact fractional operators")
    prepare.add_argument("--context-radius", type=float, required=True,
                         help="Explicit context radius in Angstrom, not an attestation of receptive-field coverage")
    prepare.add_argument("--q-descriptors", type=int, nargs="*", default=None)
    prepare.add_argument("--candidates", type=int, nargs="+", default=None)
    prepare.add_argument("--output", type=Path, help="New JSON file, never overwritten")
    parity = actions.add_parser("parity", help="Measure against actual upstream CPU graph; missing prerequisites BLOCK")
    parity.add_argument("--source-root", type=Path, required=True)
    parity.add_argument("--checkpoint", type=Path)
    parity.add_argument("--weights-sha256")
    parity.add_argument("--provenance", required=True)
    parity.add_argument("--checkpoint-kind", choices=("official_local", "synthetic_fixture"), default="official_local")
    parity.add_argument("--devices", choices=("cpu", "mps"), nargs="+", default=("cpu", "mps"))
    parity.add_argument("--reference-python", type=Path)
    parity.add_argument("--output", type=Path, help="New measurement JSON, never overwritten")
    for name in ("predict", "self-test"):
        sub = actions.add_parser(name)
        sub.add_argument("--source-root", type=Path, required=True,
                         help="User-supplied audited inference_repo; never bundled")
        sub.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
        sub.add_argument("--no-cpu-fallback", action="store_true")
        if name == "predict":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--weights-sha256")
            sub.add_argument("--input", type=Path, required=True, help="Cartesian-site JSON")
            sub.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten")
    return parser


def _prepare(args):
    from crystalx_core.domain.symmetry import parse_symm
    from crystalx_core.io.shelx_res import read_res
    from .preprocessing import CompleteSymmetry, prepare_initial_solution

    for path in (args.input, args.symmetry_config):
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Preprocessing input exceeds the 4 MiB interface limit")
    config = json.loads(args.symmetry_config.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or set(config) != {"source", "operators"}
            or not isinstance(config["operators"], list)
            or not 1 <= len(config["operators"]) <= 192):
        raise ValueError("Symmetry config requires source and 1 to 192 explicit operators")
    operators = []
    for text in config["operators"]:
        if not isinstance(text, str) or re.search(r"\d*\.\d", text):
            raise ValueError("Symmetry operators require exact fractional text, not decimal translations")
        try:
            operator = parse_symm(text)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(f"Cannot parse explicit symmetry operator {text!r}") from exc
        if operator is None:
            raise ValueError(f"Cannot parse explicit symmetry operator {text!r}")
        operators.append(operator)
    prepared = prepare_initial_solution(
        read_res(args.input), symmetry=CompleteSymmetry(tuple(operators), config["source"]),
        context_radius_angstrom=args.context_radius,
        q_peak_atomic_numbers=tuple(args.q_descriptors) if args.q_descriptors is not None else None,
        candidate_atomic_numbers=tuple(args.candidates) if args.candidates is not None else None,
    )
    provenance = asdict(prepared)
    request = provenance.pop("request")
    provenance["symmetry"] = {"source": prepared.symmetry.source, "operators": [
        {"rotation": op.rotation, "translation_fractional": [str(t) for t in op.translation]}
        for op in prepared.symmetry.operators
    ]}
    return {
        "schema": "crystalx.prepared-inference/1", "status": "EXPERIMENTAL_PREPARED",
        "request": request, "provenance": provenance, "main_mask": prepared.main_mask,
        "scientific_validation_performed": False, "probabilities_calibrated": False,
        "meaning": "Prepared experimental descriptors/context, not assigned elements or a valid structure",
    }


def _self_test(args):
    import torch
    from crystalx_core.domain.inference import ElementInferenceRequest
    from .backend import AppleSiliconBackend
    from .source import load_model_classes

    torch.manual_seed(42)
    classes = load_model_classes(args.source_root)
    model = classes.TorchMD_Net(
        classes.TorchMD_ET(hidden_channels=16, num_layers=2, num_rbf=8),
        classes.EquivariantScalar(16, num_classes=32),
    )
    request = ElementInferenceRequest(
        ("synthetic-1", "synthetic-2", "synthetic-3", "synthetic-4"),
        (6, 7, 8, 6), ((0., 0., 0.), (1.4, 0., 0.), (0., 1.5, 0.), (0., 0., 1.6)),
        (6, 7, 8),
    )
    with tempfile.TemporaryDirectory(prefix="crystalx-apple-self-test-") as tmp:
        weights = Path(tmp) / "synthetic.pth"
        torch.save(model.state_dict(), weights)
        engine = AppleSiliconBackend(args.source_root, weights)
        cpu = engine.predict_elements(request, device="cpu")
        selected = engine.predict_elements(request, device=args.device,
                                           allow_cpu_fallback=not args.no_cpu_fallback)
    deltas = []
    for first, second in zip(cpu.predictions, selected.predictions):
        other = {p.atomic_number: p.probability for p in second.topk}
        deltas.extend(abs(p.probability - other[p.atomic_number]) for p in first.topk)
    agreement = all(a.topk[0].atomic_number == b.topk[0].atomic_number
                    for a, b in zip(cpu.predictions, selected.predictions))
    return {
        "synthetic_weights": True, "trained_checkpoint_tested": False,
        "comparison": "native CPU versus selected backend, NOT versus upstream torch_cluster",
        "argmax_agreement": agreement, "max_probability_delta": max(deltas, default=0.),
        "software_tolerance": 1e-4,
        "software_smoke_passed": agreement and max(deltas, default=0.) <= 1e-4,
        "result": asdict(selected),
    }


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            import torch
            from .device import select_device
            payload = {"torch_version": torch.__version__,
                       "mps_built": torch.backends.mps.is_built(),
                       "mps_available": torch.backends.mps.is_available(),
                       "device_decision": asdict(select_device("auto")),
                       "system_changed": False}
        elif args.command == "self-test":
            payload = _self_test(args)
        elif args.command == "prepare":
            payload = _prepare(args)
        elif args.command == "parity":
            from .parity import measure_parity
            payload = measure_parity(
                args.source_root, args.checkpoint, expected_sha256=args.weights_sha256,
                checkpoint_kind=args.checkpoint_kind, provenance=args.provenance,
                devices=tuple(args.devices), reference_python=args.reference_python,
            )
        else:
            from crystalx_core.domain.inference import ElementInferenceRequest
            from .backend import AppleSiliconBackend
            if args.input.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Input JSON exceeds the 4 MiB interface limit")
            request = ElementInferenceRequest(**json.loads(args.input.read_text(encoding="utf-8")))
            engine = AppleSiliconBackend(args.source_root, args.checkpoint,
                                        expected_sha256=args.weights_sha256)
            payload = asdict(engine.predict_elements(request, device=args.device,
                                                     allow_cpu_fallback=not args.no_cpu_fallback))
        rendered = json.dumps(payload, indent=2, allow_nan=False)
        if getattr(args, "output", None):
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        else:
            print(rendered)
        if args.command == "parity" and payload["status"] == "BLOCKED":
            return 3
        return 1 if args.command == "self-test" and not payload["software_smoke_passed"] else 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "detail": str(exc)}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
