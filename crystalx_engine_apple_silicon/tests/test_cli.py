import json
from pathlib import Path

import pytest
import torch

from crystalx_engine_apple_silicon.cli import main
from crystalx_engine_apple_silicon.source import load_model_classes


SOURCE = Path(__file__).resolve().parents[2] / "crystalx_engine" / "inference_repo"


def test_synthetic_cli_is_explicit(source_root, capsys):
    assert main(["self-test", "--source-root", str(source_root), "--device", "cpu"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["synthetic_weights"] and not result["trained_checkpoint_tested"]
    assert result["argmax_agreement"] and result["software_smoke_passed"]
    assert result["result"]["device_used"] == "cpu"
    assert not result["result"]["scientific_validation_performed"]


def test_strict_mps_refuses_implicit_fallback(source_root, monkeypatch, capsys):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    assert main(["self-test", "--source-root", str(source_root), "--device", "mps", "--no-cpu-fallback"]) == 2
    assert json.loads(capsys.readouterr().err)["error_type"] == "BackendUnavailable"


def test_missing_snapshot_cannot_pass_synthetic_self_test(tmp_path, capsys):
    assert main(["self-test", "--source-root", str(tmp_path), "--device", "cpu"]) == 2
    assert json.loads(capsys.readouterr().err)["error_type"] == "SourceRevisionError"


def test_predict_writes_new_json_and_never_overwrites(source_root, tmp_path, capsys):
    classes = load_model_classes(source_root)
    model = classes.TorchMD_Net(classes.TorchMD_ET(hidden_channels=16, num_layers=1, num_rbf=8),
                                classes.EquivariantScalar(16, num_classes=32))
    weights = tmp_path / "synthetic.pth"
    torch.save(model.state_dict(), weights)
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"site_ids": ["a", "b"], "atomic_numbers": [6, 8],
                                "positions_angstrom": [[0, 0, 0], [1.4, 0, 0]],
                                "candidate_atomic_numbers": [6, 8]}))
    original = sites.read_bytes()
    output = tmp_path / "result.json"
    command = ["predict", "--source-root", str(source_root), "--checkpoint", str(weights),
               "--input", str(sites), "--output", str(output), "--device", "cpu"]
    assert main(command) == 0
    result = json.loads(output.read_text())
    assert result["candidate_atomic_numbers"] == [6, 8]
    assert len(result["predictions"]) == 2
    saved = output.read_bytes()
    assert main(command) == 2
    assert json.loads(capsys.readouterr().err)["error_type"] == "FileExistsError"
    assert output.read_bytes() == saved and sites.read_bytes() == original


@pytest.mark.parametrize("payload", ["not json", "[]", "{}"])
def test_invalid_json_fails_cleanly(tmp_path, payload, capsys):
    source = tmp_path / "input.json"
    source.write_text(payload)
    assert main(["predict", "--source-root", str(SOURCE), "--checkpoint", str(tmp_path / "unused.pth"),
                 "--input", str(source), "--device", "cpu"]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "failed"
