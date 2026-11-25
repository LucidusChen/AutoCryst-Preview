"""Opt-in JSON boundary; hand-authored fixtures only, no trained or scientific pass."""

import json

import pytest

from crystalx_engine_apple_silicon.cli import main
from test_backend import SOURCE


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "synthetic.res"
    model.write_text("TITL synthetic\nCELL .71073 10 10 10 90 90 90\nLATT -1\n"
                     "SFAC C O\nUNIT 1 1\nC1 1 .1 .2 .3 11 .05\n"
                     "O1 2 .2 .2 .3 11 .05\nHKLF 4\nEND\n")
    symmetry = tmp_path / "symmetry.json"
    symmetry.write_text(json.dumps({"source": "hand-authored P1 fixture", "operators": ["x,y,z"]}))
    return model, symmetry


def prepare_args(inputs):
    return ["prepare", "--input", str(inputs[0]), "--symmetry-config", str(inputs[1]),
            "--context-radius", "1", "--candidates", "6", "8"]


def test_prepare_json_request_round_trips_to_neutral_contract(inputs, capsys):
    from crystalx_core.domain.inference import ElementInferenceRequest
    assert main(prepare_args(inputs)) == 0
    payload = json.loads(capsys.readouterr().out)
    request = ElementInferenceRequest(**payload["request"])
    assert request.atomic_numbers == (6, 8)
    assert request.positions_angstrom[0] == pytest.approx((1, 2, 3))
    assert payload["status"] == "EXPERIMENTAL_PREPARED"
    assert payload["main_mask"] == [True, True]
    assert payload["scientific_validation_performed"] is False
    assert payload["provenance"]["symmetry"]["operators"][0]["translation_fractional"] == ["0", "0", "0"]


def test_prepare_never_clobbers_an_existing_artifact(inputs, tmp_path, capsys):
    output = tmp_path / "out.json"
    output.write_text("keep existing")
    assert main(prepare_args(inputs) + ["--output", str(output)]) == 2
    assert output.read_text() == "keep existing"


@pytest.mark.parametrize("operators", [[], ["x+0.1,y,z"], ["bad"], ["x+1/0,y,z"], ["-x,-y,-z"]])
def test_prepare_blocks_missing_malformed_or_mismatched_symmetry(inputs, operators, capsys):
    inputs[1].write_text(json.dumps({"source": "test", "operators": operators}))
    assert main(prepare_args(inputs)) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "failed"


def test_parity_without_weights_is_blocked_and_never_reports_measurement(capsys):
    assert main(["parity", "--source-root", str(SOURCE), "--provenance", "No owner-supplied checkpoint"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert payload["trained_comparison_measured"] is False
    assert payload["workers"] == {} and payload["comparisons"] == {}
    assert payload["scientific_tolerance"] is None
