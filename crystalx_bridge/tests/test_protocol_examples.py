import json
from pathlib import Path

import pytest

from crystalx_bridge.helper import Bridge, OPERATIONS


def test_every_operation_and_diagnostic_action_has_schema_valid_golden_frames():
    jsonschema = pytest.importorskip("jsonschema")
    root = Path(__file__).parents[1]
    schema = json.loads((root / "protocol.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    cases = json.loads((root / "tests/golden/operations.json").read_text())["cases"]
    assert {v["request"]["op"] for v in cases} == set(OPERATIONS)
    assert {v["request"]["params"]["action"] for v in cases if v["request"]["op"] == "checkcif"} == {"start", "status", "cancel"}
    for case in cases:
        for key in ("request", "response", "error_response"):
            validator.validate(case[key])
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(case["error_request"])


def test_golden_errors_are_actual_dispatch_results_without_backend_access():
    class NoService:
        def __getattr__(self, name):
            pytest.fail("Invalid envelope should not access the backend")
    bridge = Bridge(NoService())
    cases = json.loads((Path(__file__).parent / "golden/operations.json").read_text())["cases"]
    for case in cases:
        assert bridge.handle(case["error_request"]) == case["error_response"]
