"""Independent diagnostics use fabricated reports, never proprietary programs."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from crystalx_core.app.checkcif import CheckCifStatus, PinnedArtifact, run_local_checkcif
from crystalx_core.app.models import RunMode
from crystalx_core.domain.tools import ToolKind
from crystalx_core.private_acceptance.sealing import seal_run
from crystalx_core.tools.adapter import FileRecord, ToolInvocationRecord
from crystalx_core.tools.runner import CancelToken, ExecutionPlan


class FabricatedAdapter:
    """An injected adapter for evidence aggregation only; no process is launched."""
    kind = ToolKind.PLATON_SPECIAL
    identity = SimpleNamespace(sha256="c" * 64)

    def __init__(self, text="PLAT001_ALERT_2_C Synthetic test alert\n", *, success=True, alter_input=False):
        self.text, self.success, self.alter_input = text, success, alter_input
        self.calls = 0
        self.last_plan = None

    def build_plan(self, **kwargs):
        self.last_plan = ExecutionPlan(argv=("fabricated-no-process", *kwargs.pop("args")), **kwargs)
        return self.last_plan

    def invoke(self, plan, *, log_dir, cancel):
        self.calls += 1
        inputs = tuple(FileRecord.of(plan.cwd / name, base=plan.cwd) for name in plan.declared_inputs)
        if self.alter_input:
            (plan.cwd / "candidate.cif").write_text("changed synthetic CIF")
        outputs = ()
        if self.text is not None:
            (plan.cwd / "candidate.chk").write_text(self.text)
            outputs = (FileRecord.of(plan.cwd / "candidate.chk", base=plan.cwd),)
        record = ToolInvocationRecord(self.kind, plan.label, "fabricated-no-process", "c" * 64,
            "synthetic", "synthetic", "synthetic-only", False, plan.argv, str(plan.cwd), {}, plan.env_unset,
            "", "", plan.timeout_s, "synthetic-start", "synthetic-end", 0, 0 if self.success else 1,
            "synthetic", "no actual program ran", self.success, inputs=inputs, outputs=outputs)
        return SimpleNamespace(succeeded=self.success, cancelled=False), record


@pytest.fixture
def sealed(tmp_path):
    private = tmp_path.resolve() / "runs"
    run = private / "invented-candidate"
    run.mkdir(parents=True)
    for name, content in (("candidate.res", "synthetic sealed model"), ("candidate.cif", "data_synthetic\n"),
                          ("candidate.fcf", "data_synthetic_reflections\n")):
        (run / name).write_text(content)
    seal = seal_run(run, private_root=private, case_id="invented", track="de_novo",
                    outputs={"model": "candidate.res", "cif": "candidate.cif", "fcf": "candidate.fcf"})
    return private, run, seal


def invoke(sealed, adapter=None, **kwargs):
    private, run, seal = sealed
    adapter = adapter or FabricatedAdapter()
    toolchain = SimpleNamespace(mode=RunMode.REAL, platon_adapter=lambda: adapter)
    parameters = dict(private_root=private, expected_seal_sha256=seal.seal_sha256,
                      output_dir=run / "independent-check", toolchain=toolchain,
                      execute=True, allow_builtin_checkdef=True)
    parameters.update(kwargs)
    return run_local_checkcif(run, **parameters), adapter


def test_plan_default_does_not_execute_or_create_directory(sealed):
    result, adapter = invoke(sealed, execute=False)
    assert result.status is CheckCifStatus.NOT_EVALUATED
    assert result.candidate_available and result.invocations == 0
    assert adapter.calls == 0
    assert not (sealed[1] / "independent-check").exists()


def test_recorded_alerts_do_not_block_raw_candidate_or_claim_official_pass(sealed):
    result, adapter = invoke(sealed)
    assert result.status is CheckCifStatus.ALERTS_RECORDED
    assert result.counts["C"] == 1
    assert result.candidate_available and result.process_verified
    assert not result.scientifically_reviewed and not result.official_iucr_online_check
    assert not result.report.format_verified
    assert adapter.last_plan.env_unset == ("DISPLAY", "CHECKDEF")
    assert adapter.last_plan.argv[-2:] == ("-u", "candidate.cif")
    assert adapter.last_plan.limits.cpu_seconds is not None
    assert json.loads((result.output_directory / "result.json").read_text())["candidate_available"]
    assert (sealed[1] / "candidate.res").read_text() == "synthetic sealed model"


@pytest.mark.parametrize("text,success,status", [
    ("synthetic report without known alert grammar\n", True, CheckCifStatus.NOT_EVALUATED),
    (None, True, CheckCifStatus.FAILED), ("", True, CheckCifStatus.FAILED),
    ("PLAT001_ALERT_2_A Synthetic alert\n", False, CheckCifStatus.FAILED),
])
def test_missing_unverified_or_failed_validation_remains_independent(sealed, text, success, status):
    result, _ = invoke(sealed, FabricatedAdapter(text, success=success))
    assert result.status is status
    assert result.candidate_available


def test_no_fcf_means_no_launch_and_no_shelxl_fallback(sealed):
    private, run, _ = sealed
    # A fresh seal contains a candidate and CIF, but no declared FCF evidence.
    other = private / "missing-fcf"
    other.mkdir()
    (other / "model.res").write_text("invented model")
    (other / "model.cif").write_text("invented cif")
    seal = seal_run(other, private_root=private, case_id="invented", track="de_novo",
                    outputs={"model": "model.res", "cif": "model.cif"})
    result, adapter = invoke((private, other, seal))
    assert result.status is CheckCifStatus.NOT_EVALUATED
    assert adapter.calls == 0 and result.candidate_available


def test_invalid_seal_never_starts_validation(sealed):
    result, adapter = invoke(sealed, expected_seal_sha256="f" * 64)
    assert not result.candidate_available and adapter.calls == 0


def test_resource_policy_mock_mode_and_cancellation_block_before_launch(sealed):
    result, adapter = invoke(sealed, allow_builtin_checkdef=False)
    assert result.status is CheckCifStatus.NOT_EVALUATED and adapter.calls == 0
    cancel = CancelToken()
    cancel.cancel()
    result, adapter = invoke(sealed, cancel=cancel)
    assert result.invocations == 0 and adapter.calls == 0
    result, adapter = invoke(sealed, toolchain=SimpleNamespace(mode=RunMode.MOCK_DEMO))
    assert result.invocations == 0 and adapter.calls == 0


def test_staged_input_tampering_is_failed_but_original_candidate_survives(sealed):
    result, _ = invoke(sealed, FabricatedAdapter(alter_input=True))
    assert result.status is CheckCifStatus.FAILED
    assert result.candidate_available


def test_private_fresh_output_and_pinned_explicit_checkdef(sealed, tmp_path):
    private, run, _ = sealed
    definition = private / "synthetic-check.def"
    definition.write_text("fabricated resource, not a PLATON definition")
    pin = PinnedArtifact(definition, hashlib.sha256(definition.read_bytes()).hexdigest())
    result, _ = invoke(sealed, checkdef=pin, allow_builtin_checkdef=False)
    assert (result.output_directory / "check.def").read_bytes() == definition.read_bytes()
    again, adapter = invoke(sealed)
    assert again.invocations == 0 and adapter.calls == 0
    outside, adapter = invoke(sealed, output_dir=tmp_path.resolve() / "public-output")
    assert outside.invocations == 0 and adapter.calls == 0


def test_original_sealed_candidate_tampering_is_not_reported_available(sealed):
    adapter = FabricatedAdapter()
    original = adapter.invoke

    def corrupt(plan, **kwargs):
        answer = original(plan, **kwargs)
        (sealed[1] / "candidate.res").write_text("changed original synthetic model")
        return answer

    adapter.invoke = corrupt
    result, _ = invoke(sealed, adapter)
    assert result.status is CheckCifStatus.FAILED and not result.candidate_available
