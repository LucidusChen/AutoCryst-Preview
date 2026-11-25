import dataclasses
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from crystalx_bridge import BRIDGE_PROTOCOL
from crystalx_bridge.helper import Bridge, MAX_REQUEST_BYTES, describe, serve
from crystalx_core.app.demo import write_demo_case
from crystalx_core.app.models import RunMode
from crystalx_core.app.reconstruction_backend import FRONTEND_CONTRACT, ReconstructionBackendService, ReconstructionRequest
from crystalx_core.app.reconstruction_synthetic import toolchain_factory, assigner_factory
from crystalx_core.app.candidate_outputs import candidate_scene, pinned_candidate
from crystalx_core.private_acceptance.sealing import SealError
from crystalx_core.workflow.refinement_cycle import CyclePolicy


def call(bridge, op, **params):
    return bridge.handle({"protocol": BRIDGE_PROTOCOL, "id": "request-1", "op": op, "params": params})


def wait_for(function, done):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        result = function()
        if done(result):
            return result
        time.sleep(.01)
    pytest.fail("Synthetic worker did not reach expected state")


@pytest.fixture
def demo(tmp_path):
    repo = tmp_path.resolve() / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/runs/\n")
    ins, hkl = write_demo_case(repo / "runs/inputs")
    request = ReconstructionRequest(ins, hkl, mode=RunMode.MOCK_DEMO, cycle=CyclePolicy(iterations=2))
    service = ReconstructionBackendService(repo, repo / "runs/reconstruction",
        toolchain_factory=toolchain_factory, assigner_factory=assigner_factory)
    bridge = Bridge(service, synthetic_only=True, checkcif_toolchain_factory=lambda _: toolchain_factory(request))
    return repo, request, service, bridge


def complete(demo):
    _, request, _, bridge = demo
    response = call(bridge, "submit", frontend_contract=FRONTEND_CONTRACT, request=request.to_record())
    assert response["ok"], response
    run_id = response["result"]["run_id"]
    status = wait_for(lambda: call(bridge, "status", run_id=run_id),
                      lambda r: r["ok"] and r["result"]["phase"] not in ("queued", "running"))
    assert status["result"]["phase"] == "completed", status
    return run_id


def test_hello_and_version_guard(demo):
    *_, bridge = demo
    hello = call(bridge, "hello")
    assert hello["protocol"] == BRIDGE_PROTOCOL and hello["frontend_contract"] == FRONTEND_CONTRACT
    assert hello["result"]["automatic_replay"] is False
    response = bridge.handle({"protocol": "unknown", "id": "v", "op": "submit", "params": {}})
    assert response["error"]["code"] == "protocol_mismatch"
    assert not list(bridge.service.runs_root.glob("r-*"))


@pytest.mark.parametrize("language", ["zh", "ja", "en"])
def test_describe_reuses_every_web_field_group_label_and_stop(language):
    from crystalx_web import reconstruction as web
    from crystalx_web import reconstruction_text as text
    from crystalx_core.app.reconstruction_stops import STOP_CODES
    record = describe(language)
    for group, fields in web.cycle_groups().items():
        assert [s.path for s in fields] == [s["path"] for s in record["groups"][group]]
        for s in record["groups"][group]:
            assert s["label"] == text.field_label(s["path"], language)
    assert set(record["stops"]) == set(STOP_CODES)
    assert all(record["stops"][code]["label"] == text.stop_explanation(code, language) for code in STOP_CODES)
    fields = record["dataclasses"]["ReconstructionRequest"]
    assert {f["name"] for f in fields} == {f.name for f in dataclasses.fields(ReconstructionRequest)}
    assert next(f for f in fields if f["name"] == "device")["default"] == "auto"


def test_presentation_import_has_no_gradio_or_torch():
    code = "import sys; from crystalx_bridge.helper import describe; describe('en'); assert not any(k=='gradio' or k.startswith('gradio.') or k=='torch' for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


@pytest.mark.parametrize("change,code", [
    ({"frontend_contract": "historic"}, "contract_mismatch"),
    ({"unexpected": True}, "invalid_params"),
])
def test_preview_rejects_unknown_fields_and_contract_before_submit(demo, change, code):
    _, request, service, bridge = demo
    params = {"frontend_contract": FRONTEND_CONTRACT, "request": request.to_record(), **change}
    assert call(bridge, "preview", **params)["error"]["code"] == code
    assert not list(service.runs_root.glob("r-*"))


def test_references_cannot_enter_request_and_demo_is_pinned(demo):
    _, request, service, bridge = demo
    record = {**request.to_record(), "expert_res": "forbidden.res"}
    assert call(bridge, "submit", frontend_contract=FRONTEND_CONTRACT, request=record)["error"]["code"] == "invalid_params"
    request.hkl_path.write_text(request.hkl_path.read_text() + "1 2 3 4 5\n")
    response = call(bridge, "preview", frontend_contract=FRONTEND_CONTRACT, request=request.to_record())
    assert not response["ok"] or response["result"]["phase"] == "blocked"
    assert not list(service.runs_root.glob("r-*"))


def test_parent_alias_canonicalized_but_leaf_alias_stays_rejected(demo):
    repo, request, _, bridge = demo
    parent = repo / "runs/alias"
    parent.symlink_to(request.ins_path.parent, target_is_directory=True)
    record = {**request.to_record(), "ins_path": str(parent / request.ins_path.name), "hkl_path": str(parent / request.hkl_path.name)}
    assert call(bridge, "preview", frontend_contract=FRONTEND_CONTRACT, request=record)["result"]["ready_static"]
    leaf = request.ins_path.with_name("leaf.ins")
    leaf.symlink_to(request.ins_path)
    record["ins_path"] = str(leaf)
    response = call(bridge, "preview", frontend_contract=FRONTEND_CONTRACT, request=record)
    assert not response["ok"] or not response["result"]["ready_static"]


def test_async_submit_cancel_and_responsiveness(demo):
    repo, request, _, _ = demo
    entered, release = threading.Event(), threading.Event()
    def factory(req):
        actual = assigner_factory(req)
        class Delayed:
            def assign(self, *args, **kwargs):
                entered.set()
                assert release.wait(5)
                return actual.assign(*args, **kwargs)
        return Delayed()
    service = ReconstructionBackendService(repo, repo / "runs/async", toolchain_factory=toolchain_factory, assigner_factory=factory)
    bridge = Bridge(service, synthetic_only=True)
    response = call(bridge, "submit", frontend_contract=FRONTEND_CONTRACT, request=request.to_record())
    assert response["ok"] and entered.wait(4)
    run_id = response["result"]["run_id"]
    try:
        assert call(bridge, "hello")["ok"]
        assert call(bridge, "status", run_id=run_id)["ok"]
        result = call(bridge, "cancel", run_id=run_id)["result"]
        assert result == {"requested": True, "terminal_observed": False}
    finally:
        release.set()
    final = wait_for(lambda: service.status(run_id), lambda r: r["phase"] not in ("queued", "running"))
    assert final["phase"] == "cancelled"


def test_summary_status_restart_and_cancel_no_replay(demo):
    repo, _, service, bridge = demo
    run_id = complete(demo)
    before = sorted(str(v.relative_to(service.runs_root)) for v in service.runs_root.rglob("invocation.json"))
    again = Bridge(ReconstructionBackendService(repo, service.runs_root), synthetic_only=True)
    assert call(again, "status", run_id=run_id)["result"]["phase"] == "completed"
    summary = call(again, "summary", run_id=run_id)["result"]
    assert summary["mode"] == "mock_demo" and summary["scientifically_validated"] is False
    assert call(again, "cancel", run_id=run_id)["result"]["requested"] is False
    assert before == sorted(str(v.relative_to(service.runs_root)) for v in service.runs_root.rglob("invocation.json"))


def test_export_and_scene_use_unchanged_pin_and_core_viewer(demo):
    repo, _, service, bridge = demo
    run_id = complete(demo)
    candidate = pinned_candidate(service, run_id)
    original = candidate.read("model")
    response = call(bridge, "export", run_id=run_id, destination=str(repo / "exported"))
    assert response["ok"] and not response["result"]["missing_roles"]
    model = next(v for v in response["result"]["outputs"] if v["role"] == "model")
    assert Path(model["path"]).read_bytes() == original
    assert call(bridge, "export", run_id=run_id, destination=str(repo / "exported"))["ok"] is False
    scene = call(bridge, "scene", run_id=run_id)["result"]
    from crystalx_core.geometry.model import build_viewer_payload
    from crystalx_core.io.shelx_res import read_res
    expected = build_viewer_payload(read_res(model["path"]), stage="refined", source_label="Sealed unvalidated candidate")
    # Temporary source filenames differ, geometry and the payload authority are identical.
    assert scene["payload"] == expected
    assert scene["seal_sha256"] == candidate.seal.seal_sha256
    assert candidate.read("model") == original
    assert not list(service.runs_root.glob(".scene-*"))
    assert call(bridge, "scene", run_id=run_id, max_sites=1)["ok"] is False


@pytest.mark.parametrize("tamper", ["model", "pin", "pin_symlink", "pin_hardlink", "phase"])
def test_seal_and_pin_tampering_block_before_export(demo, tamper):
    repo, _, service, bridge = demo
    run_id = complete(demo)
    root = service.run_root(run_id)
    pin = service.runs_root / f"{run_id}.seal-pin.json"
    if tamper == "model":
        (root / "outputs/ai_solved.res").write_text("changed")
    elif tamper == "pin":
        data = json.loads(pin.read_text());data["seal_sha256"] = "0" * 64;pin.write_text(json.dumps(data))
    elif tamper == "phase":
        state = root / "state.json";data = json.loads(state.read_text());data["phase"] = "cancelled";data["stop_code"] = "cancelled";state.write_text(json.dumps(data))
    else:
        backup = pin.with_suffix(".backup");pin.rename(backup)
        if tamper == "pin_symlink":pin.symlink_to(backup)
        else:os.link(backup, pin)
    response = call(bridge, "export", run_id=run_id, destination=str(repo / "not-created"))
    assert response["error"]["code"] == "candidate_unavailable"
    assert not (repo / "not-created").exists()
    assert call(bridge, "scene", run_id=run_id)["error"]["code"] == "candidate_unavailable"


def test_cif_absence_is_explicit_not_regenerated(demo):
    _, _, service, bridge = demo
    run_id = complete(demo)
    # A separately created RES-only terminal seal exercises supported archive absence.
    from crystalx_core.private_acceptance.sealing import seal_run
    root = service.run_root(run_id)
    pin = service.runs_root / f"{run_id}.seal-pin.json"
    (root / ".acceptance-seal.json").unlink()
    seal = seal_run(root, private_root=service.runs_root, case_id="synthetic-candidate", track="de_novo",
                    terminal_state="completed", outputs={"model": "outputs/ai_solved.res"})
    pin.write_text(json.dumps({"run_root": str(root), "seal_sha256": seal.seal_sha256}))
    assert call(bridge, "scene", run_id=run_id)["ok"]
    response = call(bridge, "checkcif", action="start", run_id=run_id, allow_builtin_checkdef=True)
    assert response["ok"]
    check_id = response["result"]["checkcif_id"]
    status = wait_for(lambda: call(bridge, "checkcif", action="status", run_id=run_id, checkcif_id=check_id),
                      lambda r: r["ok"] and r["result"]["phase"] not in ("queued", "running"))
    assert status["result"]["result"]["status"] == "NOT_EVALUATED"
    assert status["result"]["result"]["invocations"] == 0
    assert "sealed_cif_and_fcf_required_no_uncontrolled_shelxl_regeneration" in status["result"]["result"]["reasons"]


def test_local_checkcif_remains_separate_durable_and_cancellable(demo):
    repo, _, service, bridge = demo
    run_id = complete(demo)
    response = call(bridge, "checkcif", action="start", run_id=run_id, allow_builtin_checkdef=True)
    assert response["ok"]
    check_id = response["result"]["checkcif_id"]
    state = wait_for(lambda: call(bridge, "checkcif", action="status", run_id=run_id, checkcif_id=check_id),
                     lambda r: r["ok"] and r["result"]["phase"] not in ("queued", "running"))
    assert state["result"]["result"]["status"] == "NOT_EVALUATED"
    assert state["result"]["result"]["invocations"] == 0  # Synthetic tools cannot validate.
    restarted = Bridge(ReconstructionBackendService(repo, service.runs_root))
    assert call(restarted, "checkcif", action="status", run_id=run_id, checkcif_id=check_id) == state
    cancelled = call(restarted, "checkcif", action="cancel", run_id=run_id, checkcif_id=check_id)
    assert cancelled["result"] == {"requested": False, "terminal_observed": True}


def test_cancel_active_checkcif_does_not_claim_terminal_before_worker_cleanup(demo):
    _, _, _, bridge = demo
    run_id = complete(demo)
    old = bridge.checkcif_toolchain_factory
    entered, release = threading.Event(), threading.Event()
    def delayed(params):
        entered.set()
        assert release.wait(5)
        return old(params)
    bridge.checkcif_toolchain_factory = delayed
    start = call(bridge, "checkcif", action="start", run_id=run_id, allow_builtin_checkdef=True)
    check_id = start["result"]["checkcif_id"]
    assert entered.wait(3)
    try:
        assert call(bridge, "hello")["ok"]
        cancel = call(bridge, "checkcif", action="cancel", run_id=run_id, checkcif_id=check_id)
        assert cancel["result"] == {"requested": True, "terminal_observed": False}
    finally:
        release.set()
    final = wait_for(lambda: call(bridge, "checkcif", action="status", run_id=run_id, checkcif_id=check_id),
                     lambda r: r["ok"] and r["result"]["phase"] == "cancelled")
    assert final["result"]["result"]["invocations"] == 0


def test_killed_helper_leaves_interrupted_evidence_and_never_replays(demo):
    repo, request, service, _ = demo
    record = {**request.to_record(), "cycle": CyclePolicy(iterations=5000, max_rounds=5000).to_dict()}
    message = {"protocol": BRIDGE_PROTOCOL, "id": "crash", "op": "submit",
               "params": {"frontend_contract": FRONTEND_CONTRACT, "request": record}}
    process = subprocess.Popen([sys.executable, "-m", "crystalx_bridge.helper", "--repository", str(repo),
                                "--runs-root", str(service.runs_root), "--synthetic-demo"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        process.stdin.write(json.dumps(message) + "\n");process.stdin.flush()
        response = json.loads(process.stdout.readline())
        assert response["ok"], response
        run_id = response["result"]["run_id"]
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=5)
        process.stdin.close();process.stdout.close();process.stderr.close()
    before = sorted(str(p) for p in service.runs_root.rglob("invocation.json"))
    restored = Bridge(ReconstructionBackendService(repo, service.runs_root))
    assert call(restored, "status", run_id=run_id)["result"]["phase"] == "interrupted"
    assert call(restored, "cancel", run_id=run_id)["result"]["requested"] is False
    assert sorted(str(p) for p in service.runs_root.rglob("invocation.json")) == before


@pytest.mark.parametrize("raw", [b"{\n", b'{}\n', b'[]\n', b'{"id":"x","id":"y"}\n', b'{"value":NaN}\n', b'\xff\n'])
def test_invalid_frames_do_not_desynchronize(raw, demo):
    *_, bridge = demo
    hello = {"protocol": BRIDGE_PROTOCOL, "id": "你好", "op": "hello", "params": {}}
    target = io.BytesIO()
    serve(io.BytesIO(raw + json.dumps(hello, ensure_ascii=False).encode() + b"\n"), target, bridge)
    frames = [json.loads(v) for v in target.getvalue().splitlines()]
    assert len(frames) == 2 and not frames[0]["ok"] and frames[1]["ok"] and frames[1]["id"] == "你好"


def test_oversized_frame_drains_and_keeps_following_request(demo):
    *_, bridge = demo
    hello = {"protocol": BRIDGE_PROTOCOL, "id": "after", "op": "hello", "params": {}}
    target = io.BytesIO()
    serve(io.BytesIO(b"x" * (MAX_REQUEST_BYTES * 2) + b"\n" + json.dumps(hello).encode() + b"\n"), target, bridge)
    frames = [json.loads(v) for v in target.getvalue().splitlines()]
    assert frames[0]["error"]["code"] == "frame_too_large" and frames[1]["ok"]


def test_unknown_operation_and_generated_initial_are_never_exposed(demo):
    *_, bridge = demo
    assert call(bridge, "run_from_generated_initial")["error"]["code"] == "unknown_operation"
    assert call(bridge, "hello", extra=1)["error"]["code"] == "invalid_params"


def test_cli_stdout_is_only_protocol_and_crash_is_not_replayed(demo):
    repo, _, service, _ = demo
    message = {"protocol": BRIDGE_PROTOCOL, "id": "cli", "op": "hello", "params": {}}
    result = subprocess.run([sys.executable, "-m", "crystalx_bridge.helper", "--repository", str(repo),
                             "--runs-root", str(service.runs_root), "--synthetic-demo"],
                            input=json.dumps(message) + "\n", capture_output=True, text=True, timeout=10, check=True)
    assert len(result.stdout.splitlines()) == 1 and json.loads(result.stdout)["ok"]
    # Durable interrupted state is read, never turned into a new calculation.
    run_id = complete(demo)
    root = service.run_root(run_id)
    state_path = root / "state.json";state = json.loads(state_path.read_text());state.update(phase="running", owner_pid=2147483647);state_path.write_text(json.dumps(state))
    restored = Bridge(ReconstructionBackendService(repo, service.runs_root))
    assert call(restored, "status", run_id=run_id)["result"]["phase"] == "interrupted"
    assert not call(restored, "scene", run_id=run_id)["ok"]
    assert len(list(service.runs_root.glob("r-*_*/request.json"))) <= 1
