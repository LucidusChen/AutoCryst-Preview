"""Bounded local JSON-lines bridge. Transport never invokes synchronous reconstruction."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import uuid

from crystalx_core._serde import to_jsonable
from crystalx_core.app.candidate_outputs import candidate_scene, export_candidate, pinned_candidate
from crystalx_core.app.checkcif import PinnedArtifact, run_local_checkcif
from crystalx_core.app.models import RunConfig, RunMode
from crystalx_core.app.reconstruction_backend import FRONTEND_CONTRACT, ReconstructionBackendService, ReconstructionRequest
from crystalx_core.app.reconstruction_stops import STOP_CODES
from crystalx_core.app.solve_routes import InitialRouteConfig, PhasingDataset, ProteinPhasingConfig
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.private_acceptance.sealing import SealError
from crystalx_core.refinement.adaptive import AdaptivePolicy
from crystalx_core.tools.runner import CancelToken
from crystalx_core.validation.platon import PlatonSettings
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.workflow.reconstruction import ReconstructionBlocked

from crystalx_bridge import BRIDGE_PROTOCOL

MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
OPERATIONS = ("hello", "describe", "preview", "submit", "status", "summary", "cancel", "export", "scene", "checkcif")


class BridgeError(ValueError):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)


def _canonical_parent(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Path string required")
    path = Path(value).absolute()
    # Resolve parent aliases such as /var; keep leaf aliases for core checks to reject.
    return str(path.parent.resolve(strict=True) / path.name)


def _request(value):
    if not isinstance(value, dict):
        raise ValueError("Request object required")
    record = json.loads(json.dumps(value, allow_nan=False))
    for key in ("ins_path", "hkl_path", "checkpoint"):
        if record.get(key) is not None:
            record[key] = _canonical_parent(record[key])
    if "tool_paths" in record:
        record["tool_paths"] = [(name, _canonical_parent(path)) for name, path in record["tool_paths"]]
    protein = (record.get("route") or {}).get("protein")
    if protein:
        for dataset in protein.get("datasets", []):
            dataset["path"] = _canonical_parent(dataset["path"])
    return ReconstructionRequest.from_record(record)


def _dataclass_description(cls):
    fields = []
    for f in dataclasses.fields(cls):
        default = f.default
        if default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        item = {"name": f.name, "type": str(f.type), "required": default is dataclasses.MISSING}
        if default is not dataclasses.MISSING:
            item["default"] = to_jsonable(default)
        fields.append(item)
    return fields


def describe(language):
    # These are the Web's existing non-Gradio presentation authorities, not copies.
    from crystalx_web import reconstruction as forms
    from crystalx_web import reconstruction_text as text
    from crystalx_web.i18n import LANGS
    if language not in LANGS:
        raise ValueError("Unsupported language")
    groups = {group: [{**dataclasses.asdict(spec), "label": text.field_label(spec.path, language),
                       "choice_labels": {v: text.choice_label(spec.path, v, language) for v in spec.choices}}
                      for spec in specs] for group, specs in forms.cycle_groups().items()}
    return {"language": language, "presentation_contract": forms.EXPECTED_CONTRACT,
            "presentation_review_pending": forms.EXPECTED_CONTRACT != FRONTEND_CONTRACT,
            "dataclasses": {cls.__name__: _dataclass_description(cls) for cls in
                            (ReconstructionRequest, InitialRouteConfig, ProteinPhasingConfig,
                             PhasingDataset, CyclePolicy, AdaptivePolicy)},
            "groups": groups,
            "stops": {code: {**dataclasses.asdict(info), "label": text.stop_explanation(code, language)}
                      for code, info in STOP_CODES.items()},
            "decisions": {code: text.decision_label(code, language) for code in text.DECISIONS}}


def _write_json(path, record):
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with os.fdopen(os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as stream:
        json.dump(to_jsonable(record), stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _only(params, allowed, required=()):
    if not isinstance(params, dict) or set(params) - set(allowed) or not set(required) <= set(params):
        raise BridgeError("invalid_params", "Unknown or missing operation fields")


class Bridge:
    """Own services and volatile cancellation handles; all job records are durable."""
    def __init__(self, service, *, synthetic_only=False, checkcif_toolchain_factory=None):
        self.service = service
        self.synthetic_only = synthetic_only
        self.checkcif_toolchain_factory = checkcif_toolchain_factory
        self.checks = {}  # Thread/cancel handles only; status/results live in the run folder.
        self.lock = threading.Lock()

    def _check_root(self, run_id, check_id):
        if not isinstance(check_id, str) or re.fullmatch(r"c-[a-f0-9]{32}", check_id) is None:
            raise ValueError("Invalid local diagnostic id")
        root = self.service.run_root(run_id) / "bridge" / "checkcif" / check_id
        if root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError("Unknown local diagnostic")
        return root

    def _check_status(self, run_id, check_id):
        root = self._check_root(run_id, check_id)
        record = json.loads((root / "state.json").read_text())
        if record["phase"] in ("queued", "running"):
            try:
                os.kill(record["owner_pid"], 0)
            except ProcessLookupError:
                record = {**record, "phase": "interrupted", "reason": "owner_process_interrupted_no_replay"}
        return record

    def _check_start(self, params):
        run_id = params["run_id"]
        candidate = pinned_candidate(self.service, run_id)
        timeout = params.get("timeout_s", 600.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("Diagnostic timeout must be in (0,3600]")
        allow = params.get("allow_builtin_checkdef", False)
        strict = params.get("strict_native", True)
        if type(allow) is not bool or type(strict) is not bool:
            raise ValueError("Boolean local diagnostic policies required")
        checkdef = None
        if params.get("checkdef_path") is not None:
            if not isinstance(params.get("checkdef_sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", params["checkdef_sha256"]) is None:
                raise ValueError("Explicit check.def SHA-256 required")
            checkdef = PinnedArtifact(Path(_canonical_parent(params["checkdef_path"])), params["checkdef_sha256"])
        tool_path = Path(_canonical_parent(params["platon_path"])) if params.get("platon_path") else None
        check_id = "c-" + uuid.uuid4().hex
        parent = self.service.run_root(run_id)
        for component in ("bridge", "checkcif"):
            parent = parent / component
            if parent.is_symlink():
                raise ValueError("Aliased diagnostic parent")
            parent.mkdir(exist_ok=True, mode=0o700)
            if parent.resolve() != parent:
                raise ValueError("Noncanonical diagnostic parent")
        root = parent / check_id
        root.mkdir(mode=0o700)
        cancel = CancelToken()
        initial = {"checkcif_id": check_id, "run_id": run_id, "owner_pid": os.getpid(),
                   "phase": "queued", "local_only": True, "official_iucr_online_check": False,
                   "seal_sha256": candidate.seal.seal_sha256}
        _write_json(root / "request.json", {**params, "frontend_contract": FRONTEND_CONTRACT})
        _write_json(root / "state.json", initial)

        def work():
            _write_json(root / "state.json", {**initial, "phase": "running"})
            try:
                if self.checkcif_toolchain_factory:
                    toolchain = self.checkcif_toolchain_factory(params)
                else:
                    toolchain = RealToolchain.for_config(RunConfig(toolbox_dir=self.service.repository / "toolbox",
                        strict_native=strict, tool_paths=(("platon", tool_path),) if tool_path else ()))
                result = run_local_checkcif(candidate.seal.run_root, private_root=self.service.runs_root,
                    expected_seal_sha256=candidate.seal.seal_sha256, output_dir=root / "outputs",
                    toolchain=toolchain, settings=PlatonSettings(timeout_s=float(timeout)), checkdef=checkdef,
                    allow_builtin_checkdef=allow, execute=True, cancel=cancel).to_record()
                _write_json(root / "result.json", result)
                _write_json(root / "state.json", {**initial, "phase": "cancelled" if cancel.cancelled else "completed", "result": result})
            except Exception as error:
                _write_json(root / "state.json", {**initial, "phase": "failed", "error_type": type(error).__name__})
            finally:
                with self.lock:
                    self.checks.pop((run_id, check_id), None)

        thread = threading.Thread(target=work, name="autocryst-local-checkcif-" + check_id, daemon=True)
        with self.lock:
            self.checks[(run_id, check_id)] = (thread, cancel)
        thread.start()
        return initial

    def dispatch(self, op, params):
        if op == "hello":
            _only(params, ())
            return {"operations": OPERATIONS, "max_request_bytes": MAX_REQUEST_BYTES,
                    "max_response_bytes": MAX_RESPONSE_BYTES, "transport": "local-json-lines",
                    "synthetic_only": self.synthetic_only, "automatic_replay": False,
                    "scientifically_validated": False}
        if op == "describe":
            _only(params, ("language",))
            return describe(params.get("language", "en"))
        if op in ("preview", "submit"):
            _only(params, ("frontend_contract", "request"), ("frontend_contract", "request"))
            if params["frontend_contract"] != FRONTEND_CONTRACT:
                raise BridgeError("contract_mismatch", "Frontend must review and accept the current backend contract")
            request = _request(params["request"])
            if self.synthetic_only and request.mode is not RunMode.MOCK_DEMO:
                raise BridgeError("mode_mismatch", "This helper accepts built-in synthetic demo inputs only")
            if op == "preview":
                return self.service.preview(request)
            handle = self.service.submit(request)
            return {"run_id": handle.run_id}
        if op in ("status", "summary", "cancel"):
            _only(params, ("run_id",), ("run_id",))
            self.service.run_root(params["run_id"])
            if op == "cancel":
                return {"requested": self.service.cancel(params["run_id"]), "terminal_observed": False}
            return getattr(self.service, op)(params["run_id"])
        if op == "export":
            _only(params, ("run_id", "destination"), ("run_id", "destination"))
            return export_candidate(self.service, params["run_id"], Path(_canonical_parent(params["destination"])))
        if op == "scene":
            _only(params, ("run_id", "max_sites"), ("run_id",))
            return candidate_scene(self.service, params["run_id"], max_sites=params.get("max_sites", 1000))
        if op == "checkcif":
            action = params.get("action")
            if action in ("status", "cancel"):
                _only(params, ("action", "run_id", "checkcif_id"), ("action", "run_id", "checkcif_id"))
                state = self._check_status(params["run_id"], params["checkcif_id"])
                if action == "status":
                    return state
                with self.lock:
                    job = self.checks.get((params["run_id"], params["checkcif_id"]))
                    if job:
                        job[1].cancel("cancelled by local frontend")
                return {"requested": bool(job), "terminal_observed": state["phase"] not in ("queued", "running")}
            _only(params, ("action", "run_id", "timeout_s", "allow_builtin_checkdef", "checkdef_path",
                           "checkdef_sha256", "platon_path", "strict_native"), ("action", "run_id"))
            if action != "start":
                raise ValueError("Unknown local diagnostic action")
            return self._check_start(params)
        raise BridgeError("unknown_operation", "Unsupported operation")

    def handle(self, message):
        request_id = message.get("id") if isinstance(message, dict) else None
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128 or any(ord(c) < 32 for c in request_id):
            request_id = None
        base = {"protocol": BRIDGE_PROTOCOL, "frontend_contract": FRONTEND_CONTRACT, "id": request_id}
        try:
            _only(message, ("protocol", "id", "op", "params"), ("protocol", "id", "op", "params"))
            if message["protocol"] != BRIDGE_PROTOCOL:
                raise BridgeError("protocol_mismatch", "Unsupported bridge protocol")
            if request_id is None or not isinstance(message["op"], str) or not isinstance(message["params"], dict):
                raise BridgeError("invalid_request", "A request id, operation and params object are required")
            result = to_jsonable(self.dispatch(message["op"], message["params"]))
            return {**base, "ok": True, "result": result}
        except BridgeError as error:
            return {**base, "ok": False, "error": {"code": error.code, "message": error.message}}
        except SealError:
            code, text = "candidate_unavailable", "Pinned candidate unavailable or changed"
        except ReconstructionBlocked:
            code, text = "blocked", "Backend prerequisites or scientific guards blocked this operation"
        except (ValueError, TypeError, KeyError, FileNotFoundError):
            code, text = "invalid_params", "Invalid operation fields, inputs or run id"
        except OSError:
            code, text = "local_io_error", "A local file operation failed"
        except Exception:
            code, text = "internal_error", "Local helper failed; existing run evidence is preserved"
        return {**base, "ok": False, "error": {"code": code, "message": text}}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def serve(source, destination, bridge):
    """Binary streams; oversize frames are drained before accepting the next line."""
    while True:
        line = source.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST_BYTES:
            while line and not line.endswith(b"\n"):
                line = source.readline(MAX_REQUEST_BYTES + 1)
            response = {"protocol": BRIDGE_PROTOCOL, "frontend_contract": FRONTEND_CONTRACT, "id": None,
                        "ok": False, "error": {"code": "frame_too_large", "message": "Request exceeds transport bound"}}
        else:
            try:
                message = json.loads(line.decode("utf-8"), object_pairs_hook=_unique,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
                response = bridge.handle(message)
            except (ValueError, UnicodeError, RecursionError):
                response = {"protocol": BRIDGE_PROTOCOL, "frontend_contract": FRONTEND_CONTRACT, "id": None,
                            "ok": False, "error": {"code": "invalid_json", "message": "A finite unique-key UTF-8 JSON object is required"}}
        try:
            data = json.dumps(response, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        except (ValueError, RecursionError):
            data = b""
        if not data or len(data) > MAX_RESPONSE_BYTES:
            response = {"protocol": BRIDGE_PROTOCOL, "frontend_contract": FRONTEND_CONTRACT, "id": response.get("id"),
                        "ok": False, "error": {"code": "response_too_large", "message": "Response cannot fit transport bound"}}
            data = json.dumps(response).encode() + b"\n"
        destination.write(data)
        destination.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="AutoCryst local JSON-lines helper")
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args(argv)
    # Reserve the original stdout pipe; prints from model/native worker threads go to stderr.
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    kwargs = {}
    if args.synthetic_demo:
        from crystalx_core.app.reconstruction_synthetic import toolchain_factory, assigner_factory
        kwargs = {"toolchain_factory": toolchain_factory, "assigner_factory": assigner_factory}
    service = ReconstructionBackendService(args.repository, args.runs_root, **kwargs)
    serve(sys.stdin.buffer, output, Bridge(service, synthetic_only=args.synthetic_demo))


if __name__ == "__main__":
    main()
