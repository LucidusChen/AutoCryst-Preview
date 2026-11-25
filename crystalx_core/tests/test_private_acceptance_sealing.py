"""Private seal regression tests use only invented byte strings and tmp paths."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from crystalx_core.private_acceptance.sealing import (
    SEAL_FILENAME, SealError, read_private_reference, read_sealed_output,
    seal_run, verify_seal, write_private_json,
)


@pytest.fixture
def private_run(tmp_path):
    private = tmp_path.resolve() / "ignored-runs"
    run = private / "synthetic-run"
    run.mkdir(parents=True)
    (run / "model.res").write_bytes(b"hand-authored synthetic model\n")
    (run / "stdout.txt").write_bytes(b"synthetic process log\n")
    return private, run


def seal(private, run, **kwargs):
    return seal_run(run, private_root=private, outputs={"model": "model.res", "stdout": "stdout.txt"},
                    case_id="synthetic-case", track="de_novo", **kwargs)


def test_seal_roundtrip_explicit_outputs_only(private_run):
    private, run = private_run
    (run / "unlisted.txt").write_text("never glob me")
    result = seal(private, run, completion={"exit_code": 0, "terminated": True})
    assert result.externally_pinned
    assert {f.role for f in result.files} == {"model", "stdout"}
    assert result.file_for_role("model").sha256 == hashlib.sha256(b"hand-authored synthetic model\n").hexdigest()
    assert verify_seal(run, private_root=private, expected_seal_sha256=result.seal_sha256) == result
    assert not verify_seal(run, private_root=private).externally_pinned
    assert read_sealed_output(result, private_root=private, role="model") == b"hand-authored synthetic model\n"
    assert os.stat(run / SEAL_FILENAME).st_mode & 0o777 == 0o600


def test_refuses_overwrite(private_run):
    private, run = private_run
    original = seal(private, run)
    with pytest.raises(SealError):
        seal(private, run)
    assert verify_seal(run, private_root=private).seal_sha256 == original.seal_sha256


def test_uppercase_spec_track_normalizes_to_seal_contract(private_run):
    private, run = private_run
    result = seal_run(run, private_root=private, outputs={"model": "model.res"},
                      case_id="synthetic-case", track="DE_NOVO")
    assert result.track == "de_novo"


@pytest.mark.parametrize("path", ["../answer.res", "/tmp/out.res", "dir/../../x", "a//b", "./model.res", "x\\y", "x:y"])
def test_output_path_escape_rejected(private_run, path):
    private, run = private_run
    with pytest.raises(SealError):
        seal_run(run, private_root=private, outputs={"model": path}, case_id="synthetic-case", track="de_novo")
    assert not (run / SEAL_FILENAME).exists()


def test_private_root_boundary(private_run):
    private, run = private_run
    with pytest.raises(SealError):
        seal_run(run, private_root=private / "different", outputs={"model": "model.res"}, case_id="synthetic-case", track="de_novo")
    with pytest.raises(SealError):
        seal_run(private, private_root=private, outputs={"model": "model.res"}, case_id="synthetic-case", track="de_novo")


@pytest.mark.parametrize("location", ["file", "parent", "run", "private"])
def test_symlink_components_rejected(private_run, location):
    private, run = private_run
    if location == "file":
        (run / "linked.res").symlink_to(run / "model.res")
        path = "linked.res"
    elif location == "parent":
        (run / "linked").symlink_to(run, target_is_directory=True)
        path = "linked/model.res"
    elif location == "run":
        linked = private / "linked-run"
        linked.symlink_to(run, target_is_directory=True)
        run = linked
        path = "model.res"
    else:
        linked = private.parent / "linked-private"
        linked.symlink_to(private, target_is_directory=True)
        private = linked
        run = linked / run.name
        path = "model.res"
    with pytest.raises(SealError):
        seal_run(run, private_root=private, outputs={"model": path}, case_id="synthetic-case", track="de_novo")


def test_hardlink_rejected(private_run):
    private, run = private_run
    os.link(run / "model.res", run / "model-hardlink.res")
    with pytest.raises(SealError):
        seal(private, run)


@pytest.mark.parametrize("change", ["content", "delete", "symlink", "directory"])
def test_verify_detects_changed_or_swapped_outputs(private_run, change):
    private, run = private_run
    result = seal(private, run)
    model = run / "model.res"
    if change == "content":
        model.write_bytes(b"different output")
    else:
        model.unlink()
        if change == "symlink":
            model.symlink_to(run / "stdout.txt")
        elif change == "directory":
            model.mkdir()
    with pytest.raises(SealError):
        verify_seal(run, private_root=private, expected_seal_sha256=result.seal_sha256)


def test_manifest_rewrite_rejected_by_external_pin(private_run):
    private, run = private_run
    result = seal(private, run)
    path = run / SEAL_FILENAME
    payload = json.loads(path.read_text())
    payload["terminal_state"] = "failed"
    path.write_text(json.dumps(payload))
    with pytest.raises(SealError):
        verify_seal(run, private_root=private, expected_seal_sha256=result.seal_sha256)


@pytest.mark.parametrize("edit", [
    lambda p: p.update(track=[]),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(completion={"object": {"answer": "must not parse"}}),
    lambda p: p["files"][0].update(path="../escaped"),
    lambda p: p["files"][0].update(size=True),
    lambda p: p["files"].append(p["files"][0]),
])
def test_malicious_manifest_fails_closed(private_run, edit):
    private, run = private_run
    seal(private, run)
    path = run / SEAL_FILENAME
    payload = json.loads(path.read_text())
    edit(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(SealError):
        verify_seal(run, private_root=private)


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":1}', '{"x":NaN}', 'not-json'])
def test_malformed_seal(private_run, text):
    private, run = private_run
    (run / SEAL_FILENAME).write_text(text)
    with pytest.raises(SealError):
        verify_seal(run, private_root=private)


def test_safe_private_report_only_and_exclusive(private_run):
    private, run = private_run
    path = write_private_json(run, private_root=private, relative="grade.json", payload={"synthetic": True})
    assert json.loads(path.read_text()) == {"synthetic": True}
    with pytest.raises(SealError):
        write_private_json(run, private_root=private, relative="grade.json", payload={"new": True})
    for filename in (SEAL_FILENAME, "../public.json", "outside/missing.json"):
        with pytest.raises(SealError):
            write_private_json(run, private_root=private, relative=filename, payload={})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"nested": "forbidden"}])
def test_invalid_completion(private_run, value):
    private, run = private_run
    with pytest.raises(SealError):
        seal(private, run, completion={"value": value})


def test_bounded_model_and_reference_reads(private_run):
    private, run = private_run
    result = seal(private, run)
    with pytest.raises(SealError):
        read_sealed_output(result, private_root=private, role="model", max_bytes=3)
    with pytest.raises(SealError):
        read_private_reference(run / "model.res", case_root=private / "different")
    (run / "linked.res").symlink_to(run / "model.res")
    with pytest.raises(SealError):
        read_private_reference(run / "linked.res", case_root=private)


def test_swap_during_hash_rejected(private_run, monkeypatch):
    private, run = private_run
    import crystalx_core.private_acceptance.sealing as module

    original = module.os.read
    mutated = False

    def malicious_read(fd, size):
        nonlocal mutated
        content = original(fd, size)
        if not mutated:
            mutated = True
            (run / "model.res").unlink()
            (run / "model.res").symlink_to(run / "stdout.txt")
        return content

    monkeypatch.setattr(module.os, "read", malicious_read)
    with pytest.raises(SealError):
        seal(private, run)


def test_unknown_role_and_state_rejected(private_run):
    private, run = private_run
    for outputs in ({1: "model.res", "model": "model.res"}, {"bad role": "model.res"},
                    {"a": "model.res", "b": "model.res"}, {"seal": SEAL_FILENAME}):
        with pytest.raises(SealError):
            seal_run(run, private_root=private, outputs=outputs, case_id="synthetic-case", track="de_novo")
    with pytest.raises(SealError):
        seal(private, run, terminal_state="running")
