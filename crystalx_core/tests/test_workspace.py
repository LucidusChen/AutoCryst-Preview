"""The run workspace.

Three properties are load-bearing and each has a test that would fail loudly if it regressed:

1. the user's original files are never modified;
2. nothing inside a run is ever overwritten;
3. a run that dies mid-way still leaves a readable record.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from conftest import write_hkl, write_ins
from crystalx_core.domain.candidate import (
    Candidate,
    CandidateProvenance,
    CandidateStatus,
    Lineage,
    Mutation,
    MutationKind,
)
from crystalx_core.domain.ids import RUN_ID_PATTERN, make_candidate_id, make_stage_id, slugify
from crystalx_core.domain.onion import OnionStage, StageKind, StageState
from crystalx_core.domain.result import RunResult, StopReason, StopReasonKind
from crystalx_core.errors import WorkspaceError
from crystalx_core.workspace.events import EventKind, read_events
from crystalx_core.workspace.hashing import sha256_file, sha256_object
from crystalx_core.workspace.store import RunWorkspace


@pytest.fixture
def workspace(tmp_path: Path, frozen_clock, fixed_entropy) -> RunWorkspace:
    return RunWorkspace.create(
        tmp_path / "runs", label="Ag peptide", clock=frozen_clock, entropy=fixed_entropy
    )


# --------------------------------------------------------------------------------------
# creation and identity
# --------------------------------------------------------------------------------------


def test_run_id_is_sortable_and_well_formed(workspace: RunWorkspace):
    assert RUN_ID_PATTERN.match(str(workspace.run_id))
    assert str(workspace.run_id) == "r-20260818T064500Z-1a2b3c4d"


def test_directory_name_carries_run_id_and_slug(workspace: RunWorkspace):
    assert workspace.layout.root.name == f"{workspace.run_id}_{slugify('Ag peptide')}"


def test_run_json_records_versions(workspace: RunWorkspace):
    payload = json.loads(workspace.layout.run_json.read_text())
    assert payload["run_id"] == str(workspace.run_id)
    assert payload["core_version"]
    assert payload["schema_version"] >= 1


def test_creating_over_an_existing_run_fails(tmp_path: Path, frozen_clock, fixed_entropy):
    RunWorkspace.create(tmp_path / "runs", label="x", clock=frozen_clock, entropy=fixed_entropy)
    with pytest.raises(WorkspaceError):
        RunWorkspace.create(tmp_path / "runs", label="x", clock=frozen_clock, entropy=fixed_entropy)


def test_two_runs_do_not_share_a_directory(tmp_path: Path, fixed_clock, fixed_entropy):
    first = RunWorkspace.create(tmp_path / "runs", label="a", clock=fixed_clock, entropy=fixed_entropy)
    second = RunWorkspace.create(tmp_path / "runs", label="b", clock=fixed_clock, entropy=fixed_entropy)
    assert first.layout.root != second.layout.root


# --------------------------------------------------------------------------------------
# input immutability
# --------------------------------------------------------------------------------------


def test_originals_are_untouched_and_copies_are_read_only(workspace: RunWorkspace, tmp_path: Path):
    source_dir = tmp_path / "user_data"
    source_dir.mkdir()
    ins = write_ins(source_dir / "case.ins")
    hkl = write_hkl(source_dir / "case.hkl")
    before = {p: sha256_file(p) for p in (ins, hkl)}
    before_mtimes = {p: p.stat().st_mtime for p in (ins, hkl)}

    staged_ins = workspace.stage_input("ins", ins)
    staged_hkl = workspace.stage_input("hkl", hkl)

    for path, digest in before.items():
        assert sha256_file(path) == digest, f"{path} was modified"
        assert path.stat().st_mtime == before_mtimes[path]

    for staged in (staged_ins, staged_hkl):
        assert staged.staged_path.exists()
        assert sha256_file(staged.staged_path) == before[staged.original_path]
        mode = stat.S_IMODE(staged.staged_path.stat().st_mode)
        assert mode == 0o444, f"staged copy is not read-only: {oct(mode)}"


def test_input_manifest_records_digests(workspace: RunWorkspace, tmp_path: Path):
    ins = write_ins(tmp_path / "m.ins")
    workspace.stage_input("ins", ins)
    manifest = json.loads(workspace.layout.input_manifest.read_text())
    entry = manifest["entries"][0]
    assert entry["role"] == "ins"
    assert entry["sha256"] == sha256_file(ins)
    assert entry["original_path"] == str(ins)


def test_staging_the_same_role_twice_fails(workspace: RunWorkspace, tmp_path: Path):
    ins = write_ins(tmp_path / "a.ins")
    workspace.stage_input("ins", ins)
    with pytest.raises(WorkspaceError):
        workspace.stage_input("ins", ins)


def test_staging_a_directory_fails(workspace: RunWorkspace, tmp_path: Path):
    with pytest.raises(WorkspaceError):
        workspace.stage_input("ins", tmp_path)


def test_same_filename_from_different_roles_does_not_collide(
    workspace: RunWorkspace, tmp_path: Path
):
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    a = write_ins(first_dir / "case.ins")
    b = write_ins(second_dir / "case.ins", title="different")

    staged_a = workspace.stage_input("ins", a)
    staged_b = workspace.stage_input("reference_ins", b)
    assert staged_a.staged_path != staged_b.staged_path
    assert staged_b.staged_path.exists()


# --------------------------------------------------------------------------------------
# append-only behaviour
# --------------------------------------------------------------------------------------


def test_artefacts_are_never_overwritten(workspace: RunWorkspace):
    workspace.write_text("stages/note.txt", "first")
    with pytest.raises(WorkspaceError) as excinfo:
        workspace.write_text("stages/note.txt", "second")
    assert "new iteration" in str(excinfo.value)
    assert workspace.artefact_path("stages/note.txt").read_text() == "first"


def test_writing_outside_the_workspace_is_refused(workspace: RunWorkspace):
    with pytest.raises(WorkspaceError):
        workspace.write_text("../escape.txt", "nope")


def test_opening_the_same_stage_twice_fails(workspace: RunWorkspace):
    stage = OnionStage(
        stage_id=make_stage_id(1, StageKind.INITIAL_SOLUTION.value),
        kind=StageKind.INITIAL_SOLUTION,
        index=1,
    )
    workspace.open_stage(stage)
    with pytest.raises(WorkspaceError):
        workspace.open_stage(stage)


def test_stage_state_update_is_allowed_and_logged(workspace: RunWorkspace):
    stage = OnionStage(
        stage_id=make_stage_id(1, StageKind.INITIAL_SOLUTION.value),
        kind=StageKind.INITIAL_SOLUTION,
        index=1,
    )
    workspace.open_stage(stage)
    workspace.update_stage(stage.with_state(StageState.COMPLETED, summary="two branches ran"))

    payload = json.loads(workspace.layout.stage_json(stage.stage_id).read_text())
    assert payload["state"] == "completed"
    kinds = [e["kind"] for e in read_events(workspace.layout.events)]
    assert EventKind.STAGE_STARTED in kinds
    assert EventKind.STAGE_FINISHED in kinds


# --------------------------------------------------------------------------------------
# candidate lineage
# --------------------------------------------------------------------------------------


def _provenance(parent=None, **kwargs) -> CandidateProvenance:
    return CandidateProvenance(
        created_at="2026-08-18T06:45:00+00:00",
        stage_id=make_stage_id(2, StageKind.ANCHOR.value),
        parent_id=parent,
        reflection_file_sha256="abc",
        hklf_format=4,
        **kwargs,
    )


def test_lineage_tracks_parents_and_children(workspace: RunWorkspace):
    root = Candidate(candidate_id=make_candidate_id(0), provenance=_provenance())
    workspace.add_candidate(root)

    child = Candidate(
        candidate_id=make_candidate_id(1),
        provenance=_provenance(parent=root.candidate_id),
        mutations=(
            Mutation(
                MutationKind.ELEMENT_FLIP,
                "N3 -> O3",
                atoms=("N3",),
                driver="crystalx_margin",
                margin=0.02,
            ),
        ),
    )
    workspace.add_candidate(child)

    assert workspace.lineage.roots() == (root,)
    assert workspace.lineage.children(root.candidate_id) == (child,)
    ancestry = workspace.lineage.ancestry(child.candidate_id)
    assert [str(c.candidate_id) for c in ancestry] == ["c000", "c001"]


def test_candidate_index_is_rewritten_on_every_change(workspace: RunWorkspace):
    root = Candidate(candidate_id=make_candidate_id(0), provenance=_provenance())
    workspace.add_candidate(root)
    index = json.loads(workspace.layout.candidate_index.read_text())
    assert index["count"] == 1

    workspace.update_candidate(root.with_status(CandidateStatus.REJECTED, reason="failed gate G4"))
    index = json.loads(workspace.layout.candidate_index.read_text())
    assert index["candidates"][0]["status"] == "rejected"
    assert index["candidates"][0]["rejection_reason"] == "failed gate G4"


def test_rejected_candidates_are_kept_not_deleted(workspace: RunWorkspace):
    root = Candidate(candidate_id=make_candidate_id(0), provenance=_provenance())
    workspace.add_candidate(root)
    workspace.update_candidate(root.with_status(CandidateStatus.REJECTED, reason="unbalanced charge"))
    assert workspace.layout.candidate_dir(root.candidate_id).exists()
    assert len(workspace.lineage) == 1


def test_duplicate_candidate_id_is_refused():
    lineage = Lineage()
    candidate = Candidate(candidate_id=make_candidate_id(0), provenance=_provenance())
    lineage.add(candidate)
    with pytest.raises(ValueError):
        lineage.add(candidate)


def test_unknown_parent_is_refused():
    lineage = Lineage()
    orphan = Candidate(
        candidate_id=make_candidate_id(1), provenance=_provenance(parent=make_candidate_id(0))
    )
    with pytest.raises(ValueError):
        lineage.add(orphan)


def test_status_change_is_logged_with_the_reason(workspace: RunWorkspace):
    root = Candidate(candidate_id=make_candidate_id(0), provenance=_provenance())
    workspace.add_candidate(root)
    workspace.update_candidate(root.with_status(CandidateStatus.REJECTED, reason="charge imbalance"))
    events = [e for e in read_events(workspace.layout.events) if e["kind"] == EventKind.CANDIDATE_STATUS_CHANGED]
    assert events and events[0]["reason"] == "charge imbalance"


# --------------------------------------------------------------------------------------
# crash resilience and finalisation
# --------------------------------------------------------------------------------------


def test_a_torn_event_line_does_not_make_the_log_unreadable(workspace: RunWorkspace):
    workspace.events.emit(EventKind.WARNING, "first")
    workspace.events.emit(EventKind.WARNING, "second")
    with open(workspace.layout.events, "a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026-08-18T06:45:0')  # simulate a crash mid-write

    events = list(read_events(workspace.layout.events))
    assert [e.get("message") for e in events if e["kind"] == EventKind.WARNING] == [
        "first",
        "second",
    ]


def test_partial_run_is_still_readable(tmp_path: Path, frozen_clock, fixed_entropy):
    """Everything written before an abrupt stop must remain valid on disk."""
    workspace = RunWorkspace.create(
        tmp_path / "runs", label="crash", clock=frozen_clock, entropy=fixed_entropy
    )
    ins = write_ins(tmp_path / "c.ins")
    workspace.stage_input("ins", ins)
    workspace.add_candidate(
        Candidate(candidate_id=make_candidate_id(0), provenance=_provenance())
    )
    del workspace  # no finalize() -- the process "died"

    root = next((tmp_path / "runs").iterdir())
    assert json.loads((root / "run.json").read_text())["run_id"]
    assert json.loads((root / "inputs" / "manifest.json").read_text())["entries"]
    assert json.loads((root / "candidates" / "index.json").read_text())["count"] == 1
    assert not (root / "result.json").exists()


def test_finalize_writes_result_and_closes_the_run(workspace: RunWorkspace):
    result = RunResult(
        run_id=workspace.run_id,
        stop_reason=StopReason(
            StopReasonKind.NO_TRUSTWORTHY_CANDIDATE,
            "no candidate passed the eligibility gates",
            suggested_next_action="run another shell-expansion stage",
        ),
    )
    workspace.finalize(result)

    payload = json.loads(workspace.layout.result_json.read_text())
    assert payload["stop_reason"]["kind"] == "no_trustworthy_candidate"
    assert workspace.is_finalized
    with pytest.raises(WorkspaceError):
        workspace.write_text("stages/after.txt", "too late")


# --------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------


def test_sha256_object_is_stable_across_key_order():
    assert sha256_object({"a": 1, "b": [2, 3]}) == sha256_object({"b": [2, 3], "a": 1})


def test_sha256_file_matches_content(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"hello")
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()
