"""Codex recheck F1/F2 (2026-10-22): one reflection contract, and current archive integrity.

F1 -- the pipeline and ``verify-run`` decide which reflections the refinement may use with the same
function (:mod:`crystalx_core.app.reflections`): SHELXT's matching ``.hkl``, or the original
reflections only under the evidenced "Orientation as input" fallback. Every run here uses the
synthetic demo toolchain, whose SHELXT stand-in writes the orientation phrase itself: these tests
prove the contract between pipeline and checker, **not** that real SHELXT output has that format.

F2 -- ``verify-run`` re-hashes the retained staged INS/HKL when it runs, separately from the run's
own end-of-run observation, and does not require the original upload path to exist.

Only disposable files under pytest's temporary directories are edited.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from conftest import write_ins
from crystalx_core.app import ChemicalPriorInput, RunConfig, RunMode, RunService, SolveRequest, write_demo_case
from crystalx_core.app.acceptance import verify_run
from crystalx_core.app.demo import demo_digests
from crystalx_core.app.mock_toolchain import MockToolchain
from crystalx_core.app.reflections import (
    CELL_TOLERANCE_A,
    ReflectionSource,
    decide_from_files,
    required_shelxt_outputs,
)

BRANCH = "stages/s1-initial_solution/branches/b00-shelxt"
STEM = "crystalx_demo"


def _run(tmp_path: Path, **mock) -> Path:
    service = RunService(
        tmp_path / "runs", toolchain_factory=lambda _r: MockToolchain(step_seconds=0.02, **mock)
    )
    ins, hkl = write_demo_case(tmp_path / "demo_input")
    handle = service.submit(
        SolveRequest(
            ins_path=ins,
            hkl_path=hkl,
            prior=ChemicalPriorInput(metals=("Zn",)),
            config=RunConfig(mock_step_seconds=0.02),
            mode=RunMode.MOCK_DEMO,
            label="demo",
        )
    )
    service.wait(handle.run_id, timeout=30)
    return handle.run_dir


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def _rewrite_summary(run_dir: Path, change) -> None:
    summary = _summary(run_dir)
    change(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _check(verdict, name):
    return next(c for c in verdict.checks if c.name == name)


def _stages(run_dir: Path) -> dict[str, str]:
    return {s["name"]: s["state"] for s in json.loads((run_dir / "state.json").read_text())["stages"]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tamper(path: Path, extra: bytes) -> None:
    """Append to a disposable fixture file. Staged inputs are read-only by design, so the test
    makes its own copy writable first (as a non-root user could not otherwise change it)."""
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + extra)


# --------------------------------------------------------------------------------------
# F1 -- the reflection contract
# --------------------------------------------------------------------------------------


def test_the_matching_hkl_path_is_recorded_with_its_digest_chain(tmp_path: Path):
    run_dir = _run(tmp_path)
    lineage = _summary(run_dir)["refinement"]["reflections_lineage"]
    assert lineage["source"] == ReflectionSource.SHELXT_MATCHING.value
    assert lineage["reflections_file"] == f"{BRANCH}/{STEM}_a.hkl"
    assert lineage["reflections_sha256"] == _sha(run_dir / BRANCH / f"{STEM}_a.hkl")
    assert lineage["refinement_copy_sha256"] == lineage["reflections_sha256"]
    assert "orientation_statement" not in lineage
    verdict = verify_run(run_dir, require_real=False)
    assert verdict.accepted, [c.to_record() for c in verdict.failed]
    assert "copied unchanged and read by SHELXL" in _check(verdict, "reflection_lineage").detail


def test_the_orientation_fallback_is_accepted_by_pipeline_and_checker_alike(tmp_path: Path):
    """Codex's reproduction: no _a.hkl, 'Orientation as input' present. Was rejected by verify-run."""
    run_dir = _run(tmp_path, shelxt_write_hkl=False)
    assert set(_stages(run_dir).values()) == {"succeeded"}
    summary = _summary(run_dir)
    lineage = summary["refinement"]["reflections_lineage"]
    assert lineage["source"] == ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT.value
    assert lineage["reflections_file"] == f"{BRANCH}/{STEM}.hkl"
    assert lineage["reflections_sha256"] == summary["inputs"]["hkl"]["sha256"]  # tied to the original
    assert "Orientation as input" in lineage["orientation_statement"]
    assert lineage["statement_format_verified"] is False
    assert lineage["cell_check"]["max_length_difference_a"] == 0.0
    assert any("not yet verified against real SHELXT output" in w for w in summary["warnings"])

    verdict = verify_run(run_dir, require_real=False)
    assert verdict.accepted, [c.to_record() for c in verdict.failed]
    assert not (run_dir / BRANCH / f"{STEM}_a.hkl").exists()
    assert any("'Orientation as input' fallback" in item and "not verified" in item for item in verdict.open_items)


def test_without_an_orientation_statement_the_fallback_is_refused_everywhere(tmp_path: Path):
    run_dir = _run(tmp_path, shelxt_write_hkl=False, shelxt_orientation_statement=False)
    assert _stages(run_dir)["shelxl"] == "blocked"
    refinement = _summary(run_dir)["refinement"]
    assert "does not state that the orientation is unchanged" in refinement["reason"]
    assert refinement["reflections_decision"]["allowed"] is False
    assert not (run_dir / "refinement").exists()
    verdict = verify_run(run_dir, require_real=False)
    failed = {c.name for c in verdict.failed}
    assert {"stages_succeeded", "reflection_lineage", "shelxl_run"} <= failed


def test_a_cell_that_contradicts_the_orientation_statement_refuses_the_fallback(tmp_path: Path):
    run_dir = _run(tmp_path, shelxt_write_hkl=False, shelxt_cell=(9.0, 7.0, 11.0, 90.0, 101.0, 90.0))
    assert _stages(run_dir)["shelxl"] == "blocked"
    refinement = _summary(run_dir)["refinement"]
    assert "CELL differs from the input CELL" in refinement["reason"]
    assert refinement["reflections_decision"]["cell_check"]["max_length_difference_a"] > CELL_TOLERANCE_A
    assert not verify_run(run_dir, require_real=False).accepted


def test_a_forged_fallback_record_is_rejected_when_shelxt_wrote_matching_reflections(tmp_path: Path):
    run_dir = _run(tmp_path)

    def forge(summary):
        lineage = summary["refinement"]["reflections_lineage"]
        lineage["source"] = ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT.value
        lineage["reflections_file"] = f"{BRANCH}/{STEM}.hkl"
        lineage["reflections_sha256"] = summary["inputs"]["hkl"]["sha256"]

    _rewrite_summary(run_dir, forge)
    verdict = verify_run(run_dir, require_real=False)
    detail = _check(verdict, "reflection_lineage").detail
    assert not verdict.accepted
    assert "fallback was not permitted" in detail
    assert "they give shelxt_matching_hkl" in detail


def test_changed_refinement_reflections_are_detected(tmp_path: Path):
    run_dir = _run(tmp_path)
    _tamper(run_dir / "refinement/shelxl-01" / f"{STEM}.hkl", b"   1   0   0    1.00    1.00\n")
    verdict = verify_run(run_dir, require_real=False)
    assert [c.name for c in verdict.failed] == ["reflection_lineage"]
    assert "refinement copy" in _check(verdict, "reflection_lineage").detail


def test_a_matching_file_appearing_after_a_fallback_run_contradicts_the_record(tmp_path: Path):
    run_dir = _run(tmp_path, shelxt_write_hkl=False)
    shutil.copyfile(run_dir / BRANCH / f"{STEM}.hkl", run_dir / BRANCH / f"{STEM}_a.hkl")
    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert "do not support the recorded source" in _check(verdict, "reflection_lineage").detail


def test_a_record_without_lineage_is_not_accepted(tmp_path: Path):
    run_dir = _run(tmp_path)
    _rewrite_summary(run_dir, lambda s: s["refinement"].pop("reflections_lineage"))
    verdict = verify_run(run_dir, require_real=False)
    assert [c.name for c in verdict.failed] == ["reflection_lineage"]


def _branch(tmp_path: Path, *, solution_cell: str, statement: bool, matching: bool) -> Path:
    branch = tmp_path / "branch"
    branch.mkdir()
    write_ins(branch / "case.ins", cell=(10.0, 11.0, 12.0, 90.0, 95.5, 90.0), latt=-1, symm=("-X, 1/2+Y, -Z",))
    (branch / "case.hkl").write_text("   1   0   0   10.00    1.00\n", encoding="utf-8")
    rem = "REM <I/s> 0.200 for 4 systematic absences" + (", Orientation as input" if statement else "")
    (branch / "case_a.res").write_text(
        "\n".join(
            [
                "TITL case_a in P2(1)",
                "REM SHELXT solution in P2(1): R1 0.100, Rweak 0.050, Alpha 0.000",
                rem,
                f"CELL 0.71073 {solution_cell}",
                "ZERR 2 0.001 0.001 0.001 0 0.01 0",
                "LATT -1",
                "SYMM -X, 1/2+Y, -Z",
                "SFAC C N O Ag",
                "UNIT 24 4 8 2",
                "Ag1 4 0.1 0.2 0.3 11.0 0.05",
                "HKLF 4",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if matching:
        (branch / "case_a.hkl").write_text("   0   1   0   10.00    1.00\n", encoding="utf-8")
    return branch


@pytest.mark.parametrize(
    ("solution_cell", "statement", "matching", "expected"),
    [
        ("10 11 12 90 95.5 90", True, True, ReflectionSource.SHELXT_MATCHING),
        ("8 9 12 90 100 90", False, True, ReflectionSource.SHELXT_MATCHING),  # SHELXT re-indexed
        ("10 11 12 90 95.5 90", True, False, ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT),
        ("10.0004 11 12 90 95.504 90", True, False, ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT),
        ("10.01 11 12 90 95.5 90", True, False, None),  # cell contradicts the statement
        ("10 11 12 90 95.5 90", False, False, None),  # no statement, no matching file
    ],
)
def test_the_shared_decision(tmp_path, solution_cell, statement, matching, expected):
    decision = decide_from_files(
        _branch(tmp_path, solution_cell=solution_cell, statement=statement, matching=matching), "case", "a"
    )
    assert decision.source is expected
    if expected is ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT:
        assert decision.file_name == "case.hkl"
        assert decision.to_record()["statement_format_verified"] is False


def test_what_shelxt_must_have_written_follows_the_source():
    assert required_shelxt_outputs("shelxt_matching_hkl", "s", "a") == ("s_a.res", "s_a.hkl")
    assert required_shelxt_outputs("original_orientation_as_input", "s", "a") == ("s_a.res",)
    assert required_shelxt_outputs(None, "s", "b") == ("s_b.res", "s_b.hkl")  # strict default
    assert required_shelxt_outputs("anything else", "s", "a") == ("s_a.res", "s_a.hkl")


# --------------------------------------------------------------------------------------
# F2 -- current integrity of the retained staged inputs
# --------------------------------------------------------------------------------------


def test_a_staged_input_changed_after_the_run_fails_current_integrity_not_history(tmp_path: Path):
    """Codex's reproduction: this used to be accepted with inputs_unchanged=True."""
    run_dir = _run(tmp_path)
    staged = run_dir / _summary(run_dir)["inputs"]["hkl"]["staged"]
    _tamper(staged, b"   9   9   9    1.00    1.00\n")
    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert _check(verdict, "inputs_at_run_end").passed  # what the run saw when it ended
    current = _check(verdict, "staged_inputs_intact")
    assert not current.passed and "hkl" in current.detail and "changed since the run" in current.detail


def test_a_missing_staged_input_fails_cleanly(tmp_path: Path):
    run_dir = _run(tmp_path)
    (run_dir / _summary(run_dir)["inputs"]["ins"]["staged"]).unlink()
    verdict = verify_run(run_dir, require_real=False)
    assert "ins" in _check(verdict, "staged_inputs_intact").detail
    assert "missing" in _check(verdict, "staged_inputs_intact").detail
    assert not verdict.accepted


def test_an_unreadable_staged_input_fails_cleanly(tmp_path: Path):
    run_dir = _run(tmp_path)
    staged = run_dir / _summary(run_dir)["inputs"]["hkl"]["staged"]
    staged.unlink()
    staged.mkdir()  # a directory where the file was: unreadable as a file, even for root
    verdict = verify_run(run_dir, require_real=False)
    assert "not a readable regular file" in _check(verdict, "staged_inputs_intact").detail
    assert not verdict.accepted


def test_the_original_upload_may_be_gone(tmp_path: Path):
    run_dir = _run(tmp_path)
    shutil.rmtree(tmp_path / "demo_input")
    verdict = verify_run(run_dir, require_real=False)
    assert verdict.accepted, [c.to_record() for c in verdict.failed]
    assert "need not still exist" in _check(verdict, "inputs_at_run_end").detail


def test_both_input_roles_are_required(tmp_path: Path):
    run_dir = _run(tmp_path)
    _rewrite_summary(run_dir, lambda s: s["inputs"].pop("hkl"))
    verdict = verify_run(run_dir, require_real=False)
    assert "hkl: no staged copy and digest recorded" in _check(verdict, "staged_inputs_intact").detail
    assert not verdict.accepted


@pytest.mark.parametrize("where", ["../outside.ins", "/absolute/outside.ins"])
def test_a_recorded_input_path_outside_the_run_is_refused(tmp_path: Path, where: str):
    run_dir = _run(tmp_path)
    staged = run_dir / _summary(run_dir)["inputs"]["ins"]["staged"]
    shutil.copyfile(staged, run_dir.parent / "outside.ins")  # same bytes, wrong place

    def move(summary):
        summary["inputs"]["ins"]["staged"] = where

    _rewrite_summary(run_dir, move)
    detail = _check(verify_run(run_dir, require_real=False), "staged_inputs_intact").detail
    assert "leaves the run directory" in detail or "not relative to the run directory" in detail


def test_expected_digests_are_compared_with_the_current_staged_copies(tmp_path: Path):
    run_dir = _run(tmp_path)
    ins_digest, hkl_digest = demo_digests()
    expected = {"ins": ins_digest, "hkl": hkl_digest}
    assert verify_run(run_dir, require_real=False, expected_inputs=expected).accepted

    # replace the staged INS and rewrite the record to match: the record agrees with itself, but the
    # retained file is no longer the expected input
    staged = run_dir / _summary(run_dir)["inputs"]["ins"]["staged"]
    _tamper(staged, b"REM appended\n")

    def follow(summary):
        summary["inputs"]["ins"]["sha256"] = _sha(staged)

    _rewrite_summary(run_dir, follow)
    verdict = verify_run(run_dir, require_real=False, expected_inputs=expected)
    assert _check(verdict, "staged_inputs_intact").passed
    assert not _check(verdict, "inputs_expected").passed
    assert "staged copy now" in _check(verdict, "inputs_expected").detail
