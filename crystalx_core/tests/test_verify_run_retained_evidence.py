"""Retained evidence must still match the invocation records when a run is checked.

All runs use the synthetic toolchain, and all mutations affect disposable fixtures only.
These checks do not establish any real SHELX output format or scientific eligibility.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from crystalx_core.app.acceptance import verify_run
from crystalx_core.app.reflections import ReflectionSource, decide_from_files
from test_verify_run_contract import BRANCH, STEM, _branch, _check, _rewrite_summary, _run, _summary, _tamper


def test_a_changed_shelxt_input_hkl_is_detected_on_the_matching_output_path(tmp_path: Path):
    run_dir = _run(tmp_path)
    _tamper(run_dir / BRANCH / f"{STEM}.hkl", b"   9   9   9    1.00    1.00\n")

    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "reflection_lineage").passed


def test_a_changed_shelxt_input_ins_invalidates_the_fallback_evidence(tmp_path: Path):
    run_dir = _run(tmp_path, shelxt_write_hkl=False)
    _tamper(run_dir / BRANCH / f"{STEM}.ins", b"REM changed after SHELXT read this file\n")

    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "reflection_lineage").passed


def test_the_recorded_solution_digest_must_agree_with_the_retained_solution(tmp_path: Path):
    run_dir = _run(tmp_path)

    def change(summary):
        summary["refinement"]["reflections_lineage"]["solution_sha256"] = "0" * 64

    _rewrite_summary(run_dir, change)
    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "reflection_lineage").passed


@pytest.mark.parametrize("recorded_path", ["../outside.hkl", "inputs/unrelated.hkl"])
def test_the_recorded_refinement_copy_must_be_the_file_shelxl_read(tmp_path: Path, recorded_path: str):
    run_dir = _run(tmp_path)

    def change(summary):
        summary["refinement"]["reflections_lineage"]["refinement_copy"] = recorded_path

    _rewrite_summary(run_dir, change)
    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "reflection_lineage").passed


def test_the_recorded_refinement_copy_digest_must_agree_with_shelxl_input(tmp_path: Path):
    run_dir = _run(tmp_path)

    def change(summary):
        summary["refinement"]["reflections_lineage"]["refinement_copy_sha256"] = "0" * 64

    _rewrite_summary(run_dir, change)
    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "reflection_lineage").passed


@pytest.mark.parametrize("name", [f"{STEM}.hkl", f"{STEM}.ins", f"{STEM}_a.hkl"])
def test_reflection_evidence_may_not_resolve_outside_the_run(tmp_path: Path, name: str):
    run_dir = _run(tmp_path)
    retained = run_dir / BRANCH / name
    outside = tmp_path / f"outside-{name}"
    shutil.copyfile(retained, outside)
    retained.unlink()
    retained.symlink_to(outside)

    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "reflection_lineage").passed


def test_a_staged_input_symlink_loop_returns_a_failed_check(tmp_path: Path):
    run_dir = _run(tmp_path)
    staged = run_dir / _summary(run_dir)["inputs"]["ins"]["staged"]
    staged.unlink()
    staged.symlink_to(staged.name)

    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "staged_inputs_intact").passed


@pytest.mark.parametrize("value", ["nan", "inf"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_nonfinite_cell_length_cannot_support_the_fallback(tmp_path: Path, value: str, axis: int):
    cell = ["10", "11", "12", "90", "95.5", "90"]
    cell[axis] = value
    branch = _branch(tmp_path, solution_cell=" ".join(cell), statement=True, matching=False)

    decision = decide_from_files(branch, "case", "a")
    assert not decision.allowed


@pytest.mark.parametrize("cell", ["10.002 11 12 90 95.5 90", "10 11 12 90 95.52 90"])
def test_the_documented_cell_tolerance_includes_its_exact_boundary(tmp_path: Path, cell: str):
    branch = _branch(tmp_path, solution_cell=cell, statement=True, matching=False)

    decision = decide_from_files(branch, "case", "a")
    assert decision.source is ReflectionSource.ORIGINAL_ORIENTATION_AS_INPUT


@pytest.mark.parametrize("field", ["orientation_statement", "cell_check"])
def test_recorded_fallback_evidence_must_agree_with_the_retained_files(tmp_path: Path, field: str):
    run_dir = _run(tmp_path, shelxt_write_hkl=False)

    def change(summary):
        lineage = summary["refinement"]["reflections_lineage"]
        if field == "orientation_statement":
            lineage[field] = "REM orientation changed"
        else:
            lineage[field]["input"][0] += 1.0

    _rewrite_summary(run_dir, change)
    verdict = verify_run(run_dir, require_real=False)
    assert not verdict.accepted
    assert not _check(verdict, "reflection_lineage").passed
