"""Git privacy checks use disposable repositories and fictional metadata only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from crystalx_core.private_acceptance.privacy import audit_repository


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text("/runs/\n/realdata_examples/**\n")
    return repo


def test_archive_description_is_private_even_without_measurement_suffix(tmp_path):
    repo = _repository(tmp_path)
    private = repo / "realdata_examples/fictional"
    private.mkdir(parents=True)
    for name in ("case.json", "INDEX.json", "README.md", ".gitkeep", "odd.extension"):
        (private / name).write_text("{}")
        assert _git(repo, "check-ignore", "--quiet", "--", str(private / name)).returncode == 0
    assert audit_repository(repo)["current_forbidden_paths"] == []
    _git(repo, "add", "-f", "realdata_examples/fictional/case.json")
    assert audit_repository(repo)["current_forbidden_paths"] == ["realdata_examples/fictional/case.json"]


def test_untracking_preserves_files_and_reports_historical_exposure(tmp_path):
    repo = _repository(tmp_path)
    private = repo / "realdata_examples/fictional"
    private.mkdir(parents=True)
    path = private / "case.json"
    path.write_text('{"fictional":"metadata"}')
    before = path.read_bytes()
    _git(repo, "add", "-f", str(path))
    _git(repo, "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
         "commit", "-qm", "synthetic historical privacy regression")
    _git(repo, "rm", "--cached", "--", str(path))
    report = audit_repository(repo)
    assert path.read_bytes() == before
    assert not report["current_forbidden_paths"]
    assert len(report["private_archive_history_commits"]) == 1
    assert report["status"] == "REVIEW_REQUIRED"


def test_scan_reports_path_and_count_without_copying_metadata_value(tmp_path):
    repo = _repository(tmp_path)
    private = repo / "realdata_examples/fictional"
    private.mkdir(parents=True)
    secret = "Fictional sensitive metadata string for this synthetic test"
    (private / "case.json").write_text(json.dumps({"description": secret}))
    (repo / "accidental.md").write_text(secret)
    report = audit_repository(repo)
    assert report["exact_metadata_copy_candidates"] == [
        {"path": "accidental.md", "matching_metadata_string_count": 1}]
    assert secret not in json.dumps(report)


def test_audit_does_not_open_coordinate_or_measurement_files(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    private = repo / "realdata_examples/fictional"
    private.mkdir(parents=True)
    (private / "answer.res").write_text("do not open")
    (private / "input.hkl").write_text("do not open")
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in {".res", ".hkl"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert audit_repository(repo)["configured_remote_count"] == 0


def test_real_ignore_policy_contains_no_archive_allowlist():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".gitignore").read_text()
    assert "/realdata_examples/**" in text
    assert "!/realdata_examples" not in text
    script = (root / "crystalx_core/scripts/preflight_git.sh").read_text()
    helper = (root / "scripts/git_preflight.py").read_text()
    assert "scripts/git_preflight.py" in script
    assert '"realdata_examples"' in helper
    assert "manifests may remain" not in script + helper
