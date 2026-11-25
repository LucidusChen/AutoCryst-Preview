"""Publication checks inspect staged synthetic bytes in disposable repositories."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[1] / "scripts/git_preflight.py"
TOKEN = "Fictional.[Private]+Case"


def git(repo, *args, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, env=env)


def stage(repo, name, data=b"plain source\n", *, env=None):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    git(repo, "add", "-f", "--", name, env=env)
    return path


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / ".private_identifiers").write_text("# fictional local list\n" + TOKEN + "\n")
    (repo / ".private_identifiers").chmod(0o600)
    stage(repo, ".gitignore", b"/.private_identifiers\n")
    stage(repo, "README.md")
    return repo


def preflight(repo, visibility="private", *, env=None):
    return subprocess.run([sys.executable, str(HELPER), "--repo", str(repo), visibility],
                          capture_output=True, text=True, env=env)


def assert_safe_output(result, *sensitive):
    output = result.stdout + result.stderr
    for token in sensitive:
        assert token.casefold() not in output.casefold()
    return output


@pytest.mark.parametrize("visibility", ["private", "public"])
def test_safe_index_passes_without_changing_index_or_objects(repository, visibility):
    repo = repository
    index = repo / ".git/index"
    before = index.read_bytes()
    objects_before = sorted(str(p.relative_to(repo)) for p in (repo / ".git/objects").rglob("*"))
    result = preflight(repo, visibility)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 staged entries" in result.stdout
    assert index.read_bytes() == before
    assert objects_before == sorted(str(p.relative_to(repo)) for p in (repo / ".git/objects").rglob("*"))
    assert not (repo / ".git/index.lock").exists()


def test_staged_private_blob_fails_even_when_worktree_was_cleaned(repository):
    path = stage(repository, "note.md", (TOKEN.swapcase() + "\n").encode())
    path.write_text("clean worktree\n")
    result = preflight(repository)
    assert result.returncode == 1
    assert "private identifier in staged blob" in assert_safe_output(result, TOKEN)


def test_staged_clean_blob_passes_despite_private_worktree_and_untracked_file(repository):
    (repository / "README.md").write_text(TOKEN)
    (repository / "untracked.txt").write_text(TOKEN)
    result = preflight(repository)
    assert result.returncode == 0, result.stdout + result.stderr


def test_literal_identifier_does_not_behave_as_a_regular_expression(repository):
    stage(repository, "literal.md", b"FictionalXPrivateCase\n")
    assert preflight(repository).returncode == 0


def test_private_path_is_rejected_without_printing_its_name(repository):
    stage(repository, "folder/" + TOKEN.swapcase() + ".md")
    result = preflight(repository)
    assert result.returncode == 1
    assert "private identifier in staged path" in assert_safe_output(result, TOKEN)


def test_nul_delimited_index_parsing_handles_spaces_tabs_and_newlines(repository):
    stage(repository, "folder/name with\ttab\nand space.md")
    assert preflight(repository).returncode == 0


@pytest.mark.parametrize("visibility", ["private", "public"])
@pytest.mark.parametrize("directory", [
    "runs", "realdata_examples", "dist", "paper_ref", "platon_instruction",
    "crystalx_engine", "toolbox", ".local-engines", ".native-engine-env",
])
def test_excluded_paths_are_rejected_for_both_visibilities(repository, directory, visibility):
    stage(repository, directory + "/text-only.md")
    result = preflight(repository, visibility)
    assert result.returncode == 1
    assert "excluded staged path" in result.stdout


def test_only_toolbox_git_placeholders_are_allowed(repository):
    stage(repository, "toolbox/.gitkeep", b"")
    stage(repository, "toolbox/.gitignore", b"*\n!.gitkeep\n!.gitignore\n")
    assert preflight(repository).returncode == 0
    stage(repository, "toolbox/.hidden-program", b"not even binary\n")
    assert preflight(repository).returncode == 1


@pytest.mark.parametrize("data", [b"text\x00payload", b"\xff\xfe\x00\x00"])
def test_binary_staged_bytes_are_rejected(repository, data):
    stage(repository, "source.txt", data)
    assert preflight(repository).returncode == 1


def test_oversized_staged_blob_is_rejected(repository):
    stage(repository, "large.txt", b"x" * (1024 * 1024 + 1))
    result = preflight(repository)
    assert result.returncode == 1
    assert "1 MiB limit" in result.stdout


@pytest.mark.parametrize("secret", [
    "gh" + "p_" + "A" * 36,
    "github_" + "pat_" + "b" * 30,
    "sk" + "-proj-" + "C" * 30,
    "AK" + "IA" + "D" * 16,
    "-----BEGIN " + "PRIVATE KEY-----",
    "API_" + "KEY=" + "E" * 32,
])
def test_possible_credentials_are_rejected_without_echoing_values(repository, secret):
    stage(repository, "config.txt", (secret + "\n").encode())
    result = preflight(repository)
    assert result.returncode == 1
    assert "possible credential" in assert_safe_output(result, secret)


@pytest.mark.parametrize("fault", ["missing", "empty", "not_ignored", "symlink"])
def test_local_identifier_list_is_required_ignored_and_regular(repository, fault):
    local = repository / ".private_identifiers"
    if fault == "missing":
        local.unlink()
    elif fault == "empty":
        local.write_text("# no actual literals\n")
    elif fault == "not_ignored":
        (repository / ".gitignore").write_text("")
    else:
        local.rename(repository / "local-list.txt")
        local.symlink_to("local-list.txt")
    result = preflight(repository)
    assert result.returncode == 1
    assert_safe_output(result, TOKEN)


def test_local_list_cannot_be_force_added(repository):
    git(repository, "add", "-f", ".private_identifiers")
    result = preflight(repository)
    assert result.returncode == 1
    assert "excluded staged path" in assert_safe_output(result, TOKEN)


def test_example_list_is_safe_and_distinct_from_actual_list(repository):
    stage(repository, ".private_identifiers.example", b"Other-Fictional-Case\n")
    assert preflight(repository).returncode == 0


@pytest.mark.parametrize("name", [".private_identifiers.bak", ".private_identifiers.local"])
def test_identifier_list_backups_cannot_be_force_added(repository, name):
    stage(repository, name, b"unlisted fictional metadata\n")
    result = preflight(repository)
    assert result.returncode == 1
    assert "excluded staged path" in result.stdout


@pytest.mark.parametrize("violation", [False, True])
def test_json_receipt_is_parseable_and_does_not_disclose_tokens(repository, violation):
    if violation:
        stage(repository, "note.md", TOKEN.encode())
    result = subprocess.run(
        [sys.executable, str(HELPER), "--repo", str(repository), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == int(violation)
    assert_safe_output(result, TOKEN)
    receipt = json.loads(result.stdout)
    assert receipt["passed"] is not violation
    assert receipt["entry_count"] == 2 + int(violation)
    assert receipt["staged_bytes"] > 0
    assert bool(receipt["violations"]) is violation
    assert receipt["scope"] == "selected_index_staged_paths_and_blobs"
    assert receipt["index_modified"] is False and receipt["remote_accessed"] is False


def test_shell_entry_preserves_visibility_and_json_flags(repository):
    shell = repository / "crystalx_core/scripts/preflight_git.sh"
    shell.parent.mkdir(parents=True)
    (repository / "scripts").mkdir()
    shutil.copyfile(HELPER, repository / "scripts/git_preflight.py")
    shutil.copyfile(HELPER.parents[1] / "crystalx_core/scripts/preflight_git.sh", shell)
    result = subprocess.run(["sh", str(shell), "public", "--json"],
                            env=dict(os.environ, PYTHON=sys.executable),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["visibility"] == "public"


def test_does_not_initialize_a_repository(tmp_path):
    result = preflight(tmp_path)
    assert result.returncode == 1
    assert not (tmp_path / ".git").exists()


def test_staged_symlink_is_refused(repository):
    (repository / "linked.md").symlink_to("README.md")
    git(repository, "add", "linked.md")
    assert preflight(repository).returncode == 1


def test_alternate_index_controls_audit_and_preserves_both_indexes(repository, tmp_path):
    repo = repository
    real_index = repo / ".git/index"
    alternate_index = tmp_path / "alternate-index"
    environment = dict(os.environ, GIT_INDEX_FILE=str(alternate_index))
    git(repo, "read-tree", "--empty", env=environment)
    stage(repo, "alternate.md", TOKEN.encode(), env=environment)
    real_before = real_index.read_bytes()
    alternate_before = alternate_index.read_bytes()
    result = preflight(repo, env=environment)
    assert result.returncode == 1
    assert "1 staged entries" in assert_safe_output(result, TOKEN)
    assert real_index.read_bytes() == real_before
    assert alternate_index.read_bytes() == alternate_before
    assert preflight(repo).returncode == 0
    assert not Path(str(alternate_index) + ".lock").exists()


def test_entire_index_is_checked_including_unchanged_head_blob(repository):
    stage(repository, "old.md", TOKEN.encode())
    git(repository, "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
        "commit", "-qm", "fictional historical fixture")
    assert not git(repository, "diff", "--cached", "--name-only").stdout
    assert preflight(repository).returncode == 1


def test_owner_private_upstream_exception_is_explicit_and_not_public(repository):
    stage(repository, 'crystalx_engine/inference_repo/model.py', b'# synthetic source\n')
    assert preflight(repository).returncode == 1
    for visibility, expected in [('private', 0), ('public', 1)]:
        result = subprocess.run([sys.executable, str(HELPER), '--repo', str(repository),
                                 visibility, '--allow-upstream-private', '--json'],
                                capture_output=True, text=True)
        assert result.returncode == expected
        assert json.loads(result.stdout)['passed'] is (expected == 0)


@pytest.mark.parametrize('name', ['weights/model.pth', 'weights/model.pt',
                                 'crystalx_core-0.5.0-source.tar.gz'])
def test_private_upstream_exception_never_admits_weights_or_archives(repository, name):
    stage(repository, 'crystalx_engine/' + name, b'synthetic excluded payload\n')
    result = subprocess.run([sys.executable, str(HELPER), '--repo', str(repository),
                             'private', '--allow-upstream-private'], capture_output=True, text=True)
    assert result.returncode == 1
