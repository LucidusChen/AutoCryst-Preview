"""Startup checks use disposable virtual environments, never mutate the owner's environment."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import venv

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_runtime", REPO / "scripts/local_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    (root / "scripts").mkdir()
    for name in ("crystalx.sh", "local_runtime.py"):
        shutil.copyfile(REPO / "scripts" / name, root / "scripts" / name)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root / ".venv")
    site = next((root / ".venv/lib").glob("python*/site-packages"))
    for command, module, distribution in runtime.PACKAGES.values():
        source = root / module / "src" / module
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("VALUE = 'installed'\n")
        (site / f"__editable__.{module}-1.0.pth").write_text(str(source.parent) + "\n")
        metadata = site / f"{distribution.replace('-', '_')}-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {distribution}\nVersion: 1.0\n")
        entry = root / ".venv/bin" / command
        # A shell entry supports spaces in the test environment's path, just like pip's wrapper.
        entry.write_text(
            '#!/bin/sh\nexec "' + str(root / ".venv/bin/python") + '" -c '
            + "'import " + module + "; import json,os,sys; print(json.dumps({"
            + '"args":sys.argv[1:],"pythonpath":os.getenv("PYTHONPATH"),'
            + '"value":' + module + '.VALUE})); sys.exit(7 if "--exit-seven" in sys.argv else 0)'
            + "' \"$@\"\n"
        )
        entry.chmod(0o755)
    return root


def launch(root, *args, **kwargs):
    env = dict(os.environ, PYTHONPATH="/nonexistent/cannot-mask-install", PYTHONHOME="/nonexistent")
    return subprocess.run(["sh", str(root / "scripts/crystalx.sh"), *args],
                          capture_output=True, text=True, env=env, timeout=30, **kwargs)


def test_clean_startup_checks_installed_imports_and_all_entry_points(project):
    result = launch(project, "check")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("installed entry point OK") == 4
    assert "Local startup check passed" in result.stdout


@pytest.mark.parametrize("command", ["core", "web", "apple", "tools"])
def test_argument_forwarding_clean_environment_and_exit_status(project, command):
    result = launch(project, command, "--exit-seven", "argument with spaces")
    assert result.returncode == 7
    payload = json.loads(result.stdout)
    assert payload["args"] == ["--exit-seven", "argument with spaces"]
    assert payload["pythonpath"] is None and payload["value"] == "installed"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS file flags")
def test_hidden_flag_recurrence_is_repaired_on_every_start(project):
    site = next((project / ".venv/lib").glob("python*/site-packages"))
    outside = project / "unrelated.txt"
    outside.write_text("leave flags alone")
    unrelated = site / "unrelated-package-file.py"
    unrelated.write_text("not an import-path file")
    paths = [project / ".venv", site.parent.parent, site.parent, site, *site.glob("*.pth")]
    originals = {p: p.stat().st_flags for p in [outside, unrelated, *paths]}
    try:
        os.chflags(outside, originals[outside] | stat.UF_HIDDEN)
        os.chflags(unrelated, originals[unrelated] | stat.UF_HIDDEN)
        for _ in range(2):
            for path in paths:
                os.chflags(path, originals[path] | stat.UF_HIDDEN)
            # Demonstrate the real interpreter failure mechanism before using the supported path.
            if not (site / "crystalx_core").is_symlink():
                broken = subprocess.run([str(project / ".venv/bin/python"), "-I", "-c", "import crystalx_core"],
                                        capture_output=True, text=True)
                assert broken.returncode != 0 and "ModuleNotFoundError" in broken.stderr
            result = launch(project, "check")
            assert result.returncode == 0, result.stderr
            assert "Repaired hidden flags" in result.stderr
            assert all(not p.stat().st_flags & stat.UF_HIDDEN for p in paths)
            assert outside.stat().st_flags & stat.UF_HIDDEN
            assert unrelated.stat().st_flags & stat.UF_HIDDEN
    finally:
        for path, flags in originals.items():
            os.chflags(path, flags)


def test_external_virtual_environment_symlink_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / ".venv").symlink_to(outside, target_is_directory=True)
    with pytest.raises(runtime.RuntimeProblem, match="not a symlink"):
        runtime.repair_hidden_flags(project)


def test_external_pth_symlink_is_refused(project, tmp_path):
    outside = tmp_path / "outside.pth"
    outside.write_text("must not change\n")
    site = next((project / ".venv/lib").glob("python*/site-packages"))
    (site / "unsafe.pth").symlink_to(outside)
    with pytest.raises(runtime.RuntimeProblem, match="symlink"):
        runtime.repair_hidden_flags(project)
    assert outside.read_text() == "must not change\n"


def test_missing_environment_reports_bootstrap_without_creating_it(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(REPO / "scripts/crystalx.sh", tmp_path / "scripts/crystalx.sh")
    result = launch(tmp_path, "check")
    assert result.returncode == 2 and "bootstrap" in result.stderr
    assert not (tmp_path / ".venv").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS file flags")
def test_child_python_recovers_flags_reapplied_after_launcher_start(project):
    assert launch(project, "check").returncode == 0
    site = next((project / ".venv/lib").glob("python*/site-packages"))
    for path in site.glob("*.pth"):
        os.chflags(path, path.stat().st_flags | stat.UF_HIDDEN)
    # No wrapper or PYTHONPATH: normal installed development links protect child interpreters.
    result = subprocess.run([str(project / ".venv/bin/python"), "-I", "-c",
                             "import crystalx_core,crystalx_web,crystalx_engine_apple_silicon,crystalx_native_tools"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # They remain hidden: imports no longer depend on Python reading these .pth files.
    assert all(p.stat().st_flags & stat.UF_HIDDEN for p in site.glob("*.pth"))


def test_an_existing_user_startup_hook_is_preserved(project):
    site = next((project / ".venv/lib").glob("python*/site-packages"))
    existing = site / "sitecustomize.py"
    existing.write_text("# existing user hook\n")
    result = launch(project, "check")
    assert result.returncode == 0, result.stderr
    assert existing.read_text() == "# existing user hook\n"


def test_missing_installed_entry_is_a_controlled_failure(project):
    (project / ".venv/bin/crystalx-web").unlink()
    result = launch(project, "check")
    assert result.returncode == 2 and "missing or unreadable" in result.stderr
    assert "Traceback" not in result.stderr


def test_mismatched_editable_metadata_does_not_get_an_import_link(project):
    site = next((project / ".venv/lib").glob("python*/site-packages"))
    next(site.glob("__editable__.crystalx_core-*.pth")).write_text("/wrong/source\n")
    result = launch(project, "check")
    assert result.returncode == 2 and "does not match this project" in result.stderr
    assert not (site / "crystalx_core").exists()


def test_existing_installed_directory_is_preserved(project):
    site = next((project / ".venv/lib").glob("python*/site-packages"))
    installed = site / "crystalx_core"
    installed.mkdir()
    (installed / "__init__.py").write_text("# preserve installed package\n")
    result = launch(project, "check")
    assert result.returncode == 2 and "preserved" in result.stderr
    assert (installed / "__init__.py").read_text() == "# preserve installed package\n"
