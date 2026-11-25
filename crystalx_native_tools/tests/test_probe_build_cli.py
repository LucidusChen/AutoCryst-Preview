"""Launch probe (opt-in), toolchain inventory (never executes), build plans (never build), CLI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import machobuild as mb
from conftest import apple_silicon, linux_host, write_exe
from crystalx_native_tools import cli
from crystalx_native_tools.build import plan_build, scan_source
from crystalx_native_tools.config import ToolsConfig, load_config
from crystalx_native_tools.probe import ProbeRefused, ProbeStore, probe_launch
from crystalx_native_tools.toolchain import ToolchainReport, inspect_toolchain

posix = pytest.mark.skipif(os.name != "posix", reason="POSIX scripts")


def _script(path: Path, body: str) -> Path:
    return write_exe(path, ("#!/bin/sh\n" + body).encode())


# --------------------------------------------------------------------------------------
# launch probe
# --------------------------------------------------------------------------------------


def test_the_probe_refuses_without_explicit_authorisation(tmp_path: Path):
    program = _script(tmp_path / "tool", "exit 0\n")
    with pytest.raises(ProbeRefused, match="authorize"):
        probe_launch("platon_special", program, authorize=False)


def test_the_probe_refuses_a_file_that_changed_since_inspection(tmp_path: Path):
    program = _script(tmp_path / "tool", "exit 0\n")
    with pytest.raises(ProbeRefused, match="changed since it was inspected"):
        probe_launch("platon_special", program, authorize=True, expected_sha256="0" * 64)


def test_strict_mode_does_not_probe_x86_64_through_rosetta(tmp_path: Path):
    program = write_exe(tmp_path / "shelxt", mb.thin(mb.CPU_X86_64))
    with pytest.raises(ProbeRefused, match="no arm64 code"):
        probe_launch("shelxt", program, authorize=True, host=apple_silicon())


@posix
def test_a_program_that_starts_is_reported_launched_in_an_isolated_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("CHECKDEF", "/somewhere/check.def")
    program = _script(tmp_path / "tool", 'echo "display=[$DISPLAY] checkdef=[$CHECKDEF] args=$#" ; pwd\nexit 3\n')
    probe = probe_launch("platon_special", program, authorize=True, host=linux_host(), work_root=tmp_path,
                         allow_translation=True)  # a script: only outside strict native mode (review D4)
    assert probe.outcome == "launched" and probe.started and probe.exit_code == 3
    assert "display=[] checkdef=[] args=0" in probe.stdout_tail
    assert "crystalx-probe-platon_special-" in probe.stdout_tail  # ran in its own empty directory
    assert probe.to_record()["meaning"].startswith("whether the operating system started")


@posix
def test_a_program_still_running_at_the_limit_is_started_and_stopped(tmp_path: Path):
    program = _script(tmp_path / "tool", "exec sleep 30\n")
    probe = probe_launch("platon", program, authorize=True, timeout_s=1.0, host=linux_host(), work_root=tmp_path,
                         allow_translation=True)
    assert probe.outcome == "running_when_stopped" and probe.started


@posix
def test_a_file_the_system_cannot_execute_is_refused(tmp_path: Path):
    program = write_exe(tmp_path / "garbage", b"\x00\x01\x02 not an executable format")
    probe = probe_launch("shelxl", program, authorize=True, host=linux_host(), work_root=tmp_path,
                         allow_translation=True)
    assert probe.outcome == "refused" and not probe.started


@posix
def test_strict_native_mode_does_not_probe_what_it_cannot_verify(tmp_path: Path):
    """Review D4: a wrapper script (or an unknown format) could start an x86_64 program via Rosetta."""
    marker = tmp_path / "started"
    wrapper = _script(tmp_path / "platon_special", f"touch {marker}\n")
    garbage = write_exe(tmp_path / "garbage", b"\x00\x01\x02 not an executable format")
    for target, word in ((wrapper, "a script"), (garbage, "unrecognised format")):
        for host in (linux_host(), apple_silicon()):
            with pytest.raises(ProbeRefused, match=word):
                probe_launch("platon_special", target, authorize=True, host=host, work_root=tmp_path)
    assert not marker.exists()


def test_the_probe_store_keeps_records_per_exact_file(tmp_path: Path):
    store = ProbeStore(tmp_path / "state" / "probes.json")
    assert store.load() == [] and store.latest("shelxt", "a" * 64) is None
    from crystalx_native_tools.probe import LaunchProbe, PROBE_SCHEMA

    record = LaunchProbe(PROBE_SCHEMA, "shelxt", "/x", "a" * 64, ("arm64",), "t", "launched", 0, None, 0.1,
                         "ok", "exit 0", "", "", {})
    store.append(record)
    assert store.latest("shelxt", "a" * 64)["outcome"] == "launched"
    assert store.latest("shelxt", "b" * 64) is None


# --------------------------------------------------------------------------------------
# toolchain: looked at, never run
# --------------------------------------------------------------------------------------


def test_the_toolchain_inventory_never_executes_anything(tmp_path: Path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the toolchain inventory must not execute anything")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    clt = tmp_path / "CommandLineTools"
    write_exe(clt / "usr" / "bin" / "clang", b"stub")
    (clt / "SDKs" / "MacOSX26.sdk").mkdir(parents=True)
    brew = tmp_path / "brew"
    write_exe(brew / "gfortran-16", mb.thin(mb.CPU_ARM64))
    report = inspect_toolchain(apple_silicon(), clt=clt, xcode=tmp_path / "noxcode", fortran_dirs=(brew,),
                               x11_lib=tmp_path / "nolibX11")
    assert report.command_line_tools == str(clt) and report.sdks == ("MacOSX26.sdk",)
    assert [c.arches for c in report.fortran] == [("arm64",)]
    assert report.can_build_fortran_arm64
    bare = inspect_toolchain(apple_silicon(), clt=tmp_path / "none", xcode=tmp_path / "none2", fortran_dirs=(),
                             x11_lib=tmp_path / "nolibX11")
    assert not bare.can_build_fortran_arm64
    assert any("installer shims" in n for n in bare.notes)
    assert inspect_toolchain(linux_host()).to_record()["system"] == "Linux"


# --------------------------------------------------------------------------------------
# build plans: never built
# --------------------------------------------------------------------------------------


def _toolchain(ready: bool, x11: bool = True) -> ToolchainReport:
    from crystalx_native_tools.toolchain import CompilerCandidate

    fortran = (CompilerCandidate("gfortran", "/opt/homebrew/bin/gfortran", "/x", ("arm64",)),) if ready else ()
    return ToolchainReport("Darwin", "/Library/Developer/CommandLineTools" if ready else None, None, (), fortran,
                           None, None, "/opt/X11/lib/libX11.6.dylib" if x11 else None, ())


def test_without_source_a_shelx_plan_says_why_and_builds_nothing(tmp_path: Path):
    plan = plan_build("shelxl", None, toolchain=_toolchain(True))
    assert plan.recipe == "no_source" and not plan.ready and plan.executed is False
    assert any("distributed as executables" in n for n in plan.notes)


def test_the_documented_graphics_platon_recipe(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "platon.f").write_text("      PROGRAM P\n      END\n")
    (src / "xdrvr.c").write_text("int x;\n")
    plan = plan_build("platon", src, toolchain=_toolchain(True), licence_ack="free academic source", build_root=tmp_path / "builds")
    assert plan.recipe == "platon_full_x11" and plan.ready and plan.executed is False
    compile_step = plan.steps[1]
    assert compile_step.argv[:2] == ("gfortran", "-O2") and "-lX11" in compile_step.argv
    assert plan.build_dir.startswith(str(tmp_path / "builds"))
    assert not Path(plan.build_dir).exists(), "a plan must not create anything"
    blocked = plan_build("platon", src, toolchain=_toolchain(True, x11=False), licence_ack="ok", build_root=tmp_path / "b")
    assert any("XQuartz" in b for b in blocked.blockers)


def test_a_single_fortran_file_gets_a_generic_unverified_plan(tmp_path: Path):
    src = tmp_path / "special"
    src.mkdir()
    (src / "platon_special.f").write_text("      END\n")
    plan = plan_build("platon_special", src, toolchain=_toolchain(True), licence_ack="ok", build_root=tmp_path / "b")
    assert plan.recipe == "single_fortran_file"
    assert "-lX11" not in plan.steps[1].argv
    assert any("unverified" in n for n in plan.notes)


def test_plans_are_blocked_without_toolchain_licence_or_layout(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.f").write_text("x")
    (src / "b.f").write_text("y")
    write_exe(src / "shelxl", mb.thin(mb.CPU_X86_64))  # a compiled program is not source
    plan = plan_build("shelxl", src, toolchain=_toolchain(False), licence_ack=None, build_root=tmp_path / "b")
    joined = " | ".join(plan.blockers)
    assert "no recognised build layout" in joined
    assert "no usable arm64 Fortran toolchain" in joined
    assert "no licence acknowledgement" in joined
    assert plan.source.executables_found == ("shelxl",)
    assert any("never rebuilt from" in n for n in plan.notes)


def test_a_makefile_tree_uses_its_own_makefile(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "Makefile").write_text("all:\n\ttrue\n")
    (src / "x.f90").write_text("end\n")
    (src / "y.f90").write_text("end\n")
    plan = plan_build("shelxt", src, toolchain=_toolchain(True), licence_ack="ok", build_root=tmp_path / "b")
    assert plan.recipe == "makefile" and plan.steps[1].argv[0] == "make"


def test_source_scans_are_stable_and_ignore_symlinks(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.f").write_text("x")
    (src / "LICENSE").write_text("terms")
    (src / "link.f").symlink_to(tmp_path)
    first, second = scan_source(src), scan_source(src)
    assert first.digest == second.digest
    assert first.fortran == ("a.f",) and first.licence_files == ("LICENSE",)
    assert any("symlink ignored" in p for p in first.problems)


# --------------------------------------------------------------------------------------
# configuration edge cases and the command line
# --------------------------------------------------------------------------------------


def test_config_edge_cases(tmp_path: Path):
    assert load_config(tmp_path / "absent.json") == ToolsConfig(source=None)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert "not readable JSON" in load_config(bad).problems[0]
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"tools": [], "search_dirs": "x", "strict_native": "yes", "toolbox": 3}))
    problems = " | ".join(load_config(wrong).problems)
    for fragment in ("'tools' must be", "'search_dirs' must be", "'strict_native' must be", "'toolbox' must be"):
        assert fragment in problems


def test_cli_inventory_exit_codes_and_json(tmp_path: Path, monkeypatch, capsys):
    # PATH still has the resolver's documented /usr/local/bin fallback. Explicitly confine
    # discovery so an owner's installed tools cannot change this not-found contract.
    original_build = cli.build_inventory
    monkeypatch.setattr(cli, "build_inventory", lambda *a, **k: original_build(*a, search_path=[], **k))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.local.json").write_text(json.dumps({"include_olex2": False, "include_known_locations": False}))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert cli.main(["inventory", "--json"]) == 3
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "crystalx.native-tools/1" and record["pipeline"] == "blocked"
    assert cli.main(["inventory"]) == 3
    assert "BLOCKED (not_found)" in capsys.readouterr().out


def test_cli_probe_needs_authorisation(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    program = _script(tmp_path / "tool", "exit 0\n")
    assert cli.main(["probe", "platon_special", "--path", str(program)]) == 2
    assert "authorize" in capsys.readouterr().err


def test_cli_build_plan_and_toolchain(tmp_path: Path, capsys):
    assert cli.main(["build-plan", "shelxl"]) == 3
    assert "executed: no" in capsys.readouterr().out
    assert cli.main(["toolchain", "--json"]) == 0
    assert "no compiler or shim was executed" in capsys.readouterr().out
