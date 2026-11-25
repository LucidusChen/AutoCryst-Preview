"""The generic external-tool adapter and its provenance record.

What must be true of every invocation of a crystallographic program:

* the executable is pinned by digest and architecture, not just by name;
* the inputs are hashed *before* the program runs;
* everything the program wrote is discovered and hashed, including files nobody asked for -- SHELXT
  and SHELXE write outputs whose names depend on what they found, and recording only the expected
  ones would lose the alternative solutions;
* a tool that cannot be used produces the resolver's diagnosis, not a confusing failure later.

No crystallographic binary is executed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import CPU_ARM64, CPU_X86_64, elf, macho_thin, write_executable
from crystalx_core.domain.tools import (
    BinaryArch,
    HostPlatform,
    ToolDiagnosticKind,
    ToolKind,
)
from crystalx_core.errors import ToolResolutionError
from crystalx_core.solvers.mock import ScriptedExecutor, ScriptedResponse
from crystalx_core.tools.adapter import ExternalToolAdapter, snapshot_directory
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.tools.runner import ProcessRunner
from crystalx_core.workspace.hashing import sha256_file

LINUX = HostPlatform("Linux", BinaryArch.X86_64)
ARM_NO_ROSETTA = HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=False)
ARM_WITH_ROSETTA = HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=True)


def resolve(toolbox: Path, kind: ToolKind, host=LINUX):
    return ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=host,
        environ={"PATH": ""},
    ).resolve(kind)


def adapter_for(toolbox: Path, kind: ToolKind, executor, host=LINUX) -> ExternalToolAdapter:
    return ExternalToolAdapter(
        resolve(toolbox, kind, host), runner=ProcessRunner(executor, base_env={})
    )


# --------------------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------------------


def test_an_unusable_tool_is_refused_at_construction_with_the_diagnosis(toolbox: Path):
    """An x86_64 SHELXT on Apple Silicon without Rosetta must fail here, clearly, not later."""
    write_executable(toolbox / "shelxt", macho_thin(CPU_X86_64))
    with pytest.raises(ToolResolutionError) as excinfo:
        ExternalToolAdapter(resolve(toolbox, ToolKind.SHELXT, ARM_NO_ROSETTA))
    assert "Rosetta" in str(excinfo.value)
    assert excinfo.value.diagnostic == ToolDiagnosticKind.ROSETTA_MISSING.value


def test_a_missing_tool_is_refused_at_construction(toolbox: Path):
    with pytest.raises(ToolResolutionError):
        ExternalToolAdapter(resolve(toolbox, ToolKind.SHELXL))


def test_translation_is_recorded_rather_than_hidden(toolbox: Path):
    write_executable(toolbox / "shelxl", macho_thin(CPU_X86_64))
    adapter = adapter_for(toolbox, ToolKind.SHELXL, ScriptedExecutor(), ARM_WITH_ROSETTA)
    assert adapter.required_translation
    assert adapter.describe().kind is ToolDiagnosticKind.OK_VIA_TRANSLATION


# --------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------


def test_the_plan_pins_the_executable_and_its_kind(toolbox: Path, tmp_path: Path):
    executable = write_executable(toolbox / "shelxt", elf(0x3E))
    adapter = adapter_for(toolbox, ToolKind.SHELXT, ScriptedExecutor())
    plan = adapter.build_plan(args=("case",), cwd=tmp_path, declared_inputs=("case.ins",))

    assert plan.argv == (str(executable), "case")
    assert plan.tool_kind is ToolKind.SHELXT
    assert plan.tool_sha256 == sha256_file(executable)
    assert plan.declared_inputs == ("case.ins",)


def test_a_graphical_tool_is_planned_headless(toolbox: Path, tmp_path: Path):
    """PLATON opens an X11 window by default; a headless job must not inherit DISPLAY."""
    write_executable(toolbox / "platon", elf(0x3E))
    adapter = adapter_for(toolbox, ToolKind.PLATON, ScriptedExecutor())
    plan = adapter.build_plan(args=("-u", "case.cif"), cwd=tmp_path)
    assert "DISPLAY" in plan.env_unset


def test_the_graphics_free_platon_build_does_not_need_the_display_removed(
    toolbox: Path, tmp_path: Path
):
    write_executable(toolbox / "platon_special", elf(0x3E))
    adapter = adapter_for(toolbox, ToolKind.PLATON_SPECIAL, ScriptedExecutor())
    plan = adapter.build_plan(args=("-u", "case.cif"), cwd=tmp_path)
    assert "DISPLAY" not in plan.env_unset


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------


def test_the_record_pins_the_executable(toolbox: Path, tmp_path: Path):
    executable = write_executable(toolbox / "shelxt", elf(0x3E))
    adapter = adapter_for(toolbox, ToolKind.SHELXT, ScriptedExecutor([ScriptedResponse()]))
    _, record = adapter.run(args=("case",), cwd=tmp_path)

    assert record.tool is ToolKind.SHELXT
    assert record.executable_path == str(executable)
    assert record.executable_sha256 == sha256_file(executable)
    assert record.executable_arch == BinaryArch.X86_64.value
    assert record.executable_source == "toolbox"
    assert record.required_translation is False
    assert record.argv[0] == str(executable)


def test_inputs_are_hashed_before_the_run(toolbox: Path, tmp_path: Path):
    write_executable(toolbox / "shelxt", elf(0x3E))
    (tmp_path / "case.ins").write_text("TITL x\n", encoding="utf-8")
    (tmp_path / "case.hkl").write_text("   1   0   0  100.00    2.00\n", encoding="utf-8")
    expected = {
        "case.ins": sha256_file(tmp_path / "case.ins"),
        "case.hkl": sha256_file(tmp_path / "case.hkl"),
    }

    adapter = adapter_for(toolbox, ToolKind.SHELXT, ScriptedExecutor([ScriptedResponse()]))
    _, record = adapter.run(
        args=("case",), cwd=tmp_path, declared_inputs=("case.ins", "case.hkl")
    )

    assert record.input_hashes == expected
    assert all(r.role == "input" for r in record.inputs)


def test_a_declared_input_that_does_not_exist_is_simply_absent(toolbox: Path, tmp_path: Path):
    write_executable(toolbox / "shelxt", elf(0x3E))
    adapter = adapter_for(toolbox, ToolKind.SHELXT, ScriptedExecutor([ScriptedResponse()]))
    _, record = adapter.run(args=("case",), cwd=tmp_path, declared_inputs=("missing.ins",))
    assert record.inputs == ()


def test_unexpected_outputs_are_discovered_and_hashed(toolbox: Path, tmp_path: Path):
    """SHELXT writes one solution per candidate space group; we cannot know how many in advance."""
    write_executable(toolbox / "shelxt", elf(0x3E))
    executor = ScriptedExecutor(
        [
            ScriptedResponse(
                creates=["case_a.res", "case_b.res", "case_c.res", "case.lxt"],
                file_contents={"case_a.res": "TITL best\n"},
            )
        ]
    )
    adapter = adapter_for(toolbox, ToolKind.SHELXT, executor)
    _, record = adapter.run(
        args=("case",), cwd=tmp_path, expected_outputs=("case_a.res",)
    )

    names = set(record.output_hashes)
    assert {"case_a.res", "case_b.res", "case_c.res", "case.lxt"} <= names
    assert record.expected_outputs == ("case_a.res",)
    assert record.missing_outputs == ()
    assert record.output_named("case_a.res").sha256  # type: ignore[union-attr]
    assert len(record.outputs_with_suffix(".res")) == 3


def test_a_modified_input_is_recorded_as_modified_not_created(toolbox: Path, tmp_path: Path):
    write_executable(toolbox / "shelxl", elf(0x3E))
    (tmp_path / "case.res").write_text("old\n", encoding="utf-8")
    executor = ScriptedExecutor(
        [ScriptedResponse(creates=["case.res"], file_contents={"case.res": "new\n"})]
    )
    adapter = adapter_for(toolbox, ToolKind.SHELXL, executor)
    _, record = adapter.run(args=("case",), cwd=tmp_path)

    modified = record.output_named("case.res")
    assert modified is not None
    assert modified.role == "modified"


def test_untouched_files_are_not_listed_as_outputs(toolbox: Path, tmp_path: Path):
    write_executable(toolbox / "shelxl", elf(0x3E))
    (tmp_path / "untouched.txt").write_text("same\n", encoding="utf-8")
    adapter = adapter_for(toolbox, ToolKind.SHELXL, ScriptedExecutor([ScriptedResponse()]))
    _, record = adapter.run(args=("case",), cwd=tmp_path)
    assert "untouched.txt" not in record.output_hashes


def test_a_stdin_script_is_recorded_verbatim_and_by_digest(toolbox: Path, tmp_path: Path):
    """SHELXC and PLATON are driven by instructions; the exact question asked must be recoverable."""
    write_executable(toolbox / "shelxc", elf(0x3E))
    script = "CELL 10 11 12 90 100 90\nSAD case.hkl\nFIND 2\nSFAC Ag\n"
    adapter = adapter_for(toolbox, ToolKind.SHELXC, ScriptedExecutor([ScriptedResponse()]))
    _, record = adapter.run(args=("case",), cwd=tmp_path, stdin_script=script)

    assert record.stdin_script == script
    assert len(record.stdin_sha256) == 64


def test_a_failed_run_still_produces_a_full_record(toolbox: Path, tmp_path: Path):
    write_executable(toolbox / "shelxt", elf(0x3E))
    executor = ScriptedExecutor([ScriptedResponse(exit_code=1, stderr="no solution\n")])
    adapter = adapter_for(toolbox, ToolKind.SHELXT, executor)
    _, record = adapter.run(args=("case",), cwd=tmp_path, expected_outputs=("case_a.res",))

    assert not record.succeeded
    assert record.diagnostic == ToolDiagnosticKind.NONZERO_EXIT.value
    assert "no solution" in record.diagnostic_message
    assert record.exit_code == 1
    assert record.executable_sha256  # provenance survives failure


def test_a_missing_expected_output_is_recorded_as_such(toolbox: Path, tmp_path: Path):
    write_executable(toolbox / "shelxt", elf(0x3E))
    adapter = adapter_for(toolbox, ToolKind.SHELXT, ScriptedExecutor([ScriptedResponse()]))
    _, record = adapter.run(args=("case",), cwd=tmp_path, expected_outputs=("case_a.res",))
    assert record.missing_outputs == ("case_a.res",)
    assert record.diagnostic == ToolDiagnosticKind.EXPECTED_OUTPUT_MISSING.value


def test_logs_are_captured_and_labelled(toolbox: Path, tmp_path: Path):
    write_executable(toolbox / "shelxt", elf(0x3E))
    work = tmp_path / "branch"
    work.mkdir()
    executor = ScriptedExecutor([ScriptedResponse(stdout="hello", stderr="warn")])
    adapter = adapter_for(toolbox, ToolKind.SHELXT, executor)
    _, record = adapter.run(args=("case",), cwd=work, log_dir=work, label="shelxt")

    assert record.stdout_path.endswith("shelxt.stdout.log")
    log = record.output_named("shelxt.stdout.log")
    assert log is not None and log.role == "log"


def test_the_record_serialises(toolbox: Path, tmp_path: Path):
    from crystalx_core._serde import to_jsonable

    write_executable(toolbox / "shelxt", elf(0x3E))
    adapter = adapter_for(toolbox, ToolKind.SHELXT, ScriptedExecutor([ScriptedResponse()]))
    _, record = adapter.run(args=("case",), cwd=tmp_path)
    payload = to_jsonable(record)
    assert payload["tool"] == "shelxt"
    assert isinstance(payload["argv"], list)


# --------------------------------------------------------------------------------------
# directory snapshot
# --------------------------------------------------------------------------------------


def test_snapshot_is_empty_for_a_missing_directory(tmp_path: Path):
    assert snapshot_directory(tmp_path / "absent") == {}


def test_snapshot_sees_nested_files(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.txt").write_text("x", encoding="utf-8")
    assert "sub/x.txt" in snapshot_directory(tmp_path)
