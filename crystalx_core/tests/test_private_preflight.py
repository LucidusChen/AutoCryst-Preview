"""Synthetic private preflight checks. These byte fixtures are never executed."""

from __future__ import annotations

import builtins
import os
import struct
from pathlib import Path

import pytest

from crystalx_core.app.models import RunMode
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.tools import (
    BinaryArch, BinaryFormat, BinaryProbe, HostPlatform, ResolutionSource,
    ToolAvailabilityReport, ToolDiagnostic, ToolDiagnosticKind, ToolIdentity, ToolKind, ToolResolution,
)
from crystalx_core.private_acceptance.preflight import (
    PreflightPolicy, evaluate_toolchain, recheck_before_dispatch,
)
from crystalx_core.workspace.hashing import sha256_file


def _signed_image(*, filetype=2, platform=1, cpu=0x0100000C, dependency=None, minimum=0x000C0000):
    identifier = b"synthetic\0"
    directory = struct.pack(">9I4BI", 0xFADE0C02, 44 + len(identifier), 0x20000, 2, 0, 44,
                            0, 0, 0, 32, 2, 0, 12, 0) + identifier
    signature = struct.pack(">IIIII", 0xFADE0CC0, 20 + len(directory), 1, 0, 20) + directory
    commands = [struct.pack("<6I", 0x32, 24, platform, minimum, 0x000E0000, 0),
                struct.pack("<IIQQ", 0x80000028, 24, 0, 0)]
    if dependency:
        text = dependency.encode() + b"\0"
        text += b"\0" * ((-len(text)) % 8)
        commands.append(struct.pack("<6I", 0xC, 24 + len(text), 24, 0, 0x10000, 0x10000) + text)
    command_bytes = sum(map(len, commands)) + 16
    commands.append(struct.pack("<4I", 0x1D, 16, 32 + command_bytes, len(signature)))
    return struct.pack("<Iii5I", 0xFEEDFACF, cpu, 0, filetype, len(commands), command_bytes, 0, 0) + b"".join(commands) + signature


class SelectedToolchain(Toolchain):
    def __init__(self, resolutions):
        self.report = ToolAvailabilityReport(HostPlatform("Darwin", BinaryArch.ARM64), resolutions)

    def availability(self):
        return self.report

    def _runner(self):
        raise AssertionError("preflight must not obtain or start a runner")


def _resolution(path, kind=ToolKind.SHELXT, diagnostic=ToolDiagnosticKind.OK, source=ResolutionSource.EXPLICIT_CONFIG):
    stat = path.stat()
    identity = ToolIdentity(kind, path, path.resolve(), source, stat.st_size, sha256_file(path),
                            stat.st_mtime, BinaryProbe(BinaryFormat.MACH_O, BinaryArch.ARM64))
    return ToolResolution(kind, identity, ToolDiagnostic(diagnostic, "synthetic resolution"))


@pytest.fixture
def native(monkeypatch):
    # Most preflight tests need the optional inspector. Absence is itself tested separately.
    host_module = pytest.importorskip("crystalx_native_tools.host")
    inspector = pytest.importorskip("crystalx_native_tools.inspect")
    host = host_module.HostFacts("Darwin", "arm64", "arm64", False, "14.0", (), "test")
    monkeypatch.setattr(host_module, "detect_host", lambda: host)
    monkeypatch.setattr(inspector, "has_xattr", lambda *_args: False)
    # Guard the test suite's entire offline boundary, including any future inspection changes.
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("preflight launched a subprocess"))
    monkeypatch.setattr(os, "system", lambda *_a: pytest.fail("preflight launched a shell"))
    return host_module, inspector


@pytest.fixture
def selected(tmp_path, native):
    path = tmp_path / "synthetic-shelxt"
    path.write_bytes(_signed_image())
    path.chmod(0o755)
    return path, SelectedToolchain({"shelxt": _resolution(path)})


def _policy(**kwargs):
    return PreflightPolicy(required_tools=(ToolKind.SHELXT,), require_platon=False, **kwargs)


def _reason(report, text):
    assert report.status == "BLOCKED"
    assert any(text in reason for reason in report.blockers), report.to_record()


def test_valid_static_does_not_claim_launch_or_validation(selected):
    path, toolchain = selected
    report = evaluate_toolchain(toolchain, policy=_policy())
    assert report.ready, report.to_record()
    record = report.to_record()
    assert record["launch"] == {"status": "NOT_TESTED", "real_invocations": 0}
    assert record["validation"]["status"] == "NOT_VERIFIED"
    assert report.tools[0]["selection"]["sha256"] == sha256_file(path)
    assert report.tools[0]["selection"]["source"] == "explicit_config"
    assert recheck_before_dispatch(toolchain, report).ready


def test_optional_inspector_absent_is_blocked(monkeypatch):
    original = builtins.__import__
    def without_native(name, *args, **kwargs):
        if name.startswith("crystalx_native_tools"):
            raise ImportError("optional package unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_native)
    _reason(evaluate_toolchain(SelectedToolchain({}), policy=_policy()), "optional_native_inspector_unavailable")


def test_demo_not_accepted_as_real(selected):
    _, toolchain = selected
    toolchain.mode = RunMode.MOCK_DEMO
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "real_toolchain_required")


@pytest.mark.parametrize("field,value,reason", [
    ("filetype", 6, "macho_not_executable"),
    ("platform", 2, "macho_platform_not_macos"),
    ("cpu", 0x01000007, "native_architecture_missing"),
    ("minimum", 0x00630000, "built for macOS"),
])
def test_native_inspector_false_positives_closed(selected, field, value, reason):
    path, toolchain = selected
    path.write_bytes(_signed_image(**{field: value}))
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), reason)


@pytest.mark.parametrize("payload", [b"#!/bin/sh\nexec intel-tool\n", b"unknown executable", b"\x7fELF" + b"\0" * 60])
def test_script_and_unknown_native_targets_are_unverified(selected, payload):
    path, toolchain = selected
    path.write_bytes(payload)
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "unsupported_executable_format")


@pytest.mark.parametrize("state,reason", [(True, "quarantined"), (None, "quarantine_state_unverified")])
def test_quarantine_must_be_known_clear(selected, native, monkeypatch, state, reason):
    _, inspector = native
    monkeypatch.setattr(inspector, "has_xattr", lambda *_a: state)
    _reason(evaluate_toolchain(selected[1], policy=_policy()), reason)


@pytest.mark.parametrize("corruption", ["command_extent", "command_count", "signature_truncated", "superblob_extent", "child_extent", "index_count", "versioned_directory_header"])
def test_truncated_load_and_signature_containers_block(selected, corruption):
    path, toolchain = selected
    data = bytearray(_signed_image())
    if corruption == "command_extent":
        struct.pack_into("<I", data, 20, len(data) + 8)
    elif corruption == "command_count":
        struct.pack_into("<I", data, 16, 1)
    elif corruption == "signature_truncated":
        del data[-1:]
    elif corruption == "superblob_extent":
        struct.pack_into(">I", data, 100, len(data) + 4)
    elif corruption == "child_extent":
        struct.pack_into(">I", data, 120, len(data) + 4)
    elif corruption == "index_count":
        struct.pack_into(">I", data, 104, 64)
    elif corruption == "versioned_directory_header":
        struct.pack_into(">I", data, 124, 0x20600)
    path.write_bytes(data)
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "macho_structure")


def test_preferred_invalid_platon_never_silently_switches_to_good_fallback(selected, tmp_path):
    _, toolchain = selected
    for kind, filetype in ((ToolKind.PLATON_SPECIAL, 6), (ToolKind.PLATON, 2)):
        path = tmp_path / kind.value
        path.write_bytes(_signed_image(filetype=filetype))
        path.chmod(0o755)
        toolchain.report.resolutions[kind.value] = _resolution(path, kind=kind)
    report = evaluate_toolchain(toolchain, policy=PreflightPolicy(required_tools=(), require_platon=True))
    assert report.tools[0]["tool"] == "platon_special"
    _reason(report, "macho_not_executable")


def test_required_resource_missing_and_existing_but_unbound_both_block(selected, tmp_path):
    path, _ = selected
    toolchain = SelectedToolchain({"platon": _resolution(path, ToolKind.PLATON)})
    _reason(evaluate_toolchain(toolchain, policy=PreflightPolicy(required_tools=())), "required_checkdef_missing")
    checkdef = tmp_path / "check.def"
    checkdef.write_text("synthetic resource\n")
    policy = PreflightPolicy(required_tools=(), checkdef_path=checkdef)
    _reason(evaluate_toolchain(toolchain, policy=policy), "checkdef_process_binding_unverified")
    policy = PreflightPolicy(required_tools=(), checkdef_path=checkdef, validation_resource_binding_verified=True)
    report = evaluate_toolchain(toolchain, policy=policy)
    assert report.ready, report.to_record()
    checkdef.write_text("modified resource\n")
    _reason(recheck_before_dispatch(toolchain, report), "dispatch_evidence_changed")


def test_selected_identity_hash_must_match_current_file(selected):
    path, toolchain = selected
    path.write_bytes(_signed_image() + b"changed")
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "selected_identity_changed")


def test_dispatch_rechecks_permissions(selected):
    path, toolchain = selected
    report = evaluate_toolchain(toolchain, policy=_policy())
    path.chmod(0o644)
    _reason(recheck_before_dispatch(toolchain, report), "dispatch_evidence_changed")


def test_dispatch_rechecks_selected_path_and_resolution(selected, tmp_path):
    path, toolchain = selected
    report = evaluate_toolchain(toolchain, policy=_policy())
    replacement = tmp_path / "new-selected-path"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o755)
    toolchain.report.resolutions["shelxt"] = _resolution(replacement)
    _reason(recheck_before_dispatch(toolchain, report), "dispatch_evidence_changed")


def test_previous_blocked_report_cannot_be_promoted_by_recheck(selected):
    path, toolchain = selected
    path.chmod(0o644)
    report = evaluate_toolchain(toolchain, policy=_policy())
    path.chmod(0o755)
    _reason(recheck_before_dispatch(toolchain, report), "previous_preflight_not_ready")


def test_missing_dependency_blocks_actual_selected_file(selected, tmp_path):
    path, toolchain = selected
    path.write_bytes(_signed_image(dependency=str(tmp_path / "missing.dylib")))
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "dependency_unverified")


def test_present_non_macho_dependency_cannot_be_treated_as_verified(selected, tmp_path):
    dependency = tmp_path / "not-a-library.dylib"
    dependency.write_text("not a Mach-O library")
    path, toolchain = selected
    path.write_bytes(_signed_image(dependency=str(dependency)))
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "dependency_unverified")


def test_missing_resolution_blocks_without_launch(native):
    _reason(evaluate_toolchain(SelectedToolchain({}), policy=_policy()), "selected_resolution_missing")


def test_synthetic_resolution_source_rejected(selected):
    path, toolchain = selected
    toolchain.report.resolutions["shelxt"] = _resolution(path, source=ResolutionSource.SYNTHETIC_DEMO)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "synthetic_source_not_permitted")


def test_no_stages_and_unimplemented_stage_cannot_pass(native):
    _reason(evaluate_toolchain(SelectedToolchain({}), policy=PreflightPolicy(required_tools=(), require_platon=False)), "no_required_stages")
    _reason(evaluate_toolchain(SelectedToolchain({}), policy=PreflightPolicy(required_tools=(ToolKind.SHELXD,), require_platon=False)), "unsupported_stage")


def test_transitive_missing_library_blocks_and_nonexecutable_dylib_is_inspectable(selected, tmp_path):
    dependency = tmp_path / "synthetic.dylib"
    dependency.write_bytes(_signed_image(filetype=6, dependency=str(tmp_path / "missing-transitive.dylib")))
    dependency.chmod(0o644)
    path, toolchain = selected
    path.write_bytes(_signed_image(dependency=str(dependency)))
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    report = evaluate_toolchain(toolchain, policy=_policy())
    _reason(report, "dependency_unverified")
    assert len(report.tools[0]["dependency_files"]) == 1


def test_dispatch_rechecks_nested_dependency_hash(selected, tmp_path):
    dependency = tmp_path / "synthetic.dylib"
    dependency.write_bytes(_signed_image(filetype=6))
    dependency.chmod(0o644)
    path, toolchain = selected
    path.write_bytes(_signed_image(dependency=str(dependency)))
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    report = evaluate_toolchain(toolchain, policy=_policy())
    assert report.ready, report.to_record()
    dependency.write_bytes(_signed_image(filetype=6) + b"different")
    _reason(recheck_before_dispatch(toolchain, report), "dispatch_evidence_changed")


@pytest.mark.parametrize("change,reason", [
    ({"filetype": 2}, "dependency_not_native_dylib"),
    ({"platform": 2}, "dependency_platform_or_minos_unverified"),
    ({"minimum": 0x00630000}, "dependency_inspection_rejected"),
    ({"truncated": True}, "dependency_structure_rejected"),
])
def test_library_inspection_context_preserves_other_rejections(selected, tmp_path, change, reason):
    dependency = tmp_path / "invalid-synthetic.dylib"
    options = dict(change)
    truncated = options.pop("truncated", False)
    image = _signed_image(**{"filetype": 6, **options})
    dependency.write_bytes(image[:-1] if truncated else image)
    dependency.chmod(0o644)
    path, toolchain = selected
    path.write_bytes(_signed_image(dependency=str(dependency)))
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), reason)


def test_linked_library_quarantine_still_blocks(selected, native, tmp_path, monkeypatch):
    dependency = tmp_path / "quarantined-synthetic.dylib"
    dependency.write_bytes(_signed_image(filetype=6))
    dependency.chmod(0o644)
    path, toolchain = selected
    path.write_bytes(_signed_image(dependency=str(dependency)))
    toolchain.report.resolutions["shelxt"] = _resolution(path)
    _, inspector = native
    monkeypatch.setattr(inspector, "has_xattr", lambda target, *_: Path(target) == dependency)
    _reason(evaluate_toolchain(toolchain, policy=_policy()), "dependency_quarantine_unverified")


def test_loader_environment_override_blocks_without_leaking_value(selected, monkeypatch):
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/private/synthetic-sensitive-path.dylib")
    report = evaluate_toolchain(selected[1], policy=_policy())
    _reason(report, "loader_environment_override_unverified")
    assert "/private/synthetic-sensitive-path.dylib" not in str(report.to_record())


def test_dispatch_rechecks_quarantine(selected, native, monkeypatch):
    _, inspector = native
    report = evaluate_toolchain(selected[1], policy=_policy())
    monkeypatch.setattr(inspector, "has_xattr", lambda *_a: True)
    _reason(recheck_before_dispatch(selected[1], report), "dispatch_evidence_changed")
