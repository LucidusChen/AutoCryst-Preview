"""Strict native policy uses Mac hardware facts even when Python is translated.

Sysctl results and platform facts are injected. All executable files are test-authored
Mach-O headers; no program is launched and no real hardware query is needed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import CPU_ARM64, CPU_X86_64, macho_thin, write_executable
from crystalx_core.app.models import RunConfig
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.tools import BinaryArch, ToolDiagnosticKind, ToolKind
from crystalx_core.tools import resolver as resolver_module


def _translated_mac(monkeypatch, *, arm64: int | None, translated: int | None) -> None:
    monkeypatch.setattr(resolver_module._platform, "system", lambda: "Darwin")
    monkeypatch.setattr(resolver_module._platform, "machine", lambda: "x86_64")
    values = {"hw.optional.arm64": arm64, "sysctl.proc_translated": translated}
    monkeypatch.setattr(resolver_module, "_sysctl_int", lambda name: values[name])


@pytest.mark.parametrize("arm64,translated", [(1, 0), (None, 1), (1, None)])
def test_apple_silicon_hardware_is_detected_from_either_sysctl_fact(
    monkeypatch, arm64: int | None, translated: int | None
):
    _translated_mac(monkeypatch, arm64=arm64, translated=translated)
    host = resolver_module.detect_host_platform(exists=lambda _p: True)
    assert host.arch is BinaryArch.ARM64
    assert host.rosetta_available is True


def test_strict_run_refuses_intel_binary_under_translated_python(tmp_path: Path, monkeypatch):
    _translated_mac(monkeypatch, arm64=1, translated=1)
    intel = write_executable(tmp_path / "shelxt", macho_thin(CPU_X86_64))
    toolchain = RealToolchain.for_config(
        RunConfig(
            tool_paths=(("shelxt", intel),),
            strict_native=True,
            include_olex2=False,
            include_known_locations=False,
        )
    )
    # Keep resolution inside the fixture: no discovery or inspection of installed tools.
    resolver = toolchain.resolver
    assert resolver is not None
    resolver.config = replace(resolver.config, search_path=())
    resolver._is_quarantined = lambda _p: False
    resolution = resolver.resolve(ToolKind.SHELXT)
    assert resolver.host.arch is BinaryArch.ARM64
    assert resolution.identity is None
    assert resolution.diagnostic.kind is ToolDiagnosticKind.TRANSLATION_REFUSED


def test_strict_run_uses_later_arm64_copy_under_translated_python(tmp_path: Path, monkeypatch):
    _translated_mac(monkeypatch, arm64=1, translated=1)
    intel = write_executable(tmp_path / "shelxt", macho_thin(CPU_X86_64))
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    native = write_executable(native_dir / "shelxt", macho_thin(CPU_ARM64))
    resolver = resolver_module.ToolResolver(
        resolver_module.ResolverConfig(
            explicit_paths={"shelxt": intel},
            extra_dirs=(native_dir,),
            search_path=(),
            include_olex2=False,
            include_known_locations=False,
            allow_translation=False,
        ),
        quarantine_checker=lambda _p: False,
    )
    resolution = resolver.resolve(ToolKind.SHELXT)
    assert resolution.identity is not None
    assert resolution.identity.path == native
    assert resolution.diagnostic.kind is ToolDiagnosticKind.OK
    assert [item.kind for item in resolution.rejected] == [ToolDiagnosticKind.TRANSLATION_REFUSED]


def test_unavailable_sysctl_falls_back_to_process_arch_without_inventing_hardware(monkeypatch):
    _translated_mac(monkeypatch, arm64=None, translated=None)
    host = resolver_module.detect_host_platform(exists=lambda _p: True)
    assert host.arch is BinaryArch.X86_64
    assert host.rosetta_available is None


def test_explicit_host_overrides_do_not_consult_this_macs_hardware(monkeypatch):
    def unexpected_sysctl(_name):
        raise AssertionError("explicit test host must not query actual hardware")

    monkeypatch.setattr(resolver_module, "_sysctl_int", unexpected_sysctl)
    host = resolver_module.detect_host_platform(system="Darwin", machine="x86_64")
    assert host.arch is BinaryArch.X86_64
    assert host.rosetta_available is None


def test_linux_autodetection_does_not_consult_macos_sysctl(monkeypatch):
    monkeypatch.setattr(resolver_module._platform, "system", lambda: "Linux")
    monkeypatch.setattr(resolver_module._platform, "machine", lambda: "aarch64")

    def unexpected_sysctl(_name):
        raise AssertionError("Linux must not query macOS hardware facts")

    monkeypatch.setattr(resolver_module, "_sysctl_int", unexpected_sysctl)
    host = resolver_module.detect_host_platform()
    assert host.arch is BinaryArch.ARM64
    assert host.rosetta_available is None
