"""Binary-format probing and executable resolution.

The concrete situation this protects against, observed on the project's own development machine:
the SHELX programs in ``toolbox/`` are x86_64 Mach-O binaries while ``platon``/``platon_special``
are arm64. On Apple Silicon the SHELX programs therefore need Rosetta 2, and if it is missing the
kernel refuses them. Reporting that as "not found" sends the user hunting for the wrong problem.

No real crystallographic binary is used here. Every "executable" is a byte pattern written by the
test, and none of them is executed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (
    CPU_ARM64,
    CPU_X86_64,
    elf,
    macho_fat,
    macho_thin,
    write_executable,
)
from crystalx_core.domain.tools import (
    BinaryArch,
    BinaryFormat,
    HostPlatform,
    ResolutionSource,
    ToolDiagnosticKind,
    ToolKind,
)
from crystalx_core.tools.binfmt import probe_binary, probe_bytes
from crystalx_core.tools.resolver import (
    ROSETTA_MARKER_PATHS,
    ResolverConfig,
    ToolResolver,
    detect_host_platform,
    rosetta_is_installed,
)


# --------------------------------------------------------------------------------------
# header probing
# --------------------------------------------------------------------------------------


def test_probes_thin_macho_arm64():
    probe = probe_bytes(macho_thin(CPU_ARM64))
    assert probe.fmt is BinaryFormat.MACH_O
    assert probe.arch is BinaryArch.ARM64


def test_probes_thin_macho_x86_64():
    probe = probe_bytes(macho_thin(CPU_X86_64))
    assert probe.fmt is BinaryFormat.MACH_O
    assert probe.arch is BinaryArch.X86_64


def test_probes_universal_macho():
    probe = probe_bytes(macho_fat([CPU_X86_64, CPU_ARM64]))
    assert probe.fmt is BinaryFormat.MACH_O_FAT
    assert probe.arch is BinaryArch.UNIVERSAL
    assert set(probe.arches) == {BinaryArch.X86_64, BinaryArch.ARM64}
    assert probe.supports(BinaryArch.ARM64)
    assert probe.supports(BinaryArch.X86_64)


def test_probes_elf():
    assert probe_bytes(elf(0x3E)).arch is BinaryArch.X86_64
    assert probe_bytes(elf(0xB7)).arch is BinaryArch.ARM64
    assert probe_bytes(elf(0x3E)).fmt is BinaryFormat.ELF


def test_probes_script():
    probe = probe_bytes(b"#!/bin/sh\necho hi\n")
    assert probe.fmt is BinaryFormat.SCRIPT
    assert probe.arch is BinaryArch.SCRIPT
    assert probe.interpreter == "/bin/sh"
    assert probe.supports(BinaryArch.ARM64)


def test_probe_handles_short_and_unknown_input():
    assert probe_bytes(b"ab").fmt is BinaryFormat.UNKNOWN
    assert probe_bytes(b"garbage bytes here").fmt is BinaryFormat.UNKNOWN


def test_probe_of_missing_file_does_not_raise(tmp_path: Path):
    probe = probe_binary(tmp_path / "absent")
    assert probe.fmt is BinaryFormat.UNKNOWN
    assert "cannot read header" in probe.note


def test_arch_support_matrix():
    arm = probe_bytes(macho_thin(CPU_ARM64))
    intel = probe_bytes(macho_thin(CPU_X86_64))
    assert arm.supports(BinaryArch.ARM64)
    assert not arm.supports(BinaryArch.X86_64)
    assert intel.supports(BinaryArch.X86_64)
    assert not intel.supports(BinaryArch.ARM64)


# --------------------------------------------------------------------------------------
# host detection
# --------------------------------------------------------------------------------------


def test_detect_host_platform_apple_silicon_with_rosetta():
    host = detect_host_platform(
        system="Darwin", machine="arm64", exists=lambda p: p == ROSETTA_MARKER_PATHS[0]
    )
    assert host.is_macos
    assert host.arch is BinaryArch.ARM64
    assert host.rosetta_available is True


def test_detect_host_platform_apple_silicon_without_rosetta():
    host = detect_host_platform(system="Darwin", machine="arm64", exists=lambda _p: False)
    assert host.rosetta_available is False


def test_detect_host_platform_linux_has_no_rosetta_concept():
    host = detect_host_platform(system="Linux", machine="x86_64")
    assert host.arch is BinaryArch.X86_64
    assert host.rosetta_available is None


def test_rosetta_probe_never_executes_anything():
    calls: list[Path] = []

    def spy(path: Path) -> bool:
        calls.append(path)
        return False

    assert rosetta_is_installed(exists=spy) is False
    assert calls == list(ROSETTA_MARKER_PATHS)


# --------------------------------------------------------------------------------------
# resolution: the eight diagnostics
# --------------------------------------------------------------------------------------


def _resolver(toolbox: Path, host: HostPlatform, **kwargs) -> ToolResolver:
    return ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox,
            search_path=[],
            include_known_locations=False,
            include_olex2=False,
        ),
        host=host,
        environ={"PATH": ""},
        **kwargs,
    )


ARM_HOST_WITH_ROSETTA = HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=True)
ARM_HOST_NO_ROSETTA = HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=False)
INTEL_HOST = HostPlatform("Darwin", BinaryArch.X86_64, rosetta_available=None)
LINUX_HOST = HostPlatform("Linux", BinaryArch.X86_64, rosetta_available=None)


def test_diagnostic_not_found(toolbox: Path):
    resolution = _resolver(toolbox, LINUX_HOST).resolve(ToolKind.SHELXT)
    assert not resolution.is_usable
    assert resolution.diagnostic.kind is ToolDiagnosticKind.NOT_FOUND
    assert resolution.identity is None
    assert resolution.searched_paths


def test_diagnostic_not_executable(toolbox: Path):
    write_executable(toolbox / "shelxt", elf(0x3E), mode=0o644)
    resolution = _resolver(toolbox, LINUX_HOST).resolve(ToolKind.SHELXT)
    assert resolution.diagnostic.kind is ToolDiagnosticKind.NOT_EXECUTABLE
    assert "chmod +x" in resolution.diagnostic.remedy


def test_diagnostic_not_a_file(toolbox: Path):
    (toolbox / "shelxt").mkdir()
    resolution = _resolver(toolbox, LINUX_HOST).resolve(ToolKind.SHELXT)
    assert resolution.diagnostic.kind is ToolDiagnosticKind.NOT_A_FILE


def test_diagnostic_ok_native(toolbox: Path):
    write_executable(toolbox / "platon_special", macho_thin(CPU_ARM64))
    resolution = _resolver(toolbox, ARM_HOST_WITH_ROSETTA).resolve(ToolKind.PLATON_SPECIAL)
    assert resolution.is_usable
    assert resolution.diagnostic.kind is ToolDiagnosticKind.OK
    assert resolution.identity is not None
    assert resolution.identity.arch is BinaryArch.ARM64
    assert resolution.identity.source is ResolutionSource.TOOLBOX
    assert resolution.identity.sha256


def test_diagnostic_ok_via_rosetta(toolbox: Path):
    """An x86_64 SHELX build on Apple Silicon with Rosetta present: usable, and said to be slower."""
    write_executable(toolbox / "shelxl", macho_thin(CPU_X86_64))
    resolution = _resolver(toolbox, ARM_HOST_WITH_ROSETTA).resolve(ToolKind.SHELXL)
    assert resolution.is_usable
    assert resolution.diagnostic.kind is ToolDiagnosticKind.OK_VIA_TRANSLATION
    assert "Rosetta" in resolution.diagnostic.message
    assert "slower" in resolution.diagnostic.message


def test_diagnostic_rosetta_missing(toolbox: Path):
    """The headline case: this must not be reported as 'not found'."""
    write_executable(toolbox / "shelxt", macho_thin(CPU_X86_64))
    resolution = _resolver(toolbox, ARM_HOST_NO_ROSETTA).resolve(ToolKind.SHELXT)
    assert not resolution.is_usable
    assert resolution.diagnostic.kind is ToolDiagnosticKind.ROSETTA_MISSING
    assert "Bad CPU type" in resolution.diagnostic.message
    assert "softwareupdate --install-rosetta" in resolution.diagnostic.remedy


def test_diagnostic_arch_mismatch(toolbox: Path):
    """arm64 binary on an Intel Mac: no translation layer exists in that direction."""
    write_executable(toolbox / "shelxt", macho_thin(CPU_ARM64))
    resolution = _resolver(toolbox, INTEL_HOST).resolve(ToolKind.SHELXT)
    assert resolution.diagnostic.kind is ToolDiagnosticKind.ARCH_MISMATCH
    assert "no translation layer" in resolution.diagnostic.message


def test_diagnostic_quarantined(toolbox: Path):
    write_executable(toolbox / "shelxt", macho_thin(CPU_ARM64))
    resolver = _resolver(toolbox, ARM_HOST_WITH_ROSETTA, quarantine_checker=lambda _p: True)
    resolution = resolver.resolve(ToolKind.SHELXT)
    assert resolution.diagnostic.kind is ToolDiagnosticKind.QUARANTINED
    assert "xattr -d" in resolution.diagnostic.remedy


def test_universal_binary_is_native_on_both_hosts(toolbox: Path):
    write_executable(toolbox / "shelxl", macho_fat([CPU_X86_64, CPU_ARM64]))
    for host in (ARM_HOST_NO_ROSETTA, INTEL_HOST):
        resolution = _resolver(toolbox, host).resolve(ToolKind.SHELXL)
        assert resolution.diagnostic.kind is ToolDiagnosticKind.OK, host.arch


# --------------------------------------------------------------------------------------
# precedence and PLATON preference
# --------------------------------------------------------------------------------------


def test_explicit_config_beats_toolbox(tmp_path: Path, toolbox: Path):
    write_executable(toolbox / "shelxl", elf(0x3E))
    elsewhere = tmp_path / "custom"
    elsewhere.mkdir()
    chosen = write_executable(elsewhere / "shelxl", elf(0x3E))

    resolver = ToolResolver(
        ResolverConfig(
            explicit_paths={ToolKind.SHELXL.value: chosen},
            toolbox_dir=toolbox,
            search_path=[],
            include_known_locations=False,
            include_olex2=False,
        ),
        host=LINUX_HOST,
        environ={"PATH": ""},
    )
    resolution = resolver.resolve(ToolKind.SHELXL)
    assert resolution.identity is not None
    assert resolution.identity.path == chosen
    assert resolution.identity.source is ResolutionSource.EXPLICIT_CONFIG


def test_toolbox_beats_path(tmp_path: Path, toolbox: Path):
    on_path = tmp_path / "bin"
    on_path.mkdir()
    write_executable(on_path / "shelxl", elf(0x3E))
    preferred = write_executable(toolbox / "shelxl", elf(0x3E))

    resolver = ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox,
            search_path=[on_path],
            include_known_locations=False,
            include_olex2=False,
        ),
        host=LINUX_HOST,
        environ={"PATH": str(on_path)},
    )
    resolution = resolver.resolve(ToolKind.SHELXL)
    assert resolution.identity is not None
    assert resolution.identity.path == preferred
    assert resolution.identity.source is ResolutionSource.TOOLBOX


def test_platon_special_is_preferred_for_automation(toolbox: Path):
    write_executable(toolbox / "platon", macho_thin(CPU_ARM64))
    write_executable(toolbox / "platon_special", macho_thin(CPU_ARM64))
    resolution = _resolver(toolbox, ARM_HOST_WITH_ROSETTA).resolve_platon()
    assert resolution.kind is ToolKind.PLATON_SPECIAL


def test_platon_falls_back_when_special_is_absent(toolbox: Path):
    write_executable(toolbox / "platon", macho_thin(CPU_ARM64))
    resolution = _resolver(toolbox, ARM_HOST_WITH_ROSETTA).resolve_platon()
    assert resolution.kind is ToolKind.PLATON
    assert resolution.is_usable


def test_platon_fallback_reports_the_special_build_failure(toolbox: Path):
    """When neither is usable the report is about the fallback, not silence."""
    resolution = _resolver(toolbox, ARM_HOST_WITH_ROSETTA).resolve_platon()
    assert not resolution.is_usable
    assert resolution.kind is ToolKind.PLATON


def test_resolve_all_reports_the_mixed_architecture_situation(toolbox: Path):
    """Reproduces the development machine: x86_64 SHELX, arm64 PLATON, no Rosetta."""
    for name in ("shelxt", "shelxl", "shelxd", "shelxe", "shelxc", "shelxs", "anode"):
        write_executable(toolbox / name, macho_thin(CPU_X86_64))
    for name in ("platon", "platon_special"):
        write_executable(toolbox / name, macho_thin(CPU_ARM64))

    report = _resolver(toolbox, ARM_HOST_NO_ROSETTA).resolve_all()

    assert report.usable(ToolKind.PLATON_SPECIAL)
    assert not report.usable(ToolKind.SHELXT)
    shelxt = report.get(ToolKind.SHELXT)
    assert shelxt is not None
    assert shelxt.diagnostic.kind is ToolDiagnosticKind.ROSETTA_MISSING

    with_rosetta = _resolver(toolbox, ARM_HOST_WITH_ROSETTA).resolve_all()
    assert with_rosetta.usable(ToolKind.SHELXT)
    assert set(with_rosetta.usable_kinds) >= {ToolKind.SHELXT, ToolKind.SHELXL, ToolKind.PLATON_SPECIAL}


def test_usr_local_bin_is_searched_even_when_absent_from_path(toolbox: Path):
    """The PLATON-on-Mac guide installs into /usr/local/bin, so we always look there."""
    resolver = ToolResolver(
        ResolverConfig(toolbox_dir=toolbox, include_known_locations=False, include_olex2=False),
        host=LINUX_HOST,
        environ={"PATH": "/nowhere"},
    )
    searched = [p for p, _ in resolver.candidate_paths(ToolKind.PLATON)]
    assert Path("/usr/local/bin/platon") in searched


def test_olex2_locations_are_searched_when_enabled(toolbox: Path):
    resolver = ToolResolver(
        ResolverConfig(toolbox_dir=toolbox, search_path=[], include_olex2=True),
        host=HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=True),
        environ={"PATH": ""},
    )
    sources = {source for _, source in resolver.candidate_paths(ToolKind.SHELXL)}
    assert ResolutionSource.OLEX2 in sources
