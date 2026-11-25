"""Mach-O parsing and static file inspection, on synthetic files written by the tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

import machobuild as mb
from conftest import apple_silicon, linux_host, write_exe
from crystalx_native_tools.inspect import inspect_file, resolve_dependency
from crystalx_native_tools.macho import parse_macho_bytes, read_macho

#: A library path that exists on no machine. Real XQuartz paths would make these tests depend on
#: whether the machine running them has XQuartz installed (the owner's Mac does).
ABSENT_X11 = "/nonexistent/crystalx-test/X11/lib/libX11.6.dylib"


def test_a_thin_arm64_executable_is_fully_described():
    data = mb.thin(mb.CPU_ARM64, libraries=("/usr/lib/libSystem.B.dylib", "@rpath/libgfortran.5.dylib"),
                   weak=("/opt/X11/lib/libXt.dylib",), rpaths=("@loader_path/../lib",), minos=(13, 1, 0),
                   identifier="platon_special")
    m = parse_macho_bytes(data)
    assert m.kind == "thin" and m.arches == ("arm64",)
    s = m.slices[0]
    assert s.filetype == "execute" and s.has_entry_point and "PIE" in s.flags
    assert [(lib.name, lib.kind) for lib in s.libraries] == [
        ("/usr/lib/libSystem.B.dylib", "load"),
        ("@rpath/libgfortran.5.dylib", "load"),
        ("/opt/X11/lib/libXt.dylib", "weak"),
    ]
    assert s.rpaths == ("@loader_path/../lib",)
    assert (s.platform, s.min_os, s.sdk) == ("macos", "13.1", "14.0")
    sig = s.code_signature
    assert sig.present and sig.adhoc and sig.linker_signed and sig.identifier == "platon_special"
    assert not s.problems


def test_a_team_signed_slice_reports_its_team():
    s = parse_macho_bytes(mb.thin(mb.CPU_ARM64, signed=True, identifier="x", team="ABCDE12345")).slices[0]
    assert s.code_signature.team_id == "ABCDE12345"


def test_x86_64_and_universal_files():
    x86 = parse_macho_bytes(mb.thin(mb.CPU_X86_64))
    assert x86.arches == ("x86_64",) and not x86.has_arm64
    assert not x86.slices[0].code_signature.present
    universal = parse_macho_bytes(mb.fat(mb.thin(mb.CPU_X86_64), mb.thin(mb.CPU_ARM64), cputypes=(mb.CPU_X86_64, mb.CPU_ARM64)))
    assert universal.kind == "fat" and universal.arches == ("x86_64", "arm64")
    assert universal.slice_for("arm64").code_signature.present
    assert universal.slice_for("x86_64").offset == 4096


def test_arm64e_counts_as_arm64_for_the_host():
    m = parse_macho_bytes(mb.thin(mb.CPU_ARM64, subtype=mb.SUBTYPE_ARM64E))
    assert m.arches == ("arm64e",) and m.has_arm64
    assert m.slice_for("arm64").arch == "arm64e"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"#!/bin/sh\necho hi\n",
        b"\x7fELF" + b"\0" * 60,
        struct.pack(">II", 0xCAFEBABE, 0x34) + b"\0" * 64,  # a Java class file shares the magic
    ],
)
def test_non_macho_files_are_not_mistaken_for_one(data):
    assert parse_macho_bytes(data) is None


def test_malformed_load_commands_are_problems_not_exceptions():
    data = bytearray(mb.thin(mb.CPU_ARM64))
    struct.pack_into("<I", data, 16, 50)  # ncmds far beyond what is there
    m = parse_macho_bytes(bytes(data))
    assert m is not None and m.slices[0].problems
    truncated = parse_macho_bytes(bytes(data[:40]))
    assert truncated is not None and truncated.slices[0].problems


def test_a_fat_slice_outside_the_file_is_reported():
    data = struct.pack(">II", 0xCAFEBABE, 1) + struct.pack(">iiIII", mb.CPU_ARM64, 0, 1 << 30, 4096, 12)
    m = parse_macho_bytes(data + b"\0" * 64)
    assert m.slices[0].problems == ("slice lies outside the file",)


# --------------------------------------------------------------------------------------
# libraries, as dyld would look for them
# --------------------------------------------------------------------------------------


def test_library_resolution_rules(tmp_path: Path):
    lib_dir = tmp_path / "lib"
    write_exe(lib_dir / "libgfortran.5.dylib", mb.thin(mb.CPU_ARM64, filetype=mb.MH_DYLIB))
    write_exe(lib_dir / "libintel.dylib", mb.thin(mb.CPU_X86_64, filetype=mb.MH_DYLIB))
    exe_dir = tmp_path / "bin"
    common = dict(executable_dir=exe_dir, rpaths=("@loader_path/../lib",), arch="arm64", on_macos=True)
    assert resolve_dependency("/usr/lib/libSystem.B.dylib", "load", **common).status == "system_cache"
    found = resolve_dependency("@rpath/libgfortran.5.dylib", "load", **common)
    assert found.status == "present" and found.arch_ok is True and not found.blocks_launch
    assert resolve_dependency("@executable_path/../lib/libgfortran.5.dylib", "load", **common).status == "present"
    wrong = resolve_dependency(str(lib_dir / "libintel.dylib"), "load", **common)
    assert wrong.arch_ok is False and wrong.blocks_launch
    missing = resolve_dependency(ABSENT_X11, "load", **common)
    assert missing.status == "missing" and missing.blocks_launch
    weak = resolve_dependency(ABSENT_X11, "weak", **common)
    assert weak.status == "missing_weak" and not weak.blocks_launch
    no_rpath = resolve_dependency("@rpath/x.dylib", "load", executable_dir=exe_dir, rpaths=(), arch="arm64", on_macos=True)
    assert no_rpath.status == "unresolved"
    elsewhere = resolve_dependency(ABSENT_X11, "load", executable_dir=exe_dir, rpaths=(), arch="arm64", on_macos=False)
    assert elsewhere.status == "not_checked" and not elsewhere.blocks_launch


def test_inspection_reports_identity_and_launch_blockers(tmp_path: Path):
    good = write_exe(tmp_path / "good", mb.thin(mb.CPU_ARM64))
    report = inspect_file(good, apple_silicon())
    assert report.exists and report.regular and report.executable
    assert report.format == "mach_o" and report.arches == ("arm64",) and report.run_slice == "arm64"
    assert len(report.sha256) == 64 and report.code_signature["adhoc"] is True
    assert not report.problems

    unsigned = inspect_file(write_exe(tmp_path / "unsigned", mb.thin(mb.CPU_ARM64, signed=False)), apple_silicon())
    assert any("without a code signature" in p for p in unsigned.problems)

    future = inspect_file(write_exe(tmp_path / "future", mb.thin(mb.CPU_ARM64, minos=(30, 0, 0))), apple_silicon())
    assert any(p.startswith("built for macOS 30.0") for p in future.problems)

    linked = inspect_file(write_exe(tmp_path / "linked", mb.thin(mb.CPU_ARM64, libraries=(ABSENT_X11,))), apple_silicon())
    assert any("libX11" in p and "missing" in p for p in linked.problems)

    not_exec = inspect_file(write_exe(tmp_path / "plain", mb.thin(mb.CPU_ARM64), mode=0o644), apple_silicon())
    assert "not marked executable" in not_exec.problems

    assert inspect_file(tmp_path / "nothing", apple_silicon()).format == "missing"


def test_on_a_non_mac_host_libraries_are_not_judged(tmp_path: Path):
    linked = write_exe(tmp_path / "linked", mb.thin(mb.CPU_ARM64, libraries=(ABSENT_X11,)))
    report = inspect_file(linked, linux_host())
    assert report.run_slice is None
    assert {d.status for d in report.dependencies} == {"not_checked"}
    assert not report.problems


def test_quarantine_is_a_problem(tmp_path: Path, monkeypatch):
    import crystalx_native_tools.inspect as inspect_module

    monkeypatch.setattr(inspect_module, "has_xattr", lambda path, name: True)
    report = inspect_file(write_exe(tmp_path / "q", mb.thin(mb.CPU_ARM64)), apple_silicon())
    assert report.quarantined is True and "carries the macOS quarantine attribute" in report.problems


def test_read_macho_on_a_file(tmp_path: Path):
    path = write_exe(tmp_path / "x", mb.thin(mb.CPU_X86_64))
    assert read_macho(path).arches == ("x86_64",)
    assert read_macho(tmp_path / "missing") is None
