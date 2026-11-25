"""Executable format and architecture detection by reading header bytes.

The program is **never executed** to find out what it is. A few dozen bytes are read and the file is
closed.

This exists because of a concrete situation on this project's development machine: the SHELX
executables in ``toolbox/`` are x86_64 Mach-O binaries while ``platon``/``platon_special`` are arm64.
On Apple Silicon the SHELX programs therefore run only under Rosetta 2, and if Rosetta is absent the
kernel refuses them with ``Bad CPU type in executable``. Naive code reports that as "shelxt not
found", which sends the user looking for the wrong problem.
"""

from __future__ import annotations

import struct
from pathlib import Path

from crystalx_core.domain.tools import BinaryArch, BinaryFormat, BinaryProbe

__all__ = ["probe_binary", "probe_bytes", "HEADER_READ_BYTES"]

#: Enough for a Mach-O fat header with several architectures.
HEADER_READ_BYTES = 4096

# Mach-O magics
_MH_MAGIC = 0xFEEDFACE
_MH_CIGAM = 0xCEFAEDFE
_MH_MAGIC_64 = 0xFEEDFACF
_MH_CIGAM_64 = 0xCFFAEDFE
_FAT_MAGIC = 0xCAFEBABE
_FAT_CIGAM = 0xBEBAFECA
_FAT_MAGIC_64 = 0xCAFEBABF
_FAT_CIGAM_64 = 0xBFBAFECA

# Mach-O cpu types
_CPU_ARCH_ABI64 = 0x01000000
_CPU_TYPE_X86 = 7
_CPU_TYPE_ARM = 12
_CPU_TYPE_X86_64 = _CPU_TYPE_X86 | _CPU_ARCH_ABI64  # 0x01000007
_CPU_TYPE_ARM64 = _CPU_TYPE_ARM | _CPU_ARCH_ABI64  # 0x0100000C

# ELF e_machine
_EM_386 = 0x03
_EM_X86_64 = 0x3E
_EM_ARM = 0x28
_EM_AARCH64 = 0xB7

_MACHO_CPU_TO_ARCH = {
    _CPU_TYPE_X86_64: BinaryArch.X86_64,
    _CPU_TYPE_ARM64: BinaryArch.ARM64,
    _CPU_TYPE_X86: BinaryArch.I386,
    _CPU_TYPE_ARM: BinaryArch.ARM32,
}

_ELF_MACHINE_TO_ARCH = {
    _EM_X86_64: BinaryArch.X86_64,
    _EM_AARCH64: BinaryArch.ARM64,
    _EM_386: BinaryArch.I386,
    _EM_ARM: BinaryArch.ARM32,
}


def probe_binary(path: Path | str) -> BinaryProbe:
    """Read the first bytes of ``path`` and classify it."""
    target = Path(path)
    try:
        with open(target, "rb") as handle:
            head = handle.read(HEADER_READ_BYTES)
    except OSError as exc:
        return BinaryProbe(
            fmt=BinaryFormat.UNKNOWN,
            arch=BinaryArch.UNKNOWN,
            note=f"cannot read header: {exc}",
        )
    return probe_bytes(head)


def probe_bytes(head: bytes) -> BinaryProbe:
    """Classify an executable from its leading bytes. Pure, so it is trivially testable."""
    if len(head) < 4:
        return BinaryProbe(BinaryFormat.UNKNOWN, BinaryArch.UNKNOWN, note="file too short")

    if head[:2] == b"#!":
        line = head.split(b"\n", 1)[0].decode("utf-8", errors="replace")
        return BinaryProbe(
            fmt=BinaryFormat.SCRIPT,
            arch=BinaryArch.SCRIPT,
            interpreter=line[2:].strip(),
        )

    if head[:4] == b"\x7fELF":
        return _probe_elf(head)

    magic_be = struct.unpack(">I", head[:4])[0]
    magic_le = struct.unpack("<I", head[:4])[0]

    if magic_be in (_FAT_MAGIC, _FAT_MAGIC_64) or magic_le in (_FAT_CIGAM, _FAT_CIGAM_64):
        return _probe_macho_fat(head, is_64=magic_be == _FAT_MAGIC_64)

    for magic, endian in ((magic_le, "<"), (magic_be, ">")):
        if magic in (_MH_MAGIC, _MH_MAGIC_64, _MH_CIGAM, _MH_CIGAM_64):
            return _probe_macho_thin(head, endian)

    if head[:2] == b"MZ":
        return BinaryProbe(BinaryFormat.PE, BinaryArch.UNKNOWN, note="Windows PE, arch not parsed")

    return BinaryProbe(BinaryFormat.UNKNOWN, BinaryArch.UNKNOWN, note="unrecognised magic")


def _probe_elf(head: bytes) -> BinaryProbe:
    if len(head) < 20:
        return BinaryProbe(BinaryFormat.ELF, BinaryArch.UNKNOWN, note="truncated ELF header")
    little_endian = head[5] == 1
    fmt = "<H" if little_endian else ">H"
    machine = struct.unpack_from(fmt, head, 18)[0]
    arch = _ELF_MACHINE_TO_ARCH.get(machine, BinaryArch.UNKNOWN)
    note = "" if arch is not BinaryArch.UNKNOWN else f"unmapped ELF e_machine 0x{machine:x}"
    return BinaryProbe(BinaryFormat.ELF, arch, note=note)


def _probe_macho_thin(head: bytes, endian: str) -> BinaryProbe:
    if len(head) < 8:
        return BinaryProbe(BinaryFormat.MACH_O, BinaryArch.UNKNOWN, note="truncated Mach-O header")
    cputype = struct.unpack_from(f"{endian}i", head, 4)[0]
    arch = _MACHO_CPU_TO_ARCH.get(cputype & 0xFFFFFFFF, BinaryArch.UNKNOWN)
    if arch is BinaryArch.UNKNOWN:
        arch = _MACHO_CPU_TO_ARCH.get(cputype, BinaryArch.UNKNOWN)
    note = "" if arch is not BinaryArch.UNKNOWN else f"unmapped Mach-O cputype {cputype}"
    return BinaryProbe(BinaryFormat.MACH_O, arch, note=note)


def _probe_macho_fat(head: bytes, *, is_64: bool) -> BinaryProbe:
    """Mach-O universal binary: the fat header is always big-endian on disk."""
    if len(head) < 8:
        return BinaryProbe(
            BinaryFormat.MACH_O_FAT, BinaryArch.UNKNOWN, note="truncated fat header"
        )
    nfat = struct.unpack_from(">I", head, 4)[0]
    entry_size = 32 if is_64 else 20
    arches: list[BinaryArch] = []
    for index in range(min(nfat, 16)):
        offset = 8 + index * entry_size
        if offset + 4 > len(head):
            break
        cputype = struct.unpack_from(">i", head, offset)[0]
        arch = _MACHO_CPU_TO_ARCH.get(cputype & 0xFFFFFFFF) or _MACHO_CPU_TO_ARCH.get(cputype)
        if arch is not None and arch not in arches:
            arches.append(arch)
    return BinaryProbe(
        fmt=BinaryFormat.MACH_O_FAT,
        arch=BinaryArch.UNIVERSAL,
        arches=tuple(arches),
        note=f"{nfat} slices",
    )
