"""Static Mach-O reader: architectures, linked libraries, run paths, minimum OS and code signature.

Only bounded byte ranges are read. A file is never executed, mapped or modified. Thin 32/64-bit
files in either byte order and universal ("fat") files, including ``FAT_MAGIC_64``, are handled.
A malformed file yields a :class:`MachOFile` whose slices carry ``problems`` rather than an
exception, so one damaged program cannot hide the state of the others.

Why each field matters on Apple Silicon:

* **architectures** -- macOS runs an arm64 slice natively; an x86_64-only file needs Rosetta 2.
* **linked libraries / run paths** -- a program built with Homebrew ``gfortran`` loads
  ``libgfortran`` from wherever Homebrew put it; the ordinary PLATON build also loads XQuartz's
  ``libX11``. If such a library is missing the program cannot start at all.
* **code signature** -- arm64 code must carry at least an ad-hoc signature or the kernel kills it
  at launch. The linker adds one automatically, so absence usually means the file was altered.
* **minimum OS** -- a build for a newer macOS than the host refuses to start.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LinkedLibrary",
    "CodeSignature",
    "MachOSlice",
    "MachOFile",
    "read_macho",
    "parse_macho_bytes",
    "MAX_LOAD_COMMAND_BYTES",
    "MAX_SIGNATURE_BYTES",
]

MAX_LOAD_COMMAND_BYTES = 4 * 1024 * 1024
MAX_SIGNATURE_BYTES = 2 * 1024 * 1024
_MAX_FAT_ARCHES = 32

_MH_MAGIC = 0xFEEDFACE
_MH_MAGIC_64 = 0xFEEDFACF
_FAT_MAGIC = 0xCAFEBABE
_FAT_MAGIC_64 = 0xCAFEBABF

_CPU_ARCH_ABI64 = 0x01000000
_CPU_ARCH_ABI64_32 = 0x02000000
_CPU_NAMES = {
    7: "i386",
    7 | _CPU_ARCH_ABI64: "x86_64",
    12: "arm",
    12 | _CPU_ARCH_ABI64: "arm64",
    12 | _CPU_ARCH_ABI64_32: "arm64_32",
    18: "ppc",
    18 | _CPU_ARCH_ABI64: "ppc64",
}
_CPU_SUBTYPE_MASK = 0x00FFFFFF
_CPU_SUBTYPE_ARM64E = 2

_FILETYPES = {1: "object", 2: "execute", 3: "fvmlib", 4: "core", 5: "preload", 6: "dylib",
              7: "dylinker", 8: "bundle", 9: "dylib_stub", 10: "dsym", 11: "kext_bundle"}
_HEADER_FLAGS = {0x1: "NOUNDEFS", 0x4: "DYLDLINK", 0x80: "TWOLEVEL", 0x8000: "WEAK_DEFINES",
                 0x10000: "BINDS_TO_WEAK", 0x200000: "PIE"}

_LC_REQ_DYLD = 0x80000000
_LC_UNIXTHREAD = 0x5
_LC_LOAD_DYLIB = 0xC
_LC_ID_DYLIB = 0xD
_LC_LOAD_WEAK_DYLIB = 0x18 | _LC_REQ_DYLD
_LC_RPATH = 0x1C | _LC_REQ_DYLD
_LC_CODE_SIGNATURE = 0x1D
_LC_REEXPORT_DYLIB = 0x1F | _LC_REQ_DYLD
_LC_LAZY_LOAD_DYLIB = 0x20
_LC_LOAD_UPWARD_DYLIB = 0x23 | _LC_REQ_DYLD
_LC_VERSION_MIN_MACOSX = 0x24
_LC_MAIN = 0x28 | _LC_REQ_DYLD
_LC_BUILD_VERSION = 0x32
_DYLIB_KINDS = {
    _LC_LOAD_DYLIB: "load",
    _LC_LOAD_WEAK_DYLIB: "weak",
    _LC_REEXPORT_DYLIB: "reexport",
    _LC_LAZY_LOAD_DYLIB: "lazy",
    _LC_LOAD_UPWARD_DYLIB: "upward",
}
_PLATFORMS = {1: "macos", 2: "ios", 3: "tvos", 4: "watchos", 5: "bridgeos", 6: "maccatalyst",
              7: "iossimulator", 8: "tvossimulator", 9: "watchossimulator", 10: "driverkit",
              11: "visionos", 12: "visionossimulator"}

_CSMAGIC_EMBEDDED_SIGNATURE = 0xFADE0CC0
_CSMAGIC_CODEDIRECTORY = 0xFADE0C02
_CSSLOT_CODEDIRECTORY = 0
_CS_ADHOC = 0x2
_CS_LINKER_SIGNED = 0x20000


@dataclass(frozen=True)
class LinkedLibrary:
    name: str
    """The install name as recorded, e.g. ``/usr/lib/libSystem.B.dylib`` or ``@rpath/libgfortran.5.dylib``."""

    kind: str
    """``load``, ``weak`` (may be absent at run time), ``reexport``, ``lazy`` or ``upward``."""

    current_version: str = ""
    compatibility_version: str = ""


@dataclass(frozen=True)
class CodeSignature:
    present: bool
    adhoc: bool | None = None
    linker_signed: bool | None = None
    identifier: str | None = None
    team_id: str | None = None
    note: str = ""


@dataclass(frozen=True)
class MachOSlice:
    arch: str
    cputype: int
    cpusubtype: int
    offset: int
    size: int
    filetype: str = "unknown"
    flags: tuple[str, ...] = ()
    libraries: tuple[LinkedLibrary, ...] = ()
    rpaths: tuple[str, ...] = ()
    platform: str | None = None
    min_os: str | None = None
    sdk: str | None = None
    has_entry_point: bool = False
    code_signature: CodeSignature = field(default_factory=lambda: CodeSignature(present=False))
    problems: tuple[str, ...] = ()

    @property
    def is_arm64(self) -> bool:
        return self.arch in ("arm64", "arm64e")


@dataclass(frozen=True)
class MachOFile:
    kind: str
    """``thin`` or ``fat``."""

    slices: tuple[MachOSlice, ...]
    problems: tuple[str, ...] = ()

    @property
    def arches(self) -> tuple[str, ...]:
        return tuple(s.arch for s in self.slices)

    @property
    def has_arm64(self) -> bool:
        return any(s.is_arm64 for s in self.slices)

    def slice_for(self, arch: str) -> MachOSlice | None:
        """The slice the host would run: ``arm64`` matches an ``arm64e`` slice too."""
        for candidate in self.slices:
            if candidate.arch == arch:
                return candidate
        if arch == "arm64":
            for candidate in self.slices:
                if candidate.arch == "arm64e":
                    return candidate
        return None

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "arches": list(self.arches),
            "problems": list(self.problems),
            "slices": [
                {
                    "arch": s.arch,
                    "filetype": s.filetype,
                    "flags": list(s.flags),
                    "platform": s.platform,
                    "min_os": s.min_os,
                    "sdk": s.sdk,
                    "libraries": [
                        {"name": lib.name, "kind": lib.kind, "current_version": lib.current_version}
                        for lib in s.libraries
                    ],
                    "rpaths": list(s.rpaths),
                    "code_signature": {
                        "present": s.code_signature.present,
                        "adhoc": s.code_signature.adhoc,
                        "linker_signed": s.code_signature.linker_signed,
                        "identifier": s.code_signature.identifier,
                        "team_id": s.code_signature.team_id,
                    },
                    "problems": list(s.problems),
                }
                for s in self.slices
            ],
        }


class _Reader:
    """Bounded random access to a file or a byte string."""

    def __init__(self, source: Path | bytes) -> None:
        self._data = source if isinstance(source, (bytes, bytearray)) else None
        self._path = None if self._data is not None else Path(source)
        self.size = len(self._data) if self._data is not None else self._path.stat().st_size

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length <= 0 or offset >= self.size:
            return b""
        length = min(length, self.size - offset)
        if self._data is not None:
            return bytes(self._data[offset : offset + length])
        with open(self._path, "rb") as handle:
            handle.seek(offset)
            return handle.read(length)


def _version(value: int) -> str:
    major, minor, patch = value >> 16, (value >> 8) & 0xFF, value & 0xFF
    return f"{major}.{minor}" + (f".{patch}" if patch else "")


def _cstring(blob: bytes, start: int) -> str:
    if start < 0 or start >= len(blob):
        return ""
    end = blob.find(b"\0", start)
    raw = blob[start:] if end < 0 else blob[start:end]
    return raw.decode("utf-8", errors="replace")


def _arch_name(cputype: int, cpusubtype: int) -> str:
    name = _CPU_NAMES.get(cputype & 0xFFFFFFFF)
    if name is None:
        return f"unknown(0x{cputype & 0xFFFFFFFF:x})"
    if name == "arm64" and (cpusubtype & _CPU_SUBTYPE_MASK) == _CPU_SUBTYPE_ARM64E:
        return "arm64e"
    return name


def read_macho(path: Path | str) -> MachOFile | None:
    """Parse ``path`` if it is a Mach-O file; ``None`` if it is not one (or cannot be read)."""
    try:
        reader = _Reader(Path(path))
    except OSError:
        return None
    return _parse(reader)


def parse_macho_bytes(data: bytes) -> MachOFile | None:
    """The same, for bytes already in memory (used by the tests)."""
    return _parse(_Reader(data))


def _parse(reader: _Reader) -> MachOFile | None:
    head = reader.read(0, 8)
    if len(head) < 8:
        return None
    magic_be = struct.unpack(">I", head[:4])[0]
    if magic_be in (_FAT_MAGIC, _FAT_MAGIC_64):
        count = struct.unpack(">I", head[4:8])[0]
        # 0xCAFEBABE is also the Java class-file magic; there the next word is a version >= 45.
        if 0 < count <= _MAX_FAT_ARCHES:
            return _parse_fat(reader, count, is_64=magic_be == _FAT_MAGIC_64)
        return None
    endian = _thin_endian(head[:4])
    if endian is None:
        return None
    single = _parse_slice(reader, 0, reader.size)
    return MachOFile(kind="thin", slices=(single,))


def _thin_endian(magic: bytes) -> str | None:
    le = struct.unpack("<I", magic)[0]
    be = struct.unpack(">I", magic)[0]
    if le in (_MH_MAGIC, _MH_MAGIC_64):
        return "<"
    if be in (_MH_MAGIC, _MH_MAGIC_64):
        return ">"
    return None


def _parse_fat(reader: _Reader, count: int, *, is_64: bool) -> MachOFile:
    entry = 32 if is_64 else 20
    table = reader.read(8, entry * count)
    slices: list[MachOSlice] = []
    problems: list[str] = []
    if len(table) < entry * count:
        problems.append("truncated universal header")
    for index in range(len(table) // entry):
        chunk = table[index * entry : (index + 1) * entry]
        if is_64:
            cputype, cpusubtype, offset, size, _align, _reserved = struct.unpack(">iiQQII", chunk)
        else:
            cputype, cpusubtype, offset, size, _align = struct.unpack(">iiIII", chunk)
        if offset + size > reader.size or size <= 0:
            slices.append(
                MachOSlice(
                    arch=_arch_name(cputype, cpusubtype),
                    cputype=cputype,
                    cpusubtype=cpusubtype,
                    offset=offset,
                    size=size,
                    problems=("slice lies outside the file",),
                )
            )
            continue
        parsed = _parse_slice(reader, offset, size)
        if parsed.arch != _arch_name(cputype, cpusubtype):
            parsed = MachOSlice(
                **{**parsed.__dict__, "problems": parsed.problems + ("universal table and slice header disagree on the CPU",)}
            )
        slices.append(parsed)
    return MachOFile(kind="fat", slices=tuple(slices), problems=tuple(problems))


def _parse_slice(reader: _Reader, offset: int, size: int) -> MachOSlice:
    head = reader.read(offset, 32)
    endian = _thin_endian(head[:4]) if len(head) >= 4 else None
    if endian is None or len(head) < 28:
        return MachOSlice(arch="unknown", cputype=0, cpusubtype=0, offset=offset, size=size,
                          problems=("no Mach-O header at this offset",))
    magic = struct.unpack(f"{endian}I", head[:4])[0]
    is_64 = magic == _MH_MAGIC_64
    cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags = struct.unpack(f"{endian}iiIIII", head[4:28])
    header_size = 32 if is_64 else 28
    problems: list[str] = []
    if sizeofcmds > MAX_LOAD_COMMAND_BYTES:
        problems.append(f"load commands ({sizeofcmds} bytes) exceed the reader's limit; truncated")
        sizeofcmds = MAX_LOAD_COMMAND_BYTES
    blob = reader.read(offset + header_size, sizeofcmds)
    if len(blob) < sizeofcmds:
        problems.append("load commands are truncated")

    libraries: list[LinkedLibrary] = []
    rpaths: list[str] = []
    platform = min_os = sdk = None
    has_entry = False
    signature = CodeSignature(present=False)
    cursor = 0
    for _ in range(ncmds):
        if cursor + 8 > len(blob):
            problems.append("load command list ends early")
            break
        cmd, cmdsize = struct.unpack_from(f"{endian}II", blob, cursor)
        if cmdsize < 8 or cursor + cmdsize > len(blob):
            problems.append(f"malformed load command 0x{cmd:x}")
            break
        body = blob[cursor : cursor + cmdsize]
        if cmd in _DYLIB_KINDS and cmdsize >= 24:
            name_offset, _ts, current, compat = struct.unpack_from(f"{endian}IIII", body, 8)
            libraries.append(
                LinkedLibrary(
                    name=_cstring(body, name_offset),
                    kind=_DYLIB_KINDS[cmd],
                    current_version=_version(current),
                    compatibility_version=_version(compat),
                )
            )
        elif cmd == _LC_RPATH and cmdsize >= 12:
            (path_offset,) = struct.unpack_from(f"{endian}I", body, 8)
            rpaths.append(_cstring(body, path_offset))
        elif cmd == _LC_BUILD_VERSION and cmdsize >= 24:
            plat, minos, sdkv, _ntools = struct.unpack_from(f"{endian}IIII", body, 8)
            platform, min_os, sdk = _PLATFORMS.get(plat, f"platform-{plat}"), _version(minos), _version(sdkv)
        elif cmd == _LC_VERSION_MIN_MACOSX and cmdsize >= 16 and min_os is None:
            minos, sdkv = struct.unpack_from(f"{endian}II", body, 8)
            platform, min_os, sdk = "macos", _version(minos), _version(sdkv)
        elif cmd in (_LC_MAIN, _LC_UNIXTHREAD):
            has_entry = True
        elif cmd == _LC_CODE_SIGNATURE and cmdsize >= 16:
            dataoff, datasize = struct.unpack_from(f"{endian}II", body, 8)
            signature = _read_signature(reader, offset + dataoff, min(datasize, MAX_SIGNATURE_BYTES))
        cursor += cmdsize

    return MachOSlice(
        arch=_arch_name(cputype, cpusubtype),
        cputype=cputype,
        cpusubtype=cpusubtype,
        offset=offset,
        size=size,
        filetype=_FILETYPES.get(filetype, f"type-{filetype}"),
        flags=tuple(name for bit, name in _HEADER_FLAGS.items() if flags & bit),
        libraries=tuple(libraries),
        rpaths=tuple(rpaths),
        platform=platform,
        min_os=min_os,
        sdk=sdk,
        has_entry_point=has_entry,
        code_signature=signature,
        problems=tuple(problems),
    )


def _read_signature(reader: _Reader, start: int, length: int) -> CodeSignature:
    blob = reader.read(start, length)
    if len(blob) < 12:
        return CodeSignature(present=True, note="signature data unreadable or truncated")
    magic, _length, count = struct.unpack_from(">III", blob, 0)
    if magic != _CSMAGIC_EMBEDDED_SIGNATURE:
        return CodeSignature(present=True, note=f"unrecognised signature container 0x{magic:x}")
    for index in range(min(count, 64)):
        pos = 12 + index * 8
        if pos + 8 > len(blob):
            break
        slot, blob_offset = struct.unpack_from(">II", blob, pos)
        if slot != _CSSLOT_CODEDIRECTORY:
            continue
        if blob_offset + 44 > len(blob):
            return CodeSignature(present=True, note="code directory truncated")
        cd_magic, _cd_len, version, flags, _hash_off, ident_off = struct.unpack_from(">IIIIII", blob, blob_offset)
        if cd_magic != _CSMAGIC_CODEDIRECTORY:
            return CodeSignature(present=True, note="code directory has an unexpected magic")
        identifier = _cstring(blob, blob_offset + ident_off) or None
        team_id = None
        if version >= 0x20200 and blob_offset + 52 <= len(blob):
            (team_off,) = struct.unpack_from(">I", blob, blob_offset + 48)
            if team_off:
                team_id = _cstring(blob, blob_offset + team_off) or None
        return CodeSignature(
            present=True,
            adhoc=bool(flags & _CS_ADHOC),
            linker_signed=bool(flags & _CS_LINKER_SIGNED),
            identifier=identifier,
            team_id=team_id,
        )
    return CodeSignature(present=True, note="no code directory in the signature")
