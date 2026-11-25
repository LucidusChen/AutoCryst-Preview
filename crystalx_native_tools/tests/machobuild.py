"""Build small, well-formed Mach-O files for tests. Header and load commands only; no code.

These are synthetic test fixtures written by the tests themselves -- not copies or excerpts of any
real program.
"""

from __future__ import annotations

import struct

CPU_X86_64 = 0x01000007
CPU_ARM64 = 0x0100000C
SUBTYPE_ARM64E = 2
MH_EXECUTE = 2
MH_DYLIB = 6
LC_LOAD_DYLIB = 0xC
LC_LOAD_WEAK_DYLIB = 0x80000018
LC_RPATH = 0x8000001C
LC_CODE_SIGNATURE = 0x1D
LC_MAIN = 0x80000028
LC_BUILD_VERSION = 0x32


def _pad(data: bytes, align: int = 8) -> bytes:
    return data + b"\0" * ((-len(data)) % align)


def dylib_command(name: str, cmd: int = LC_LOAD_DYLIB) -> bytes:
    payload = _pad(name.encode() + b"\0")
    size = 24 + len(payload)
    return struct.pack("<IIIIII", cmd, size, 24, 2, 0x10000, 0x10000) + payload


def rpath_command(path: str) -> bytes:
    payload = _pad(path.encode() + b"\0")
    return struct.pack("<III", LC_RPATH, 12 + len(payload), 12) + payload


def build_version(minos: tuple[int, int, int] = (12, 0, 0), sdk: tuple[int, int, int] = (14, 0, 0),
                  platform: int = 1) -> bytes:
    def enc(v):
        return (v[0] << 16) | (v[1] << 8) | v[2]

    return struct.pack("<IIIIII", LC_BUILD_VERSION, 24, platform, enc(minos), enc(sdk), 0)


def main_command() -> bytes:
    return struct.pack("<IIQQ", LC_MAIN, 24, 0, 0)


def signature_blob(identifier: str = "tool", *, adhoc: bool = True, team: str | None = None) -> bytes:
    """An embedded-signature SuperBlob holding one CodeDirectory (big-endian, as on disk)."""
    ident = identifier.encode() + b"\0"
    team_bytes = (team.encode() + b"\0") if team else b""
    header_len = 52
    ident_off = header_len
    team_off = header_len + len(ident) if team else 0
    flags = (0x2 if adhoc else 0) | (0x20000 if adhoc else 0)
    cd = struct.pack(">IIIIIIIIIBBBBIII", 0xFADE0C02, 0, 0x20400, flags, 0, ident_off, 0, 0, 0, 32, 2, 0, 12, 0, 0, team_off)
    cd = cd[:header_len] + ident + team_bytes
    cd = struct.pack(">II", 0xFADE0C02, len(cd)) + cd[8:]
    super_blob = struct.pack(">III", 0xFADE0CC0, 12 + 8 + len(cd), 1) + struct.pack(">II", 0, 20) + cd
    return super_blob


def thin(
    cputype: int = CPU_ARM64,
    *,
    subtype: int = 0,
    filetype: int = MH_EXECUTE,
    libraries: tuple[str, ...] = ("/usr/lib/libSystem.B.dylib",),
    weak: tuple[str, ...] = (),
    rpaths: tuple[str, ...] = (),
    minos: tuple[int, int, int] | None = (12, 0, 0),
    signed: bool | None = None,
    identifier: str = "tool",
    team: str | None = None,
    platform: int = 1,
) -> bytes:
    """A thin 64-bit little-endian Mach-O (``platform`` 1 = macOS, 2 = iOS). arm64 files are signed by default, like linker output."""
    if signed is None:
        signed = cputype == CPU_ARM64
    commands = []
    if minos is not None:
        commands.append(build_version(minos, platform=platform))
    commands.append(main_command())
    commands.extend(dylib_command(lib) for lib in libraries)
    commands.extend(dylib_command(lib, LC_LOAD_WEAK_DYLIB) for lib in weak)
    commands.extend(rpath_command(p) for p in rpaths)
    blob = signature_blob(identifier, team=team) if signed else b""
    ncmds = len(commands) + (1 if signed else 0)
    sizeofcmds = sum(len(c) for c in commands) + (16 if signed else 0)
    header_size = 32
    body_start = header_size + sizeofcmds
    data_offset = _pad(b"\0" * body_start, 16).__len__()
    if signed:
        commands.append(struct.pack("<IIII", LC_CODE_SIGNATURE, 16, data_offset, len(blob)))
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, cputype, subtype, filetype, ncmds, sizeofcmds, 0x200085, 0)
    image = header + b"".join(commands)
    image = image + b"\0" * (data_offset - len(image))
    return image + blob


def fat(*slices: bytes, cputypes: tuple[int, ...], subtypes: tuple[int, ...] | None = None) -> bytes:
    """A universal file wrapping the given thin images (big-endian fat header, 4 KiB aligned)."""
    subtypes = subtypes or tuple(0 for _ in slices)
    header = struct.pack(">II", 0xCAFEBABE, len(slices))
    offset = 4096
    table = b""
    body = b""
    for image, cputype, subtype in zip(slices, cputypes, subtypes):
        table += struct.pack(">iiIII", cputype, subtype, offset + len(body), len(image), 12)
        body += image + b"\0" * ((-len(image)) % 4096)
    head = header + table
    return head + b"\0" * (offset - len(head)) + body
