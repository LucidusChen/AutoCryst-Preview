"""Readers and writers for crystallographic file formats.

Reading is token-anchored, never substring matching. Writing never overwrites a user file: a
candidate belongs in a run workspace, not beside the original.
"""

from __future__ import annotations

from crystalx_core.io.shelx_reader import (
    COMPLETENESS_UNAVAILABLE,
    logical_lines,
    read_ins_header,
    scan_hkl,
)
from crystalx_core.io.shelx_res import QPeak, ResAtom, ResFile, ResRFactors, read_res
from crystalx_core.io.shelx_writer import (
    SHELX_LINE_LIMIT,
    AtomRecord,
    InsDocument,
    format_atom_record,
    render_ins,
    wrap_shelx_line,
    write_ins,
)

__all__ = [
    "COMPLETENESS_UNAVAILABLE",
    "SHELX_LINE_LIMIT",
    "AtomRecord",
    "InsDocument",
    "QPeak",
    "ResAtom",
    "ResFile",
    "ResRFactors",
    "format_atom_record",
    "logical_lines",
    "read_ins_header",
    "read_res",
    "render_ins",
    "scan_hkl",
    "wrap_shelx_line",
    "write_ins",
]
