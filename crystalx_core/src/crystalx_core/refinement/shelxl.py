"""Bounded, preliminary SHELXL refinement of a solution candidate.

This stage takes a SHELXT ``<stem>_a.res`` (or any ``.res`` with atoms) and runs a **small, fixed
number of isotropic least-squares cycles**. It exists to turn a trial structure into something with
real R values, a difference map and a CIF, so that PLATON has something to check and a person has
something to look at.

It is deliberately *not* a refinement strategy:

* no anisotropic displacement parameters, no hydrogen placement, no restraints, no disorder;
* no element reassignment, no deletion of atoms, no peak picking;
* a fixed cycle count, not "until converged".

So its output is labelled **preliminary** everywhere it appears. A low R1 after four isotropic
cycles on a heavy-atom skeleton says the heavy atoms are roughly right; it says nothing about the
light atoms, the hand, the space group, or whether the model is chemically sensible.

## What the instruction rewrite changes, and why

Every edit is returned in ``changes`` and written into the run record, because a silent edit to a
crystallographer's input is exactly what this project refuses to do.

==================  =====================================================================
Instruction         Treatment
==================  =====================================================================
``L.S.``/``CGLS``   replaced by the explicit bounded method (default ``L.S. n``)
``PLAN``            replaced by ``PLAN 20`` -- difference peaks for the viewer
``LIST``            replaced by ``LIST 4`` -- writes the ``.fcf``. **PLATON needs it**: without
                    an FCF, PLATON launches its own external SHELXL job to regenerate one,
                    which on this machine would be an uncontrolled second x86_64 process
``ACTA``            ensured for L.S.; omitted for CGLS (native-incompatible)
``FMAP``            ``FMAP 2`` -- difference Fourier
``ANIS``            removed when refining isotropically
``Q`` peaks         removed from the atom list; SHELXL regenerates them
``FVAR``            inserted only if absent (SHELXL requires an overall scale factor)
anything else       kept verbatim
==================  =====================================================================
"""

from __future__ import annotations

import re
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.io.shelx_reader import logical_lines
from crystalx_core.tools.runner import ResourceLimits

__all__ = [
    "RefinementSettings",
    "PreparedRefinement",
    "prepare_refinement_ins",
    "stage_refinement",
    "CONTROL_INSTRUCTIONS",
]

#: Instructions this stage owns. Existing copies are removed and replaced by the stage's own.
CONTROL_INSTRUCTIONS = frozenset({"L.S.", "CGLS", "PLAN", "LIST", "ACTA", "FMAP"})

#: Header instructions after which the control block is inserted.
_HEADER_INSTRUCTIONS = frozenset(
    {"TITL", "CELL", "ZERR", "LATT", "SYMM", "SFAC", "DISP", "UNIT", "REM", "TEMP", "SIZE"}
)

_Q_PEAK = re.compile(r"^Q\d+$", re.IGNORECASE)
_INSTRUCTION_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9.]{0,3}$")


@dataclass(frozen=True)
class RefinementSettings:
    """The bounds of a preliminary refinement. Recorded verbatim in the run."""

    cycles: int = 4
    """Least-squares cycles. Small on purpose: this is a first look, not a finished model."""

    isotropic: bool = True
    method: str = "ls"
    damp_factor: float | None = None
    damp_shift_esd_limit: float = 15.0
    plan_peaks: int = 20
    fourier_map: int = 2
    write_cif: bool = True
    write_fcf: bool = True
    strip_q_peaks: bool = True
    timeout_s: float = 900.0
    limits: ResourceLimits | None = None

    def __post_init__(self) -> None:
        if self.method not in ("ls", "cgls"):
            raise ValueError("refinement method must be ls or cgls")
        if self.damp_factor is not None and (type(self.damp_factor) not in (int, float) or not math.isfinite(self.damp_factor) or self.damp_factor < 0):
            raise ValueError("DAMP factor must be finite and nonnegative")
        if type(self.damp_shift_esd_limit) not in (int, float) or not math.isfinite(self.damp_shift_esd_limit) or self.damp_shift_esd_limit < 0:
            raise ValueError("DAMP shift/esd limit must be finite and nonnegative")
        if not 1 <= self.cycles <= 50:
            raise ValueError("RefinementSettings.cycles must be between 1 and 50")
        if type(self.plan_peaks) is not int or not 0 <= self.plan_peaks <= 999:
            raise ValueError("RefinementSettings.plan_peaks must be between 0 and 999")
        if type(self.fourier_map) is not int or self.fourier_map not in (2,3,4):
            raise ValueError("RefinementSettings.fourier_map must be 2, 3 or 4")

    def to_record(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "isotropic": self.isotropic,
            "method": self.method, "damp_factor": self.damp_factor,
            "damp_shift_esd_limit": self.damp_shift_esd_limit,
            "plan_peaks": self.plan_peaks,
            "fourier_map": self.fourier_map,
            "write_cif": self.write_cif,
            "write_fcf": self.write_fcf,
            "strip_q_peaks": self.strip_q_peaks,
            "timeout_s": self.timeout_s,
            "limits": self.limits.to_record() if self.limits else None,
        }


@dataclass(frozen=True)
class PreparedRefinement:
    """The refinement input as written, plus an account of every change made to it."""

    workdir: Path
    stem: str
    ins_path: Path
    hkl_path: Path
    source_res: Path
    changes: tuple[str, ...] = field(default_factory=tuple)
    atoms_kept: int = 0
    q_peaks_removed: int = 0

    @property
    def expected_outputs(self) -> tuple[str, ...]:
        return (f"{self.stem}.res", f"{self.stem}.lst")


def prepare_refinement_ins(
    solution_text: str, settings: RefinementSettings
) -> tuple[str, tuple[str, ...], int, int]:
    """Rewrite a solution ``.res`` into a bounded refinement ``.ins``.

    Returns ``(text, changes, atoms_kept, q_peaks_removed)``. Lines between the header and ``HKLF``
    that this stage does not own are kept byte-for-byte, including ``=`` continuations.
    """
    raw_lines = solution_text.splitlines()
    changes: list[str] = []
    out: list[str] = []
    atoms_kept = 0
    q_removed = 0
    saw_fvar = False
    saw_hklf = False
    control_inserted = False
    removed: dict[str, int] = {}
    if settings.method == "cgls" and settings.write_cif:
        changes.append("deferred ACTA/CIF: CGLS is incompatible with ACTA; native CIF requires an explicitly assessed nonzero LS refinement; zero-cycle and zero-shift ACTA are unsupported locally")
    if settings.fourier_map != 2 and settings.write_cif:
        changes.append("deferred ACTA/CIF: FMAP 3/4 is incompatible with ACTA; preserve RES/FCF without an unbudgeted FMAP 2 refinement")

    def control_block() -> list[str]:
        command = "CGLS" if settings.method == "cgls" else "L.S."
        block = [f"{command} {settings.cycles}", f"PLAN {settings.plan_peaks}"]
        if settings.damp_factor is not None:
            block.append(f"DAMP {settings.damp_factor:g} {settings.damp_shift_esd_limit:g}")
        if settings.write_fcf:
            block.append("LIST 4")
        block.append(f"FMAP {settings.fourier_map}")
        if settings.write_cif and settings.method != "cgls" and settings.fourier_map==2:
            block.append("ACTA")
        return block

    for logical in _logical_from_text(raw_lines):
        word = logical.instruction
        if word == "END":
            break
        if word in CONTROL_INSTRUCTIONS or (word == "DAMP" and settings.damp_factor is not None) or (settings.isotropic and word == "ANIS"):
            removed[word] = removed.get(word, 0) + 1
            continue

        is_atom = _is_atom_line(logical.text)
        if is_atom and settings.strip_q_peaks and _Q_PEAK.match(logical.text.split()[0]):
            q_removed += 1
            continue

        if not control_inserted and (word == "FVAR" or is_atom or word == "HKLF"):
            out.extend(control_block())
            control_inserted = True
            changes.append("inserted control block: " + "; ".join(control_block()))
        if is_atom and not saw_fvar:
            out.append("FVAR 1.00000")
            saw_fvar = True
            changes.append("inserted FVAR 1.00000 (no overall scale factor was present)")

        if word == "FVAR":
            saw_fvar = True
        if is_atom:
            atoms_kept += 1
        out.extend(logical.raw_lines)

        if word == "HKLF":
            saw_hklf = True
            break

    if not control_inserted:
        out.extend(control_block())
        changes.append("inserted control block at end: " + "; ".join(control_block()))
    if not saw_hklf:
        out.append("HKLF 4")
        changes.append("appended HKLF 4 (the solution file carried none)")
    out.append("END")

    for word, count in sorted(removed.items()):
        changes.append(f"removed {count} existing {word} instruction(s); this stage sets its own")
    if q_removed:
        changes.append(f"removed {q_removed} Q peak(s) from the atom list; SHELXL regenerates them")
    if atoms_kept == 0:
        changes.append("WARNING: the solution contains no atoms; SHELXL will have nothing to refine")
    return "\n".join(out) + "\n", tuple(changes), atoms_kept, q_removed


def stage_refinement(
    *,
    solution_res: Path,
    hkl: Path,
    workdir: Path,
    stem: str,
    settings: RefinementSettings,
) -> PreparedRefinement:
    """Write ``<stem>.ins`` and copy the reflections to ``<stem>.hkl`` inside ``workdir``.

    The reflection file is **copied**, never linked: a refinement directory must stay valid even if
    the run's input copy is later moved, and nothing SHELXL does can reach the original.
    """
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", stem):
        raise ValueError(f"Unsafe refinement stem {stem!r}")
    workdir.mkdir(parents=True, exist_ok=True)
    text, changes, atoms_kept, q_removed = prepare_refinement_ins(
        solution_res.read_text(encoding="utf-8", errors="replace"), settings
    )
    ins_path = workdir / f"{stem}.ins"
    hkl_path = workdir / f"{stem}.hkl"
    if ins_path.exists() or hkl_path.exists():
        raise FileExistsError(f"Refinement inputs already exist in {workdir}; refusing to overwrite")
    ins_path.write_text(text, encoding="utf-8")
    shutil.copyfile(hkl, hkl_path)
    return PreparedRefinement(
        workdir=workdir,
        stem=stem,
        ins_path=ins_path,
        hkl_path=hkl_path,
        source_res=solution_res,
        changes=changes,
        atoms_kept=atoms_kept,
        q_peaks_removed=q_removed,
    )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Logical:
    text: str
    raw_lines: tuple[str, ...]

    @property
    def instruction(self) -> str:
        tokens = self.text.split()
        return tokens[0].upper() if tokens else ""


def _logical_from_text(lines: list[str]):
    """Join SHELX ``=`` continuations without touching the raw text that is written back."""
    buffer: list[str] = []
    raw: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        raw.append(line.rstrip("\r"))
        content = stripped.strip() if buffer else stripped
        if content.endswith("=") and not content.upper().startswith(("TITL", "REM")):
            buffer.append(content[:-1].rstrip())
            continue
        buffer.append(content)
        yield _Logical(" ".join(part for part in buffer if part).strip(), tuple(raw))
        buffer, raw = [], []
    if buffer:
        yield _Logical(" ".join(buffer).strip(), tuple(raw))


def _is_atom_line(text: str) -> bool:
    """``label sfac x y z ...`` with a label that is not an instruction keyword."""
    from crystalx_core.io.shelx_res import SHELX_INSTRUCTIONS

    tokens = text.split()
    if len(tokens) < 5 or tokens[0].upper() in SHELX_INSTRUCTIONS:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9_'$]*$", tokens[0]):
        return False
    try:
        int(tokens[1])
        float(tokens[2]), float(tokens[3]), float(tokens[4])
    except ValueError:
        return False
    return True


_ = logical_lines  # the file-based joiner; this module needs the text-based variant above
