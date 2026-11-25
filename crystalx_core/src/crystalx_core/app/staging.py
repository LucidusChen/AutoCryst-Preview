"""Putting a case where a SHELX program can read it, and saying exactly what was changed.

SHELX programs take a *stem* and read ``<stem>.ins`` / ``<stem>.hkl`` from the current directory.
Staging copies the run's read-only input copies into a branch directory under a safe stem. By
default the instruction file is copied **byte for byte**: in particular ``LATT`` and ``SYMM`` are
kept, because SHELXT takes the Laue group from them and searches only space groups within it.
Removing them (as an earlier harness in this project did) silently restricts SHELXT to Laue
group -1.

The one permitted edit is an explicit composition override from the user, and it is recorded line
by line.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.workspace.hashing import sha256_file

__all__ = [
    "safe_stem",
    "CompositionOverrideError",
    "apply_composition_override",
    "StagedFiles",
    "stage_case",
]

_STEM_OK = re.compile(r"[^A-Za-z0-9_\-]+")
_NUMBER = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


def safe_stem(name: str, *, fallback: str = "case", max_length: int = 32) -> str:
    """A SHELX-safe stem derived from a user's file name.

    Only ``[A-Za-z0-9_-]`` survive; everything else becomes ``_``. The stem never comes from the
    user verbatim, so a file name cannot steer where a program writes.
    """
    stem = Path(name).stem if name else ""
    stem = _STEM_OK.sub("_", stem).strip("_-")[:max_length].strip("_-")
    if not stem or not stem[0].isalnum():
        return fallback
    return stem


class CompositionOverrideError(ValueError):
    """The override cannot be applied without changing something the user did not ask to change."""


def _is_atom_record(tokens: list[str]) -> bool:
    from crystalx_core.io.shelx_res import SHELX_INSTRUCTIONS

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


def apply_composition_override(
    ins_text: str, composition: tuple[tuple[str, float], ...]
) -> tuple[str, tuple[str, ...]]:
    """Replace ``SFAC``/``UNIT`` with the user's composition. Returns ``(text, changes)``.

    Refuses (raises :class:`CompositionOverrideError`) rather than guessing when:

    * the file already holds atom records -- renumbering SFAC would silently change their elements;
    * an ``SFAC`` line uses the long form with scattering-factor coefficients;
    * an ``SFAC``/``UNIT`` record is continued with ``=``.

    ``DISP`` records for elements no longer present are dropped, and each drop is recorded.
    """
    if not composition:
        return ins_text, ()
    elements = [symbol for symbol, _ in composition]
    counts = [count for _, count in composition]

    lines = ins_text.splitlines()
    changes: list[str] = []
    out: list[str] = []
    old_sfac: list[str] = []
    old_unit = ""
    sfac_written = False
    unit_written = False

    for index, line in enumerate(lines):
        tokens = line.split()
        word = tokens[0].upper() if tokens else ""
        if word == "END":
            out.extend(lines[index:])
            break
        if _is_atom_record(tokens):
            raise CompositionOverrideError(
                "The instruction file already contains atom records. Replacing SFAC would change "
                "which element each of them is, so the override is refused. Remove the atoms or "
                "leave the composition as written."
            )
        if word in ("SFAC", "UNIT") and line.rstrip().endswith("="):
            raise CompositionOverrideError(
                f"The {word} record is continued onto another line; it cannot be replaced safely."
            )
        if word == "SFAC":
            symbols = tokens[1:]
            if any(_NUMBER.match(t) for t in symbols):
                raise CompositionOverrideError(
                    "An SFAC record gives explicit scattering-factor coefficients (long form). "
                    "Replacing it would discard them, so the override is refused."
                )
            old_sfac.extend(symbols)
            if not sfac_written:
                out.append("SFAC " + " ".join(elements))
                sfac_written = True
            continue
        if word == "UNIT":
            old_unit = " ".join(tokens[1:])
            if not unit_written:
                out.append("UNIT " + " ".join(f"{c:g}" for c in counts))
                unit_written = True
            continue
        if word == "DISP" and len(tokens) > 1:
            if tokens[1].capitalize() not in elements:
                changes.append(f"dropped {line.strip()!r}: {tokens[1]} is not in the new composition")
                continue
        out.append(line)

    if not sfac_written:
        raise CompositionOverrideError("The instruction file has no SFAC record to replace.")
    if not unit_written:
        index = next(i for i, l in enumerate(out) if l.startswith("SFAC "))
        out.insert(index + 1, "UNIT " + " ".join(f"{c:g}" for c in counts))
        changes.append("inserted a UNIT record (the file had none)")

    changes.insert(0, f"SFAC {' '.join(old_sfac) or '(none)'} -> SFAC {' '.join(elements)}")
    changes.insert(1, f"UNIT {old_unit or '(none)'} -> UNIT {' '.join(f'{c:g}' for c in counts)}")
    return "\n".join(out) + "\n", tuple(changes)


@dataclass(frozen=True)
class StagedFiles:
    """What was put into a working directory, and how it relates to the run's input copies."""

    stem: str
    ins_path: Path
    hkl_path: Path
    ins_source_sha256: str
    ins_staged_sha256: str
    hkl_sha256: str
    changes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ins_is_verbatim(self) -> bool:
        return self.ins_source_sha256 == self.ins_staged_sha256

    def to_record(self) -> dict[str, object]:
        return {
            "stem": self.stem,
            "ins": self.ins_path.name,
            "hkl": self.hkl_path.name,
            "ins_source_sha256": self.ins_source_sha256,
            "ins_staged_sha256": self.ins_staged_sha256,
            "ins_is_verbatim": self.ins_is_verbatim,
            "hkl_sha256": self.hkl_sha256,
            "changes": list(self.changes),
        }


def stage_case(
    *,
    work_dir: Path,
    stem: str,
    ins_source: Path,
    hkl_source: Path,
    composition: tuple[tuple[str, float], ...] = (),
) -> StagedFiles:
    """Write ``<stem>.ins`` and ``<stem>.hkl`` into ``work_dir``. Refuses to overwrite."""
    if safe_stem(stem, fallback="") != stem:
        raise ValueError(f"Unsafe SHELX stem {stem!r}")
    ins_path = work_dir / f"{stem}.ins"
    hkl_path = work_dir / f"{stem}.hkl"
    for target in (ins_path, hkl_path):
        if target.exists():
            raise FileExistsError(f"{target.name} already exists in {work_dir}; refusing to overwrite")

    source_digest = sha256_file(ins_source)
    changes: tuple[str, ...] = ()
    if composition:
        text, changes = apply_composition_override(
            ins_source.read_text(encoding="utf-8", errors="replace"), composition
        )
        ins_path.write_bytes(text.encode("utf-8"))
        staged_digest = sha256_file(ins_path)
    else:
        shutil.copyfile(ins_source, ins_path)
        staged_digest = source_digest
    shutil.copyfile(hkl_source, hkl_path)
    return StagedFiles(
        stem=stem,
        ins_path=ins_path,
        hkl_path=hkl_path,
        ins_source_sha256=source_digest,
        ins_staged_sha256=staged_digest,
        hkl_sha256=sha256_file(hkl_path),
        changes=changes,
    )
