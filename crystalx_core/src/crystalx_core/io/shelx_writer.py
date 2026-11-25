"""Writing SHELX instruction files.

Two rules govern this module.

**Never write over a user's file.** Every entry point takes an explicit destination and refuses to
overwrite unless the caller says so. Candidate files belong in a run workspace, not beside the
originals.

**A lock must be visible in the file.** When a :class:`LockDirective` fixes a coordinate, the writer
emits SHELXL's free-variable form (``10.xxxxx`` means "fix at ``0.xxxxx``") and the directive's own
``instruction_lines`` are emitted as comments above the affected block, so a crystallographer reading
the ``.ins`` can see what was imposed and why without consulting the run record.

Line-length handling: SHELX reads the first 80 columns of an instruction, so long ``SYMM``, ``SFAC``
and ``UNIT`` records are wrapped with the ``=`` continuation marker rather than being silently
truncated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from crystalx_core.domain.data import HklfFormat, InsHeader, UnitCell
from crystalx_core.domain.hypothesis import LockDirective, LockMode, format_fixed_shelxl_value
from crystalx_core.errors import InputValidationError
from crystalx_core.domain.ids import AtomId

__all__ = [
    "SHELX_LINE_LIMIT",
    "AtomRecord",
    "InsDocument",
    "wrap_shelx_line",
    "format_atom_record",
    "render_ins",
    "write_ins",
]

#: SHELX reads the first 80 characters of an instruction line.
SHELX_LINE_LIMIT = 80

#: SHELXL free-variable code meaning "occupancy fixed at 1.0".
FIXED_FULL_OCCUPANCY = "11.00000"


@dataclass(frozen=True)
class AtomRecord:
    """One atom row in a SHELX instruction file.

    ``sfac_index`` is 1-based into the ``SFAC`` list, which is how SHELX identifies the element.
    ``u_values`` is either a single isotropic U or the six anisotropic components in SHELX order
    (U11, U22, U33, U23, U13, U12).
    """

    label: str
    sfac_index: int
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    u_values: tuple[float, ...] = (0.05,)
    fix_coordinates: bool = False
    fix_occupancy: bool = True
    part: int | None = None
    residue: int | None = None
    comment: str = ""

    def __post_init__(self) -> None:
        if self.sfac_index < 1:
            raise InputValidationError(
                f"SFAC index for atom {self.label!r} must be 1-based, got {self.sfac_index}"
            )
        if len(self.u_values) not in (1, 6):
            raise InputValidationError(
                f"Atom {self.label!r} needs 1 isotropic or 6 anisotropic U values, "
                f"got {len(self.u_values)}"
            )
        if not self.label.strip():
            raise InputValidationError("An atom record needs a label")

    @property
    def atom_id(self) -> AtomId:
        return AtomId(self.label)

    @property
    def is_anisotropic(self) -> bool:
        return len(self.u_values) == 6


def format_atom_record(atom: AtomRecord) -> str:
    """Render one atom row.

    A fixed coordinate is written as ``10.xxxxx``. This is the only place that convention is
    applied on the writing side; :func:`crystalx_core.domain.hypothesis.format_fixed_shelxl_value`
    owns the arithmetic.
    """
    if atom.fix_coordinates:
        coords = [format_fixed_shelxl_value(value) for value in (atom.x, atom.y, atom.z)]
    else:
        coords = [f"{value:.5f}" for value in (atom.x, atom.y, atom.z)]

    if atom.fix_occupancy and abs(atom.occupancy - 1.0) < 1e-9:
        sof = FIXED_FULL_OCCUPANCY
    elif atom.fix_occupancy:
        sof = f"{10.0 + atom.occupancy:.5f}"
    else:
        sof = f"{atom.occupancy:.5f}"

    fields = [f"{atom.label:<6s}", f"{atom.sfac_index:>3d}", *coords, sof]
    fields.extend(f"{value:.5f}" for value in atom.u_values)
    return " ".join(fields)


def wrap_shelx_line(line: str, *, limit: int = SHELX_LINE_LIMIT) -> list[str]:
    """Split an over-long instruction across ``=`` continuation lines.

    Splitting happens on whitespace only, so a numeric field is never cut in half. A single token
    longer than the limit is emitted as-is rather than corrupted; that situation means the caller
    built something SHELX could not have read anyway.
    """
    if len(line) <= limit:
        return [line]
    tokens = line.split()
    if not tokens:
        return [line]

    wrapped: list[str] = []
    current = tokens[0]
    for token in tokens[1:]:
        # +3 leaves room for " " before the token and " =" at the end of the line.
        if len(current) + 1 + len(token) + 2 <= limit:
            current = f"{current} {token}"
        else:
            wrapped.append(f"{current} =")
            current = f"     {token}"
    wrapped.append(current)
    return wrapped


@dataclass
class InsDocument:
    """An instruction file being assembled.

    Deliberately a builder rather than a template: a candidate ``.ins`` is produced by taking a
    parsed header, replacing the atom list, and adding instructions, and every one of those steps
    should be explicit and inspectable.
    """

    title: str
    cell: UnitCell
    wavelength_a: float
    z: float
    sfac: tuple[str, ...]
    unit: tuple[float, ...]
    latt: int | None = None
    symm: tuple[str, ...] = ()
    zerr: tuple[float, ...] = ()
    pre_atom_instructions: list[str] = field(default_factory=list)
    atoms: list[AtomRecord] = field(default_factory=list)
    post_atom_instructions: list[str] = field(default_factory=list)
    hklf: HklfFormat = HklfFormat.HKLF4
    provenance_comments: list[str] = field(default_factory=list)
    lock_directives: list[LockDirective] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.sfac) != len(self.unit):
            raise InputValidationError(
                f"SFAC lists {len(self.sfac)} elements but UNIT lists {len(self.unit)} counts"
            )
        if not self.sfac:
            raise InputValidationError("An instruction file needs at least one SFAC element")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_header(
        cls,
        header: InsHeader,
        *,
        title: str | None = None,
        atoms: Sequence[AtomRecord] = (),
    ) -> "InsDocument":
        """Start from a parsed header, keeping cell, symmetry and composition intact."""
        if header.cell is None:
            raise InputValidationError(
                f"Cannot build an instruction file from {header.source}: it has no CELL record"
            )
        if header.wavelength_a is None:
            raise InputValidationError(
                f"Cannot build an instruction file from {header.source}: it has no wavelength"
            )
        return cls(
            title=title if title is not None else header.title,
            cell=header.cell,
            wavelength_a=header.wavelength_a,
            z=header.z if header.z is not None else 1.0,
            sfac=header.sfac,
            unit=header.unit,
            latt=header.latt,
            symm=header.symm,
            atoms=list(atoms),
            hklf=header.hklf_format or HklfFormat.HKLF4,
        )

    def sfac_index(self, element: str) -> int:
        """1-based SFAC index for an element, raising rather than guessing."""
        target = element.strip().capitalize()
        for index, symbol in enumerate(self.sfac, start=1):
            if symbol.strip().capitalize() == target:
                return index
        raise InputValidationError(
            f"Element {element!r} is not in the SFAC list {list(self.sfac)}. "
            "Add it to SFAC/UNIT before writing an atom of that type."
        )

    def add_atom(self, atom: AtomRecord) -> None:
        self.atoms.append(atom)

    def apply_lock(self, directive: LockDirective) -> None:
        """Apply a lock directive to the atoms it names, and keep the directive for the header.

        ``HARD_LOCK`` sets the fixed-coordinate flag on the matching atoms. ``SOFT_RESTRAINT`` emits
        the directive's own restraint lines. ``EXPLORATORY`` frees the occupancy so the data can
        vote on whether the site is real. Nothing is applied silently: in every case the directive's
        reason is written into the file as a comment.
        """
        named = {str(a) for a in directive.atom_ids}
        if directive.mode is LockMode.HARD_LOCK:
            self.atoms = [
                _with_fixed_coordinates(atom) if atom.label in named else atom
                for atom in self.atoms
            ]
        elif directive.mode is LockMode.EXPLORATORY:
            self.atoms = [
                _with_free_occupancy(atom) if atom.label in named else atom for atom in self.atoms
            ]
        self.lock_directives.append(directive)

    # ------------------------------------------------------------------ rendering
    def render(self) -> str:
        return render_ins(self)


def _with_fixed_coordinates(atom: AtomRecord) -> AtomRecord:
    return AtomRecord(
        label=atom.label,
        sfac_index=atom.sfac_index,
        x=atom.x,
        y=atom.y,
        z=atom.z,
        occupancy=atom.occupancy,
        u_values=atom.u_values,
        fix_coordinates=True,
        fix_occupancy=True,
        part=atom.part,
        residue=atom.residue,
        comment=atom.comment,
    )


def _with_free_occupancy(atom: AtomRecord) -> AtomRecord:
    return AtomRecord(
        label=atom.label,
        sfac_index=atom.sfac_index,
        x=atom.x,
        y=atom.y,
        z=atom.z,
        occupancy=atom.occupancy,
        u_values=atom.u_values[:1],
        fix_coordinates=False,
        fix_occupancy=False,
        part=atom.part,
        residue=atom.residue,
        comment=atom.comment,
    )


def render_ins(document: InsDocument) -> str:
    """Render a complete instruction file as text."""
    lines: list[str] = []

    for comment in document.provenance_comments:
        for chunk in _comment_lines(comment):
            lines.append(chunk)

    lines.append(f"TITL {document.title}".rstrip())
    cell = document.cell
    lines.extend(
        wrap_shelx_line(
            "CELL "
            + " ".join(
                f"{value:.5f}" if index == 0 else f"{value:.4f}"
                for index, value in enumerate(
                    (
                        document.wavelength_a,
                        cell.a,
                        cell.b,
                        cell.c,
                        cell.alpha,
                        cell.beta,
                        cell.gamma,
                    )
                )
            )
        )
    )
    if document.zerr:
        lines.extend(
            wrap_shelx_line("ZERR " + " ".join(f"{value:g}" for value in document.zerr))
        )
    else:
        lines.append(f"ZERR {document.z:g} 0.0000 0.0000 0.0000 0.000 0.000 0.000")

    if document.latt is not None:
        lines.append(f"LATT {document.latt}")
    for operator in document.symm:
        lines.extend(wrap_shelx_line(f"SYMM {operator}"))

    lines.extend(wrap_shelx_line("SFAC " + " ".join(document.sfac)))
    lines.extend(wrap_shelx_line("UNIT " + " ".join(f"{value:g}" for value in document.unit)))

    for instruction in document.pre_atom_instructions:
        lines.extend(wrap_shelx_line(instruction))

    if document.lock_directives:
        lines.append("")
        for directive in document.lock_directives:
            lines.extend(
                _comment_lines(
                    f"lock {directive.mode.value} on "
                    f"{', '.join(str(a) for a in directive.atom_ids)}: {directive.reason}"
                )
            )
        for directive in document.lock_directives:
            if directive.mode is LockMode.SOFT_RESTRAINT:
                for line in directive.instruction_lines:
                    lines.extend(wrap_shelx_line(line.rstrip("\n")))

    if document.atoms:
        lines.append("")
        current_part: int | None = None
        current_residue: int | None = None
        for atom in document.atoms:
            if atom.residue != current_residue:
                if current_residue is not None:
                    lines.append("RESI 0")
                if atom.residue is not None:
                    lines.append(f"RESI {atom.residue}")
                current_residue = atom.residue
            if atom.part != current_part:
                if current_part is not None:
                    lines.append("PART 0")
                if atom.part is not None:
                    lines.append(f"PART {atom.part}")
                current_part = atom.part
            if atom.comment:
                lines.extend(_comment_lines(atom.comment))
            lines.extend(wrap_shelx_line(format_atom_record(atom)))
        if current_part is not None:
            lines.append("PART 0")
        if current_residue is not None:
            lines.append("RESI 0")

    for instruction in document.post_atom_instructions:
        lines.extend(wrap_shelx_line(instruction))

    lines.append(f"HKLF {int(document.hklf)}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_ins(
    document: InsDocument, destination: Path | str, *, allow_overwrite: bool = False
) -> Path:
    """Write an instruction file, refusing to clobber an existing one by default."""
    target = Path(destination)
    if target.exists() and not allow_overwrite:
        raise InputValidationError(
            f"Refusing to overwrite {target}. Candidate files belong in a run workspace; "
            "pass allow_overwrite=True only when you own the destination.",
            evidence_paths=[target],
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document.render(), encoding="utf-8")
    return target


def _comment_lines(text: str) -> Iterable[str]:
    """Render a comment as one or more ``REM`` lines inside the column limit."""
    prefix = "REM "
    budget = SHELX_LINE_LIMIT - len(prefix)
    words = text.split()
    if not words:
        return [prefix.rstrip()]
    out: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= budget:
            current = f"{current} {word}"
        else:
            out.append(prefix + current)
            current = word
    out.append(prefix + current)
    return out
