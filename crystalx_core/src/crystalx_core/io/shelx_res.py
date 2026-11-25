"""Reading SHELX ``.res`` output.

A ``.res`` is what SHELXT and SHELXL leave behind: the same header grammar as an ``.ins``, followed
by an atom list, optionally difference-Fourier ``Q`` peaks, and -- for SHELXL -- ``REM`` lines
carrying the R factors.

Atom rows are recognised structurally (a label that is not a known instruction, followed by an SFAC
index and three coordinates) rather than by position in the file, because where the atom list starts
differs between programs and between refinement stages. The imported engine keys off
``"PLAN" in line`` and ``"FVAR" in line``; that is exactly the fragility this reader avoids.

Anisotropic atoms span two physical lines joined by ``=``; the shared logical-line reader handles
that, so a six-component U is never truncated to look isotropic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from crystalx_core.domain.data import InsHeader
from crystalx_core.errors import ParseError
from crystalx_core.io.shelx_reader import logical_lines, read_ins_header
from crystalx_core.workspace.hashing import sha256_file

__all__ = [
    "ResAtom",
    "QPeak",
    "ResRFactors",
    "ResFile",
    "read_res",
    "SHELX_INSTRUCTIONS",
]

#: SHELX instruction keywords. A first token in this set is an instruction, not an atom label.
SHELX_INSTRUCTIONS: frozenset[str] = frozenset(
    """
    TITL CELL ZERR LATT SYMM SFAC DISP UNIT LAUE REM MORE TIME END HKLF OMIT SHEL BASF TWIN
    EXTI SWAT HOPE MERG SPEC RESI MOVE ANIS AFIX HFIX FRAG FEND EXYZ EADP EQIV OMIT CONN PART
    BIND FREE DFIX DANG BUMP SAME SADI CHIV FLAT DELU SIMU DEFS ISOR NCSY SUMP L.S. CGLS BLOC
    DAMP STIR WGHT FVAR BOND CONF MPLA RTAB HTAB LIST ACTA SIZE TEMP WPDB FMAP GRID PLAN
    MOLE TREF PATT TEXP HOPE INIT PHAS PSEU SPEC RIGU RIGO ABIN ANSC ANSR PRIG XNPD
    FIND PLOP PATS GROP NTRY PSMF TEST ESEL MIND
    """.split()
)

_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_'$]*$")
_Q_PEAK_RE = re.compile(r"^Q\d+$", re.IGNORECASE)
_SHELXL_CREATION_RE = re.compile(
    r"created by SHELXL-\d{4}/\d+ at \d{2}:\d{2}:\d{2} on \d{1,2}-[A-Za-z]{3}-\d{4}"
)

# SHELXL summary lines. Matched by regex rather than token position so that a changed field width
# or an extra word does not silently produce a wrong number.
_WR2_RE = re.compile(
    r"REM\s+wR2\s*=\s*([0-9.]+)\s*,\s*GooF\s*=\s*S\s*=\s*([0-9.]+)"
    r"(?:\s*,\s*Restrained\s+GooF\s*=\s*([0-9.]+))?",
    re.IGNORECASE,
)
_R1_RE = re.compile(
    r"REM\s+R1\s*=\s*([0-9.]+)\s+for\s+(\d+)\s+Fo\s*>\s*4sig\(Fo\)\s+and\s+([0-9.]+)"
    r"\s+for\s+all\s+(\d+)\s+data",
    re.IGNORECASE,
)
_PARAMS_RE = re.compile(
    r"REM\s+(\d+)\s+parameters\s+refined\s+using\s+(\d+)\s+restraints", re.IGNORECASE
)
_HIGHEST_DIFF_RE = re.compile(
    r"REM\s+Highest\s+difference\s+peak\s+([0-9.\-]+),\s+deepest\s+hole\s+([0-9.\-]+)"
    r"(?:,\s+1-sigma\s+level\s+([0-9.\-]+))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResAtom:
    """One atom from a ``.res`` file, as written."""

    label: str
    sfac_index: int
    x: float
    y: float
    z: float
    occupancy_code: float
    """The raw site-occupancy field. ``11.00000`` means 'fixed at 1.0'; see :attr:`occupancy`."""

    u_values: tuple[float, ...] = ()
    part: int | None = None
    residue: int | None = None
    afix: int | None = None
    line_no: int = 0

    @property
    def element_prefix(self) -> str:
        """The alphabetic prefix of the label, e.g. ``Ag`` from ``Ag1``.

        This is the label's own claim about the element. It is **not** authoritative: the SFAC index
        is. Both are kept so a disagreement between them stays visible.
        """
        match = re.match(r"^[A-Za-z]+", self.label)
        return match.group() if match else ""

    @property
    def occupancy(self) -> float:
        """Occupancy with the free-variable code removed.

        A value of the form ``10.xxxxx`` or ``11.xxxxx`` means "fixed at ``xxxxx``" (free variable 1
        times 10, plus the value). Anything below 10 is a genuine refined occupancy.
        """
        if self.occupancy_code >= 10.0:
            return self.occupancy_code - 10.0
        if self.occupancy_code <= -10.0:
            return self.occupancy_code + 10.0
        return self.occupancy_code

    @property
    def occupancy_is_fixed(self) -> bool:
        return abs(self.occupancy_code) >= 10.0

    @property
    def is_anisotropic(self) -> bool:
        return len(self.u_values) == 6

    @property
    def u_iso(self) -> float | None:
        return self.u_values[0] if len(self.u_values) == 1 else None


@dataclass(frozen=True)
class QPeak:
    """A difference-Fourier peak. These drive shell expansion in the onion workflow."""

    label: str
    x: float
    y: float
    z: float
    height: float
    rank: int
    sfac_index: int = 1
    occupancy_code: float = 11.0
    u_iso: float = 0.05


@dataclass(frozen=True)
class ResRFactors:
    """Refinement statistics recovered from SHELXL's ``REM`` summary lines.

    Every field is optional: a SHELXT ``.res`` has none of them, and a partially written file may
    have only some. Absent means absent, never zero.
    """

    r1_observed: float | None = None
    r1_all: float | None = None
    n_observed: int | None = None
    n_all: int | None = None
    wr2: float | None = None
    goof: float | None = None
    restrained_goof: float | None = None
    n_parameters: int | None = None
    n_restraints: int | None = None
    highest_peak: float | None = None
    deepest_hole: float | None = None
    one_sigma_level: float | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in (
                "r1_observed",
                "r1_all",
                "wr2",
                "goof",
                "n_parameters",
                "highest_peak",
            )
        )


@dataclass(frozen=True)
class ResFile:
    """A parsed ``.res``."""

    source: Path
    sha256: str
    header: InsHeader
    atoms: tuple[ResAtom, ...] = ()
    q_peaks: tuple[QPeak, ...] = ()
    r_factors: ResRFactors = field(default_factory=ResRFactors)
    rem_lines: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def non_hydrogen_atoms(self) -> tuple[ResAtom, ...]:
        return tuple(a for a in self.atoms if a.element_prefix.upper() != "H")

    def element_of(self, atom: ResAtom) -> str:
        """Element from the SFAC index, which is what SHELX actually refines.

        Falls back to the label prefix when the index is out of range, and says so via a warning at
        parse time rather than guessing quietly here.
        """
        sfac = self.header.sfac
        if 1 <= atom.sfac_index <= len(sfac):
            return sfac[atom.sfac_index - 1]
        return atom.element_prefix

    def label_element_disagreements(self) -> tuple[tuple[str, str, str], ...]:
        """Atoms whose label prefix contradicts their SFAC element.

        ``(label, element_from_sfac, element_from_label)``. A disagreement usually means someone
        renamed atoms without renumbering SFAC, and it is exactly the kind of thing that produces a
        confidently wrong structure.
        """
        out: list[tuple[str, str, str]] = []
        for atom in self.atoms:
            from_sfac = self.element_of(atom).strip().capitalize()
            from_label = atom.element_prefix.strip().capitalize()
            if from_label and from_sfac and from_label != from_sfac:
                out.append((atom.label, from_sfac, from_label))
        return tuple(out)

    def strongest_q_peaks(self, limit: int = 10) -> tuple[QPeak, ...]:
        return tuple(sorted(self.q_peaks, key=lambda p: -p.height)[:limit])


def read_res(path: Path | str) -> ResFile:
    """Parse a SHELX ``.res`` (or a ``.ins`` with an atom list)."""
    target = Path(path)
    if not target.is_file():
        raise ParseError(f"Result file not found: {target}", source=target)

    header = read_ins_header(target)

    atoms: list[ResAtom] = []
    q_peaks: list[QPeak] = []
    rem_lines: list[str] = []
    warnings: list[str] = []

    part: int | None = None
    residue: int | None = None
    afix: int | None = None
    in_tail = False
    """True once HKLF or END has been passed. SHELXL writes the difference-Fourier peak list and
    the final summary REM lines *after* END, so the scan must continue past it -- stopping at END
    silently discards every Q peak, which is precisely the input shell expansion needs."""

    q_rank = 0
    before_cell = True

    r1_observed = r1_all = wr2 = goof = restrained_goof = None
    n_observed = n_all = n_parameters = n_restraints = None
    highest_peak = deepest_hole = one_sigma = None

    for line in logical_lines(target):
        tokens = line.tokens
        if not tokens:
            continue
        keyword = tokens[0].upper()

        # SHELXL writes this indented provenance line between TITL and CELL.
        # Recognise only its exact banner grammar in the header: unknown text,
        # especially a damaged atom row, must continue to produce a warning.
        if before_cell and _SHELXL_CREATION_RE.fullmatch(line.text):
            continue
        if keyword == "CELL":
            before_cell = False

        if keyword == "REM":
            rem_lines.append(line.text)
            match = _WR2_RE.search(line.text)
            if match:
                wr2 = float(match.group(1))
                goof = float(match.group(2))
                if match.group(3):
                    restrained_goof = float(match.group(3))
            match = _R1_RE.search(line.text)
            if match:
                r1_observed = float(match.group(1))
                n_observed = int(match.group(2))
                r1_all = float(match.group(3))
                n_all = int(match.group(4))
            match = _PARAMS_RE.search(line.text)
            if match:
                n_parameters = int(match.group(1))
                n_restraints = int(match.group(2))
            match = _HIGHEST_DIFF_RE.search(line.text)
            if match:
                highest_peak = float(match.group(1))
                deepest_hole = float(match.group(2))
                if match.group(3):
                    one_sigma = float(match.group(3))
            continue

        if keyword in ("HKLF", "END"):
            in_tail = True
            continue
        if keyword == "PART":
            part = _optional_int(tokens, 1)
            if part == 0:
                part = None
            continue
        if keyword == "RESI":
            residue = _optional_int(tokens, 1)
            if residue == 0:
                residue = None
            continue
        if keyword == "AFIX":
            afix = _optional_int(tokens, 1)
            if afix == 0:
                afix = None
            continue
        if keyword in SHELX_INSTRUCTIONS:
            continue

        parsed = _parse_atom_line(tokens)
        if parsed is None:
            if _LABEL_RE.match(tokens[0]):
                warnings.append(
                    f"line {line.first_line_no}: looks like an atom but could not be parsed: "
                    f"{line.text[:70]}"
                )
            continue

        label, sfac_index, x, y, z, sof, u_values = parsed
        if in_tail and not _Q_PEAK_RE.match(label):
            # Past HKLF/END only the peak list is part of the record; anything else is a leftover.
            warnings.append(
                f"line {line.first_line_no}: ignoring {label} after HKLF/END; it is not a Q peak"
            )
            continue
        if _Q_PEAK_RE.match(label):
            q_rank += 1
            q_peaks.append(
                QPeak(
                    label=label,
                    x=x,
                    y=y,
                    z=z,
                    height=u_values[-1] if len(u_values) >= 2 else 0.0,
                    rank=q_rank,
                    sfac_index=sfac_index,
                    occupancy_code=sof,
                    u_iso=u_values[0] if u_values else 0.05,
                )
            )
            continue

        atoms.append(
            ResAtom(
                label=label,
                sfac_index=sfac_index,
                x=x,
                y=y,
                z=z,
                occupancy_code=sof,
                u_values=u_values,
                part=part,
                residue=residue,
                afix=afix,
                line_no=line.first_line_no,
            )
        )

    for atom in atoms:
        if not 1 <= atom.sfac_index <= len(header.sfac):
            warnings.append(
                f"atom {atom.label} has SFAC index {atom.sfac_index}, outside the "
                f"{len(header.sfac)}-element SFAC list; falling back to its label prefix"
            )

    return ResFile(
        source=target,
        sha256=sha256_file(target),
        header=header,
        atoms=tuple(atoms),
        q_peaks=tuple(q_peaks),
        r_factors=ResRFactors(
            r1_observed=r1_observed,
            r1_all=r1_all,
            n_observed=n_observed,
            n_all=n_all,
            wr2=wr2,
            goof=goof,
            restrained_goof=restrained_goof,
            n_parameters=n_parameters,
            n_restraints=n_restraints,
            highest_peak=highest_peak,
            deepest_hole=deepest_hole,
            one_sigma_level=one_sigma,
        ),
        rem_lines=tuple(rem_lines),
        warnings=tuple(warnings),
    )


def _parse_atom_line(
    tokens: list[str],
) -> tuple[str, int, float, float, float, float, tuple[float, ...]] | None:
    """Recognise ``label sfac x y z sof [U...]``.

    Returns ``None`` for anything that is not shaped like an atom, so the caller can decide whether
    that is worth a warning.
    """
    if len(tokens) < 6:
        return None
    label = tokens[0]
    if not _LABEL_RE.match(label):
        return None
    if label.upper() in SHELX_INSTRUCTIONS:
        return None
    try:
        sfac_index = int(tokens[1])
    except ValueError:
        return None
    try:
        values = [float(t) for t in tokens[2:]]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    x, y, z, sof = values[0], values[1], values[2], values[3]
    u_values = tuple(values[4:])
    return label, sfac_index, x, y, z, sof, u_values


def _optional_int(tokens: list[str], index: int) -> int | None:
    if len(tokens) <= index:
        return None
    try:
        return int(float(tokens[index]))
    except ValueError:
        return None
