"""The built-in synthetic demo case.

A **fictional** small zinc complex -- an aqua bis(glycinato)zinc unit with a methanol of
crystallisation in P2(1) -- whose reflections are *calculated* from the model with crude form
factors, an overall temperature factor and deterministic noise. It is generated at runtime, never
shipped as a data file, and has no relation to any deposited or measured structure.

It exists for two reasons only:

* the application demo, where the mock toolchain (``app.mock_toolchain``) stands in for SHELXT,
  SHELXL and PLATON -- and refuses to run on anything that is not byte-identical to this case;
* tests that need a realistic-looking input without touching a single byte of real data.

Two features are deliberate:

* the methanol straddles the *a* cell edge (C5 near x = 1, O6 near x = 0.18 in the file), so the
  viewer's handling of a bond across a cell boundary is exercised -- it must list the bond, not
  draw a line through the cell;
* the space group is P2(1), a Sohncke group, so no chirality precondition is involved.
"""

from __future__ import annotations

import functools
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEMO_MARKER",
    "DEMO_STEM",
    "DEMO_CELL",
    "DemoAtom",
    "demo_atoms",
    "demo_case_texts",
    "demo_digests",
    "write_demo_case",
    "is_demo_input",
]

#: Appears in the title and a REM line of every demo file, and in every mock output.
DEMO_MARKER = "CRYSTALX-SYNTHETIC-DEMO"
DEMO_STEM = "crystalx_demo"
DEMO_CELL = (7.5, 9.2, 10.4, 90.0, 101.0, 90.0)
DEMO_WAVELENGTH = 0.71073
_SEED = 20261013
_D_MIN = 0.84
_B_OVERALL = 3.0

_ZN_FRAC = (0.25, 0.30, 0.20)
#: One glycinate (O1 carboxylate O bound to Zn, C1 carboxyl C, C2 methylene, N1 amine, O2 free O),
#: as Cartesian offsets from Zn in angstrom. The second ligand is generated from it.
_LIGAND = (
    ("O1", "O", (1.95, 0.35, 0.0)),
    ("C1", "C", (2.65, 1.45, 0.0)),
    ("C2", "C", (1.85, 2.70, 0.0)),
    ("N1", "N", (0.45, 2.00, 0.0)),
    ("O2", "O", (3.80, 1.95, 0.0)),
)
_SECOND_LABELS = ("O3", "C3", "C4", "N2", "O4")
_WATER = ("O5", "O", (0.0, 0.0, 2.05))
_METHANOL_C = (0.985, 0.72, 0.425)
_METHANOL_CO = 1.42

_SFAC = ("C", "H", "N", "O", "Zn")
_UNIT = (10, 28, 4, 12, 2)
_Z = {"C": 6, "N": 7, "O": 8, "Zn": 30}
_FORM_B = {"C": 10.0, "N": 10.0, "O": 10.0, "Zn": 6.0}


@dataclass(frozen=True)
class DemoAtom:
    label: str
    element: str
    frac: tuple[float, float, float]


def _matrix() -> tuple[tuple[float, float, float], ...]:
    from crystalx_core.geometry.model import orthogonalisation_matrix

    return orthogonalisation_matrix(DEMO_CELL)


def _to_frac(m, cart: tuple[float, float, float]) -> tuple[float, float, float]:
    z = cart[2] / m[2][2]
    y = (cart[1] - m[1][2] * z) / m[1][1]
    x = (cart[0] - m[0][1] * y - m[0][2] * z) / m[0][0]
    return (x, y, z)


@functools.lru_cache(maxsize=1)
def demo_atoms() -> tuple[DemoAtom, ...]:
    """The model the demo reflections are calculated from (asymmetric unit, 14 non-H atoms)."""
    from crystalx_core.geometry.model import frac_to_cart

    m = _matrix()
    zn_cart = frac_to_cart(m, _ZN_FRAC)

    def at(offset: tuple[float, float, float]) -> tuple[float, float, float]:
        frac = _to_frac(m, tuple(zn_cart[i] + offset[i] for i in range(3)))
        return tuple(round(v, 4) for v in frac)  # type: ignore[return-value]

    atoms = [DemoAtom("Zn1", "Zn", _ZN_FRAC)]
    atoms += [DemoAtom(label, element, at(offset)) for label, element, offset in _LIGAND]
    for (label, element, offset), new_label in zip(_LIGAND, _SECOND_LABELS):
        atoms.append(DemoAtom(new_label, element, at((-offset[0], -offset[1], offset[2] + 0.3))))
    atoms.append(DemoAtom(_WATER[0], _WATER[1], at(_WATER[2])))

    c_cart = frac_to_cart(m, _METHANOL_C)
    o_frac = _to_frac(m, (c_cart[0] + _METHANOL_CO, c_cart[1], c_cart[2]))
    o_frac = (o_frac[0] - math.floor(o_frac[0]), o_frac[1], o_frac[2])
    atoms.append(DemoAtom("C5", "C", _METHANOL_C))
    atoms.append(DemoAtom("O6", "O", tuple(round(v, 4) for v in o_frac)))  # type: ignore[arg-type]
    return tuple(atoms)


def _reflections() -> list[tuple[int, int, int, float, float]]:
    from crystalx_core.domain.data import UnitCell

    cell = UnitCell(*DEMO_CELL)
    rng = random.Random(_SEED)
    atoms = demo_atoms()
    # P2(1): (x, y, z) and (-x, y + 1/2, -z)
    positions = [
        (a.element, p)
        for a in atoms
        for p in (a.frac, (-a.frac[0], a.frac[1] + 0.5, -a.frac[2]))
    ]
    limit = [int(math.ceil(length / _D_MIN)) for length in DEMO_CELL[:3]]
    raw: list[tuple[int, int, int, float]] = []
    for h in range(-limit[0], limit[0] + 1):
        for k in range(0, limit[1] + 1):
            for l in range(0, limit[2] + 1):
                if (h, k, l) == (0, 0, 0) or (l == 0 and h < 0):
                    continue  # one representative per 2/m orbit
                d = cell.d_spacing(h, k, l)
                if d < _D_MIN:
                    continue
                s2 = 1.0 / (4.0 * d * d)
                re_part = im_part = 0.0
                for element, (x, y, z) in positions:
                    f = _Z[element] * math.exp(-_FORM_B[element] * s2) * math.exp(-_B_OVERALL * s2)
                    phase = 2.0 * math.pi * (h * x + k * y + l * z)
                    re_part += f * math.cos(phase)
                    im_part += f * math.sin(phase)
                raw.append((h, k, l, re_part * re_part + im_part * im_part))
    scale = 9000.0 / max(i for *_, i in raw)
    out = []
    for h, k, l, intensity in raw:
        i_true = intensity * scale
        sigma = 0.6 + 0.5 * math.sqrt(i_true) + 0.02 * i_true
        i_obs = i_true + rng.gauss(0.0, sigma)
        out.append((h, k, l, max(min(i_obs, 99999.0), -999.0), min(sigma, 9999.0)))
    return out


@functools.lru_cache(maxsize=1)
def demo_case_texts() -> tuple[str, str]:
    """``(ins_text, hkl_text)``. Deterministic on a given machine."""
    a, b, c, alpha, beta, gamma = DEMO_CELL
    ins_lines = [
        f"TITL {DEMO_STEM} {DEMO_MARKER} fictional generated data, not a measurement",
        f"REM {DEMO_MARKER} v1: generated by crystalx_core.app.demo for the application demo.",
        "REM Fictional aqua bis(glycinato)zinc methanol solvate; reflections calculated from the",
        "REM model with crude form factors and noise. Not experimental data; do not use for science.",
        f"CELL {DEMO_WAVELENGTH:.5f} {a:.4f} {b:.4f} {c:.4f} {alpha:.3f} {beta:.3f} {gamma:.3f}",
        "ZERR 2 0.0010 0.0012 0.0013 0.000 0.002 0.000",
        "LATT -1",
        "SYMM -X, 1/2+Y, -Z",
        "SFAC " + " ".join(_SFAC),
        "UNIT " + " ".join(str(u) for u in _UNIT),
        "HKLF 4",
        "END",
    ]
    hkl_lines = [f"{h:4d}{k:4d}{l:4d}{i:8.2f}{s:8.2f}" for h, k, l, i, s in _reflections()]
    hkl_lines.append(f"{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}")
    return "\n".join(ins_lines) + "\n", "\n".join(hkl_lines) + "\n"


@functools.lru_cache(maxsize=1)
def demo_digests() -> tuple[str, str]:
    """SHA-256 of the demo ``.ins`` and ``.hkl`` bytes, used to recognise the demo exactly."""
    ins_text, hkl_text = demo_case_texts()
    return (
        hashlib.sha256(ins_text.encode("utf-8")).hexdigest(),
        hashlib.sha256(hkl_text.encode("utf-8")).hexdigest(),
    )


def write_demo_case(directory: Path) -> tuple[Path, Path]:
    """Write ``crystalx_demo.ins`` / ``crystalx_demo.hkl`` into ``directory`` (created if needed).

    Existing files are left alone if they are already the demo, and refused otherwise.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ins_text, hkl_text = demo_case_texts()
    paths = (directory / f"{DEMO_STEM}.ins", directory / f"{DEMO_STEM}.hkl")
    for path, text in zip(paths, (ins_text, hkl_text)):
        data = text.encode("utf-8")
        if path.exists():
            if path.read_bytes() != data:
                raise FileExistsError(f"{path} exists and is not the demo file; refusing to overwrite")
            continue
        path.write_bytes(data)
    return paths


def is_demo_input(ins: Path, hkl: Path) -> bool:
    """True only when both files are byte-identical to the generated demo."""
    try:
        ins_digest = hashlib.sha256(Path(ins).read_bytes()).hexdigest()
        hkl_digest = hashlib.sha256(Path(hkl).read_bytes()).hexdigest()
    except OSError:
        return False
    return (ins_digest, hkl_digest) == demo_digests()
