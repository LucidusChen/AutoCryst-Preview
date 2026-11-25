"""Chemical context supplied by the crystallographer at Stage 0.

Everything here is an **input to the search**, never evidence for a result. A prior may order the
search, seed restraint targets, and supply fragment geometry; the acceptance evidence remains SHELXL
refinement and recorded validation output. Whether a prior influenced a candidate is always recorded
in that candidate's provenance so no result can later be mistaken for unbiased.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ATOMIC_NUMBERS",
    "atomic_number",
    "element_symbol",
    "DonorGroupKind",
    "CounterAnionKind",
    "LockRelevance",
    "DonorAtomSpec",
    "DonorGroupSpec",
    "MetalPrior",
    "CounterAnionSpec",
    "ReferenceStructure",
    "ChargeModel",
    "ChemicalPrior",
]


#: Symbol to atomic number, ordered by Z. Used only for unambiguous integer facts such as
#: "is this element heavier than Z=18". Anything wavelength-dependent (dispersion corrections) is
#: deliberately **not** here; see ``crystalx_core.domain.data.AnomalousScatteringTable``.
ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98,
}

_SYMBOL_BY_Z: dict[int, str] = {z: s for s, z in ATOMIC_NUMBERS.items()}


def atomic_number(symbol: str) -> int:
    """Atomic number for an element symbol, case-insensitively.

    Raises ``KeyError`` for an unknown symbol rather than guessing, because a silently wrong element
    is exactly the failure mode this project exists to prevent.
    """
    key = symbol.strip().capitalize()
    if key not in ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element symbol: {symbol!r}")
    return ATOMIC_NUMBERS[key]


def element_symbol(z: int) -> str:
    """Element symbol for an atomic number."""
    if z not in _SYMBOL_BY_Z:
        raise KeyError(f"Unknown atomic number: {z!r}")
    return _SYMBOL_BY_Z[z]


class DonorGroupKind(str, enum.Enum):
    """Coordinating group the user expects to be present.

    Used to bias first-shell interpretation: knowing "there are imidazoles" tells the workflow which
    donor element to expect at a given M-X distance and what fragment should hang off it.
    """

    PYRIDINE = "pyridine"
    IMIDAZOLE = "imidazole"
    AMIDE = "amide"
    AMINE = "amine"
    CARBOXYLATE = "carboxylate"
    THIOETHER = "thioether"
    THIOLATE = "thiolate"
    PHOSPHINE = "phosphine"
    NITRILE = "nitrile"
    CARBENE_NHC = "carbene_nhc"
    ALKENE = "alkene"
    AQUA = "aqua"
    HYDROXO = "hydroxo"
    HALIDE = "halide"
    OTHER = "other"


class CounterAnionKind(str, enum.Enum):
    """Counter-anions that are common and disorder-prone.

    These are called out explicitly because they are the usual cause of a structure that refines
    "almost" correctly: the cationic framework is right and the anion is smeared.
    """

    TRIFLIMIDE_NTF2 = "NTf2"
    TRIFLATE_OTF = "OTf"
    TRIFLUOROACETATE = "CF3COO"
    TETRAFLUOROBORATE = "BF4"
    HEXAFLUOROPHOSPHATE = "PF6"
    PERCHLORATE = "ClO4"
    NITRATE = "NO3"
    TETRAPHENYLBORATE = "BPh4"
    BARF = "BArF"
    HALIDE = "halide"
    OTHER = "other"


class LockRelevance(str, enum.Enum):
    """How central an element is to the peeling-onion anchor."""

    ANCHOR = "anchor"
    """A heavy/metal node expected to be locked or restrained first."""

    FIRST_SHELL = "first_shell"
    """A donor atom expected to be resolved immediately after the anchor."""

    SKELETON = "skeleton"
    """Ordinary framework atom, resolved during shell expansion."""


@dataclass(frozen=True)
class DonorAtomSpec:
    """Expected donor atom and its distance window to the metal.

    ``distance_min_a``/``distance_max_a`` are in angstrom and are used both to *propose* a donor
    assignment and to *reject* an implausible one during gate evaluation.
    """

    element: str
    distance_min_a: float
    distance_max_a: float
    note: str = ""

    def __post_init__(self) -> None:
        atomic_number(self.element)  # validate the symbol early
        if not 0.0 < self.distance_min_a < self.distance_max_a:
            raise ValueError(
                f"Invalid donor distance window for {self.element}: "
                f"{self.distance_min_a} .. {self.distance_max_a}"
            )


@dataclass(frozen=True)
class DonorGroupSpec:
    """A coordinating group the user declared, with the donor atoms it can present."""

    kind: DonorGroupKind
    donors: tuple[DonorAtomSpec, ...]
    label: str = ""
    expected_count: int | None = None


@dataclass(frozen=True)
class MetalPrior:
    """A metal the user expects, with its chemically plausible envelope."""

    element: str
    oxidation_state: int | None = None
    coordination_number_min: int | None = None
    coordination_number_max: int | None = None
    expected_sites: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        atomic_number(self.element)
        lo, hi = self.coordination_number_min, self.coordination_number_max
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"coordination_number_min ({lo}) exceeds coordination_number_max ({hi}) "
                f"for {self.element}"
            )

    @property
    def z(self) -> int:
        return atomic_number(self.element)


@dataclass(frozen=True)
class CounterAnionSpec:
    """A counter-anion the user expects, with its formal charge and multiplicity."""

    kind: CounterAnionKind
    charge: int = -1
    expected_count: int | None = None
    element_hint: str = ""
    """For ``HALIDE``/``OTHER``: the element to look for, e.g. ``"Cl"``."""

    disorder_prone: bool = True
    """Most of these anions are. Kept explicit so the workflow can pre-arm a disorder strategy."""


@dataclass(frozen=True)
class ReferenceStructure:
    """A topologically similar published/previous structure, supplied as advisory context.

    This is deliberately a separate type from anything in ``metrics`` or ``gates``: a reference may
    steer the search and supply restraint targets, but it can never be cited as evidence that a
    candidate is correct. ``influence`` records what it was actually allowed to do.
    """

    path: Path
    label: str = ""
    provenance_note: str = ""
    may_seed_restraints: bool = True
    may_order_search: bool = True
    may_supply_fragment_geometry: bool = True

    #: Always false. Present so the intent is unmissable in the record and in code review.
    counts_as_evidence: bool = False

    derived_from_same_measurement: bool = False
    """Whether this structure was solved **from the reflection file now being solved**.

    If it was, it is not a reference -- it is the answer. Feeding it in would let the pipeline
    reproduce a result it was supposed to derive, and every figure of merit downstream would be
    measuring nothing. The distinction is invisible from the file itself: a ``.res`` solved from
    ``case.hkl`` and a ``.res`` from a different crystal of the same compound look identical.
    Only the person who collected the data knows, so it has to be declared.

    Constructing a :class:`ReferenceStructure` with this set raises. Use it as ground truth for
    *grading* a finished run instead, outside the pipeline.
    """

    def __post_init__(self) -> None:
        if self.derived_from_same_measurement:
            raise ValueError(
                f"{self.path.name!r} was solved from the same measurement that is being solved "
                "now, so it is an answer key, not a reference structure. Supplying it would leak "
                "the answer into the search and make every downstream figure of merit meaningless. "
                "Keep it outside the run and use it to grade the result afterwards."
            )


@dataclass(frozen=True)
class ChargeModel:
    """Declared charge bookkeeping for the formula unit.

    Charge balance is cheap to check and is one of the most useful beginner-facing guards: two
    silver cations with one triflate means something is missing from the model.
    """

    metal_charge_total: int = 0
    ligand_charge_total: int = 0
    anion_charge_total: int = 0
    other_charge_total: int = 0
    declared_by_user: bool = False

    @property
    def net_charge(self) -> int:
        return (
            self.metal_charge_total
            + self.ligand_charge_total
            + self.anion_charge_total
            + self.other_charge_total
        )

    @property
    def is_balanced(self) -> bool:
        return self.net_charge == 0


@dataclass(frozen=True)
class ChemicalPrior:
    """Everything the crystallographer told us before solving started."""

    metals: tuple[MetalPrior, ...] = ()
    donor_groups: tuple[DonorGroupSpec, ...] = ()
    counter_anions: tuple[CounterAnionSpec, ...] = ()
    expected_solvents: tuple[str, ...] = ()
    reference_structures: tuple[ReferenceStructure, ...] = ()
    charge_model: ChargeModel | None = None
    ligand_is_neutral: bool | None = None

    is_peptide_like: bool = False
    """Gates SHELXE main-chain autotracing.

    SHELXE's autotracing assumes polypeptide geometry. On a metal-organic small molecule it produces
    confident nonsense, so it is off unless the user declares a genuine peptide. Density modification
    and hand determination are useful either way and are not gated by this flag.
    """

    enantiopure: bool | None = None
    """Whether every molecule in the crystal has the same handedness.

    ``None`` means *infer*: a peptide-like system is enantiopure unless stated otherwise, because
    ribosomal and synthetic peptides are built from L-amino acids. Set ``False`` explicitly for a
    racemic or meso preparation -- that is the escape hatch, and it has to be deliberate, because
    the consequence of getting it wrong in the permissive direction is accepting a space group the
    material cannot occupy.

    See :meth:`is_enantiopure` for the resolved answer and
    :mod:`crystalx_core.domain.symmetry` for what it is used to rule out.
    """

    free_text_notes: str = ""
    declared_elements: tuple[str, ...] = ()
    """Optional user-declared composition, kept separate from the ``.ins`` SFAC list so that an
    override of the file's own composition is always visible as an override."""

    extras: dict[str, str] = field(default_factory=dict)

    @property
    def metal_elements(self) -> tuple[str, ...]:
        return tuple(m.element for m in self.metals)

    def is_enantiopure(self) -> tuple[bool, str]:
        """Resolve :attr:`enantiopure`, returning the verdict and the reason for it.

        The reason is returned alongside the boolean because this decision can *block a solve*, and
        a block whose justification is not written down is one the crystallographer cannot argue
        with.
        """
        if self.enantiopure is True:
            return True, "the crystallographer declared the material enantiopure"
        if self.enantiopure is False:
            return False, "the crystallographer declared the material racemic or meso"
        if self.is_peptide_like:
            return True, (
                "the system was declared peptide-like, and peptides are built from L-amino acids, "
                "so the assembly is homochiral unless you say otherwise "
                "(set enantiopure=False for a racemic preparation)"
            )
        return False, "no enantiopure component was declared, so handedness is not assumed"

    def heaviest_declared_z(self) -> int | None:
        """Highest atomic number among declared metals and declared elements, or ``None``."""
        zs: list[int] = [m.z for m in self.metals]
        for symbol in self.declared_elements:
            try:
                zs.append(atomic_number(symbol))
            except KeyError:
                continue
        return max(zs) if zs else None
