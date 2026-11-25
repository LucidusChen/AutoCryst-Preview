"""Radiation source, and the guard that refuses to merge incompatible datasets.

Two datasets recorded on different X-ray sources cannot be merged: the structure factors are on
different wavelengths, so the dispersion corrections, the absorption, and the accessible resolution
all differ. Merging them silently produces a dataset that looks fine and is physically meaningless.
The guard here treats that as a hard stop rather than a warning.

The project's working assumption, set by the product owner: **Cu K-alpha unless the file says
otherwise**. Large-unit-cell coordination systems are routinely measured on a copper source, a
molybdenum source scatters weakly from C/N/O, and an inorganic dataset good enough to need merging
would not have been collected on Mo either. The assumption is recorded per run so it is visible, and
it is overridden by the wavelength actually present in the ``CELL`` record.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

__all__ = [
    "RadiationSource",
    "CHARACTERISTIC_WAVELENGTHS_A",
    "DEFAULT_RADIATION",
    "WAVELENGTH_TOLERANCE_A",
    "classify_radiation",
    "RadiationAssumption",
    "MergeBlockerKind",
    "MergeBlocker",
    "MergeCompatibilityReport",
    "assess_merge_compatibility",
    "DatasetFingerprint",
]


class RadiationSource(str, enum.Enum):
    """Laboratory X-ray sources this project recognises by wavelength."""

    CU_KA = "Cu_Ka"
    MO_KA = "Mo_Ka"
    AG_KA = "Ag_Ka"
    CR_KA = "Cr_Ka"
    GA_KA = "Ga_Ka"
    OTHER = "other"
    """Synchrotron or an unrecognised wavelength. Not an error, just not a lab tube."""

    UNKNOWN = "unknown"
    """No wavelength was available at all."""


#: Weighted K-alpha wavelengths in angstrom, as they appear in a SHELX ``CELL`` record.
CHARACTERISTIC_WAVELENGTHS_A: dict[RadiationSource, float] = {
    RadiationSource.CR_KA: 2.29100,
    RadiationSource.CU_KA: 1.54184,
    RadiationSource.GA_KA: 1.34139,
    RadiationSource.MO_KA: 0.71073,
    RadiationSource.AG_KA: 0.56087,
}

#: Half-width of the window used to recognise a source. Comfortably covers the K-alpha1/K-alpha2
#: spread and the several conventional roundings of each value (Cu is variously written 1.54056,
#: 1.54178, 1.54180, 1.54184) without letting two different sources overlap.
WAVELENGTH_TOLERANCE_A = 0.005

#: Project default when a file carries no wavelength. Recorded, never silently assumed.
DEFAULT_RADIATION = RadiationSource.CU_KA


def classify_radiation(wavelength_a: float | None) -> RadiationSource:
    """Identify the source from a wavelength, or report ``OTHER`` / ``UNKNOWN``."""
    if wavelength_a is None or wavelength_a <= 0:
        return RadiationSource.UNKNOWN
    for source, reference in CHARACTERISTIC_WAVELENGTHS_A.items():
        if abs(wavelength_a - reference) <= WAVELENGTH_TOLERANCE_A:
            return source
    return RadiationSource.OTHER


@dataclass(frozen=True)
class RadiationAssumption:
    """What the run believes about the radiation, and whether it was measured or assumed."""

    source: RadiationSource
    wavelength_a: float | None
    from_file: bool
    """True when the wavelength came from the data's own ``CELL`` record."""

    note: str = ""

    @classmethod
    def from_wavelength(cls, wavelength_a: float | None) -> "RadiationAssumption":
        if wavelength_a is None or wavelength_a <= 0:
            return cls(
                source=DEFAULT_RADIATION,
                wavelength_a=CHARACTERISTIC_WAVELENGTHS_A[DEFAULT_RADIATION],
                from_file=False,
                note=(
                    "No usable wavelength in the instruction file; assuming the project default "
                    f"({DEFAULT_RADIATION.value}). This assumption is recorded, not hidden."
                ),
            )
        source = classify_radiation(wavelength_a)
        note = ""
        if source is RadiationSource.OTHER:
            note = (
                f"Wavelength {wavelength_a:.5f} A matches no common laboratory K-alpha line; "
                "treating it as a synchrotron or bespoke source."
            )
        return cls(source=source, wavelength_a=wavelength_a, from_file=True, note=note)

    @property
    def is_assumed(self) -> bool:
        return not self.from_file


class MergeBlockerKind(str, enum.Enum):
    """Reasons two datasets must not be combined."""

    RADIATION_MISMATCH = "radiation_mismatch"
    """Different X-ray sources. The hard case this guard exists for."""

    WAVELENGTH_MISMATCH = "wavelength_mismatch"
    """Same nominal source but wavelengths outside tolerance of each other."""

    CELL_MISMATCH = "cell_mismatch"
    HKLF_MISMATCH = "hklf_mismatch"
    SPACE_GROUP_MISMATCH = "space_group_mismatch"
    ANOMALOUS_MISMATCH = "anomalous_mismatch"
    """One dataset keeps Friedel pairs, the other has merged them. Advisory, not blocking."""

    RESOLUTION_MISMATCH = "resolution_mismatch"
    """Advisory: very different resolution limits change what the merge means."""

    @property
    def is_blocking(self) -> bool:
        return self in (
            MergeBlockerKind.RADIATION_MISMATCH,
            MergeBlockerKind.WAVELENGTH_MISMATCH,
            MergeBlockerKind.CELL_MISMATCH,
            MergeBlockerKind.HKLF_MISMATCH,
            MergeBlockerKind.SPACE_GROUP_MISMATCH,
        )


@dataclass(frozen=True)
class MergeBlocker:
    kind: MergeBlockerKind
    message: str
    dataset_a: str
    dataset_b: str
    value_a: str = ""
    value_b: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.kind.is_blocking


@dataclass(frozen=True)
class DatasetFingerprint:
    """The few properties that decide whether two measurements describe the same experiment."""

    label: str
    sha256: str
    radiation: RadiationSource
    wavelength_a: float | None
    cell: tuple[float, float, float, float, float, float] | None
    hklf: int | None
    latt: int | None
    n_symm: int
    has_unmerged_friedel_pairs: bool
    d_min_a: float | None = None


@dataclass(frozen=True)
class MergeCompatibilityReport:
    """Whether a set of datasets may be merged, and exactly why not when they may not."""

    fingerprints: tuple[DatasetFingerprint, ...]
    issues: tuple[MergeBlocker, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blockers(self) -> tuple[MergeBlocker, ...]:
        return tuple(i for i in self.issues if i.is_blocking)

    @property
    def warnings(self) -> tuple[MergeBlocker, ...]:
        return tuple(i for i in self.issues if not i.is_blocking)

    @property
    def may_merge(self) -> bool:
        return not self.blockers

    def summary(self) -> str:
        if self.may_merge:
            extra = f" ({len(self.warnings)} advisory note(s))" if self.warnings else ""
            return f"{len(self.fingerprints)} datasets are compatible{extra}"
        return "merge refused: " + "; ".join(b.message for b in self.blockers)


def assess_merge_compatibility(
    fingerprints: tuple[DatasetFingerprint, ...] | list[DatasetFingerprint],
    *,
    cell_tolerance_fraction: float = 0.01,
    angle_tolerance_deg: float = 0.5,
    resolution_warn_ratio: float = 1.25,
) -> MergeCompatibilityReport:
    """Decide whether several measurements may be combined.

    Each dataset is compared against the first. A mixed-source collection -- someone has dropped a
    molybdenum dataset in among copper ones -- is caught by the radiation check and reported as a
    blocker, which is what stops the merge.
    """
    entries = tuple(fingerprints)
    if len(entries) < 2:
        return MergeCompatibilityReport(
            fingerprints=entries,
            notes=("Fewer than two datasets: nothing to merge.",),
        )

    reference = entries[0]
    issues: list[MergeBlocker] = []

    for other in entries[1:]:
        if reference.radiation is not other.radiation:
            issues.append(
                MergeBlocker(
                    kind=MergeBlockerKind.RADIATION_MISMATCH,
                    message=(
                        f"{reference.label} was measured with {reference.radiation.value} and "
                        f"{other.label} with {other.radiation.value}. Data from different X-ray "
                        "sources cannot be merged: the wavelength changes the dispersion "
                        "corrections, the absorption and the accessible resolution."
                    ),
                    dataset_a=reference.label,
                    dataset_b=other.label,
                    value_a=reference.radiation.value,
                    value_b=other.radiation.value,
                )
            )
        elif (
            reference.wavelength_a is not None
            and other.wavelength_a is not None
            and abs(reference.wavelength_a - other.wavelength_a) > WAVELENGTH_TOLERANCE_A
        ):
            issues.append(
                MergeBlocker(
                    kind=MergeBlockerKind.WAVELENGTH_MISMATCH,
                    message=(
                        f"{reference.label} is at {reference.wavelength_a:.5f} A and {other.label} "
                        f"at {other.wavelength_a:.5f} A, further apart than the "
                        f"{WAVELENGTH_TOLERANCE_A} A tolerance."
                    ),
                    dataset_a=reference.label,
                    dataset_b=other.label,
                    value_a=f"{reference.wavelength_a:.5f}",
                    value_b=f"{other.wavelength_a:.5f}",
                )
            )

        cell_issue = _compare_cells(
            reference, other, cell_tolerance_fraction, angle_tolerance_deg
        )
        if cell_issue is not None:
            issues.append(cell_issue)

        if reference.hklf is not None and other.hklf is not None and reference.hklf != other.hklf:
            issues.append(
                MergeBlocker(
                    kind=MergeBlockerKind.HKLF_MISMATCH,
                    message=(
                        f"{reference.label} is HKLF {reference.hklf} and {other.label} is "
                        f"HKLF {other.hklf}; the reflection files are in different formats."
                    ),
                    dataset_a=reference.label,
                    dataset_b=other.label,
                    value_a=str(reference.hklf),
                    value_b=str(other.hklf),
                )
            )

        if (
            reference.latt is not None
            and other.latt is not None
            and reference.latt != other.latt
        ):
            issues.append(
                MergeBlocker(
                    kind=MergeBlockerKind.SPACE_GROUP_MISMATCH,
                    message=(
                        f"{reference.label} declares LATT {reference.latt} and {other.label} "
                        f"LATT {other.latt}; they are not in the same setting."
                    ),
                    dataset_a=reference.label,
                    dataset_b=other.label,
                    value_a=str(reference.latt),
                    value_b=str(other.latt),
                )
            )

        if reference.has_unmerged_friedel_pairs != other.has_unmerged_friedel_pairs:
            issues.append(
                MergeBlocker(
                    kind=MergeBlockerKind.ANOMALOUS_MISMATCH,
                    message=(
                        f"{reference.label} and {other.label} disagree on whether Friedel mates "
                        "are kept separate; merging would discard the anomalous signal of "
                        "whichever one has it."
                    ),
                    dataset_a=reference.label,
                    dataset_b=other.label,
                    value_a=str(reference.has_unmerged_friedel_pairs),
                    value_b=str(other.has_unmerged_friedel_pairs),
                )
            )

        if (
            reference.d_min_a is not None
            and other.d_min_a is not None
            and max(reference.d_min_a, other.d_min_a)
            > resolution_warn_ratio * min(reference.d_min_a, other.d_min_a)
        ):
            issues.append(
                MergeBlocker(
                    kind=MergeBlockerKind.RESOLUTION_MISMATCH,
                    message=(
                        f"{reference.label} reaches {reference.d_min_a:.2f} A and {other.label} "
                        f"{other.d_min_a:.2f} A; the merged high-resolution shell will be "
                        "dominated by one dataset."
                    ),
                    dataset_a=reference.label,
                    dataset_b=other.label,
                    value_a=f"{reference.d_min_a:.2f}",
                    value_b=f"{other.d_min_a:.2f}",
                )
            )

    notes: list[str] = []
    if any(f.radiation is RadiationSource.UNKNOWN for f in entries):
        notes.append(
            "At least one dataset carries no wavelength; its source could not be confirmed and "
            "the comparison against it is weaker than it looks."
        )

    return MergeCompatibilityReport(
        fingerprints=entries, issues=tuple(issues), notes=tuple(notes)
    )


def _compare_cells(
    a: DatasetFingerprint,
    b: DatasetFingerprint,
    edge_tolerance_fraction: float,
    angle_tolerance_deg: float,
) -> MergeBlocker | None:
    if a.cell is None or b.cell is None:
        return None
    differences: list[str] = []
    names = ("a", "b", "c", "alpha", "beta", "gamma")
    for index, name in enumerate(names):
        first, second = a.cell[index], b.cell[index]
        if index < 3:
            allowed = edge_tolerance_fraction * max(abs(first), abs(second))
        else:
            allowed = angle_tolerance_deg
        if abs(first - second) > allowed:
            differences.append(f"{name} {first:.4f} vs {second:.4f}")
    if not differences:
        return None
    return MergeBlocker(
        kind=MergeBlockerKind.CELL_MISMATCH,
        message=(
            f"{a.label} and {b.label} do not share a unit cell within tolerance: "
            + ", ".join(differences)
        ),
        dataset_a=a.label,
        dataset_b=b.label,
        value_a=str(a.cell),
        value_b=str(b.cell),
    )
