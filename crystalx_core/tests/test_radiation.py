"""Radiation classification and the dataset-merge guard.

The rule this enforces: **two datasets from different X-ray sources must not be merged.** The
wavelength changes the dispersion corrections, the absorption and the accessible resolution, so a
merged Cu/Mo dataset looks fine and means nothing. Someone dropping a molybdenum dataset in among
copper ones is caught here and the merge stops.
"""

from __future__ import annotations

import pytest

from crystalx_core.domain.radiation import (
    CHARACTERISTIC_WAVELENGTHS_A,
    DEFAULT_RADIATION,
    DatasetFingerprint,
    MergeBlockerKind,
    RadiationAssumption,
    RadiationSource,
    assess_merge_compatibility,
    classify_radiation,
)

CU = CHARACTERISTIC_WAVELENGTHS_A[RadiationSource.CU_KA]
MO = CHARACTERISTIC_WAVELENGTHS_A[RadiationSource.MO_KA]
CELL = (24.1234, 18.5678, 31.9012, 90.0, 101.5, 90.0)


def fingerprint(label: str, **overrides) -> DatasetFingerprint:
    base = dict(
        label=label,
        sha256=f"sha-{label}",
        radiation=RadiationSource.CU_KA,
        wavelength_a=CU,
        cell=CELL,
        hklf=4,
        latt=1,
        n_symm=1,
        has_unmerged_friedel_pairs=True,
        d_min_a=0.84,
    )
    base.update(overrides)
    return DatasetFingerprint(**base)


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wavelength", "expected"),
    [
        (1.54184, RadiationSource.CU_KA),
        (1.54178, RadiationSource.CU_KA),
        (1.54056, RadiationSource.CU_KA),  # Cu K-alpha1 alone
        (0.71073, RadiationSource.MO_KA),
        (0.70930, RadiationSource.MO_KA),  # Mo K-alpha1 alone
        (0.56087, RadiationSource.AG_KA),
        (2.29100, RadiationSource.CR_KA),
        (1.34139, RadiationSource.GA_KA),
    ],
)
def test_recognises_the_common_laboratory_sources(wavelength, expected):
    assert classify_radiation(wavelength) is expected


def test_a_synchrotron_wavelength_is_other_not_a_wrong_guess():
    assert classify_radiation(0.9795) is RadiationSource.OTHER


def test_no_wavelength_is_unknown():
    assert classify_radiation(None) is RadiationSource.UNKNOWN
    assert classify_radiation(0.0) is RadiationSource.UNKNOWN
    assert classify_radiation(-1.0) is RadiationSource.UNKNOWN


def test_the_windows_do_not_overlap():
    """Every characteristic line must classify to itself and to nothing else."""
    for source, wavelength in CHARACTERISTIC_WAVELENGTHS_A.items():
        assert classify_radiation(wavelength) is source


def test_default_assumption_is_copper_and_is_flagged_as_assumed():
    """The owner's working assumption: Cu K-alpha unless the file says otherwise."""
    assumption = RadiationAssumption.from_wavelength(None)
    assert assumption.source is DEFAULT_RADIATION is RadiationSource.CU_KA
    assert assumption.is_assumed
    assert "assuming the project default" in assumption.note


def test_a_wavelength_in_the_file_overrides_the_default():
    assumption = RadiationAssumption.from_wavelength(MO)
    assert assumption.source is RadiationSource.MO_KA
    assert assumption.from_file
    assert not assumption.is_assumed


def test_an_unusual_wavelength_is_recorded_with_an_explanation():
    assumption = RadiationAssumption.from_wavelength(0.9795)
    assert assumption.source is RadiationSource.OTHER
    assert "synchrotron" in assumption.note


# --------------------------------------------------------------------------------------
# merge guard
# --------------------------------------------------------------------------------------


def test_a_single_dataset_has_nothing_to_merge():
    report = assess_merge_compatibility([fingerprint("only")])
    assert report.may_merge
    assert "nothing to merge" in report.notes[0]


def test_two_matching_copper_datasets_may_merge():
    report = assess_merge_compatibility([fingerprint("run1"), fingerprint("run2")])
    assert report.may_merge
    assert report.blockers == ()
    assert "compatible" in report.summary()


def test_a_molybdenum_dataset_among_copper_ones_stops_the_merge():
    """The headline case, exactly as the owner described it."""
    report = assess_merge_compatibility(
        [
            fingerprint("cu_run1"),
            fingerprint("cu_run2"),
            fingerprint("mo_intruder", radiation=RadiationSource.MO_KA, wavelength_a=MO),
        ]
    )
    assert not report.may_merge
    blocker = report.blockers[0]
    assert blocker.kind is MergeBlockerKind.RADIATION_MISMATCH
    assert blocker.dataset_b == "mo_intruder"
    assert "different X-ray sources cannot be merged" in blocker.message
    assert "merge refused" in report.summary()


def test_same_source_but_wavelengths_too_far_apart_is_blocked():
    report = assess_merge_compatibility(
        [fingerprint("a"), fingerprint("b", wavelength_a=CU + 0.02)]
    )
    assert not report.may_merge
    assert report.blockers[0].kind is MergeBlockerKind.WAVELENGTH_MISMATCH


def test_a_tiny_wavelength_rounding_difference_is_tolerated():
    """1.54178 and 1.54184 are the same source written two conventional ways."""
    report = assess_merge_compatibility(
        [fingerprint("a", wavelength_a=1.54178), fingerprint("b", wavelength_a=1.54184)]
    )
    assert report.may_merge


def test_a_different_unit_cell_is_blocked_and_names_the_axis():
    other = list(CELL)
    other[1] = CELL[1] * 1.10
    report = assess_merge_compatibility([fingerprint("a"), fingerprint("b", cell=tuple(other))])
    assert not report.may_merge
    blocker = report.blockers[0]
    assert blocker.kind is MergeBlockerKind.CELL_MISMATCH
    assert "b 18.5678" in blocker.message


def test_a_small_cell_difference_within_tolerance_is_accepted():
    other = list(CELL)
    other[0] = CELL[0] * 1.002
    report = assess_merge_compatibility([fingerprint("a"), fingerprint("b", cell=tuple(other))])
    assert report.may_merge


def test_an_angle_difference_beyond_tolerance_is_blocked():
    other = list(CELL)
    other[4] = CELL[4] + 3.0
    report = assess_merge_compatibility([fingerprint("a"), fingerprint("b", cell=tuple(other))])
    assert not report.may_merge
    assert "beta" in report.blockers[0].message


def test_mixed_hklf_formats_are_blocked():
    report = assess_merge_compatibility([fingerprint("a"), fingerprint("b", hklf=5)])
    assert not report.may_merge
    assert report.blockers[0].kind is MergeBlockerKind.HKLF_MISMATCH


def test_different_lattice_settings_are_blocked():
    report = assess_merge_compatibility([fingerprint("a"), fingerprint("b", latt=-1)])
    assert not report.may_merge
    assert report.blockers[0].kind is MergeBlockerKind.SPACE_GROUP_MISMATCH


def test_mismatched_anomalous_handling_warns_but_does_not_block():
    """Advisory: merging would discard one dataset's anomalous signal, which the user may accept."""
    report = assess_merge_compatibility(
        [fingerprint("a"), fingerprint("b", has_unmerged_friedel_pairs=False)]
    )
    assert report.may_merge
    assert report.warnings
    assert report.warnings[0].kind is MergeBlockerKind.ANOMALOUS_MISMATCH


def test_very_different_resolutions_warn_but_do_not_block():
    report = assess_merge_compatibility([fingerprint("a", d_min_a=0.84), fingerprint("b", d_min_a=1.6)])
    assert report.may_merge
    assert any(w.kind is MergeBlockerKind.RESOLUTION_MISMATCH for w in report.warnings)


def test_an_unknown_radiation_source_is_called_out_in_the_notes():
    report = assess_merge_compatibility(
        [
            fingerprint("a"),
            fingerprint("b", radiation=RadiationSource.UNKNOWN, wavelength_a=None),
        ]
    )
    assert any("no wavelength" in note for note in report.notes)


def test_several_blockers_are_all_reported_not_just_the_first():
    report = assess_merge_compatibility(
        [
            fingerprint("a"),
            fingerprint("b", radiation=RadiationSource.MO_KA, wavelength_a=MO, hklf=5, latt=-1),
        ]
    )
    kinds = {b.kind for b in report.blockers}
    assert kinds == {
        MergeBlockerKind.RADIATION_MISMATCH,
        MergeBlockerKind.HKLF_MISMATCH,
        MergeBlockerKind.SPACE_GROUP_MISMATCH,
    }
