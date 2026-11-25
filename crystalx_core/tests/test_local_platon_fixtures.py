"""Opt-in validation against the user's own PLATON test data.

PLATON's own README states that the program "may be used free of charge by the academic community
under the condition that it is not redistributed". Its bundled test data is part of that
distribution, so **these files are deliberately not copied into this repository**. Instead the tests
read them in place from a locally installed copy and skip when it is absent.

Run them with::

    CRYSTALX_PLATON_TEST_DIR=/path/to/platon_instruction/TEST pytest -q

Files exercised (all real SHELX instruction files written by third parties, not by this project):

* ``ADDSYM_EXAMPLE/ex2.ins``  -- a monoclinic Cc structure with ``LATT -7``, one ``SYMM`` operator,
  a full atom list between ``UNIT`` and ``HKLF``, and trailing instructions.
* ``ABSORB_EXAMPLE/s588a.ins`` -- a minimal absorption-correction input with ``CELL`` and ``HKLF``
  but no ``LATT``/``SYMM``/``SFAC``, plus ``FACE``/``ABSG`` instructions the reader must ignore.
* ``ABSTOMPA_TEST/abstest.ins`` -- a **Cu K-alpha** file, which is what most of this project's data
  will be; the radiation classifier has to recognise 1.5418 as Cu.
* ``ABSTOMPA_EXAMPLE/s555b.ins`` -- Mo K-alpha, and therefore the other half of the merge guard.
* ``ASYM_EXAMPLE/asym.ins`` -- PLATON's own ``SPGR`` extension and **no** ``HKLF`` record.
* ``PSICOR_EXAMPLE/s588a.hkl`` -- 3566 real reflections in the twelve-field direction-cosine
  layout. This file is the reason ``_infer_hklf`` exists: the reader used to call it HKLF 5.

Two defects were found by these files and fixed; the regressions live in ``test_shelx_reader.py``
with synthetic copies of the same shapes, so CI covers them without needing this data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crystalx_core.domain.data import AnomalousAvailability, HklfFormat, SpaceGroupConfidence
from crystalx_core.domain.radiation import (
    DatasetFingerprint,
    MergeBlockerKind,
    RadiationAssumption,
    RadiationSource,
    assess_merge_compatibility,
)
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl

PLATON_TEST_DIR = os.environ.get("CRYSTALX_PLATON_TEST_DIR", "")

needs_platon_data = pytest.mark.skipif(
    not PLATON_TEST_DIR or not Path(PLATON_TEST_DIR).is_dir(),
    reason="set CRYSTALX_PLATON_TEST_DIR to a local PLATON TEST directory to run these",
)


def _fixture(relative: str) -> Path:
    path = Path(PLATON_TEST_DIR) / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not present in this PLATON installation")
    return path


@needs_platon_data
def test_reads_a_real_monoclinic_ins_with_an_atom_list():
    header = read_ins_header(_fixture("ADDSYM_EXAMPLE/ex2.ins"))

    assert header.title.startswith("CcToFdd2")
    assert header.wavelength_a == pytest.approx(0.71073)
    assert header.cell is not None
    assert header.cell.beta == pytest.approx(120.11)
    assert header.latt == -7
    assert header.is_centrosymmetric_latt is False
    assert len(header.symm) == 1
    assert header.sfac == ("C", "H", "S")
    assert header.unit == (56.0, 80.0, 16.0)
    assert header.non_hydrogen_unit_total == pytest.approx(72.0)
    assert header.hklf_format is HklfFormat.HKLF4
    assert header.space_group_confidence is SpaceGroupConfidence.FROM_INS
    # The atom rows between UNIT and HKLF must not be mistaken for instructions.
    assert header.warnings == ()


@needs_platon_data
def test_reads_a_minimal_ins_without_symmetry_or_composition():
    header = read_ins_header(_fixture("ABSORB_EXAMPLE/s588a.ins"))

    assert header.title == "S588A - GAUSSIAN"
    assert header.cell is not None
    assert header.cell.a == pytest.approx(11.1364)
    # No LATT/SYMM: reported as unknown rather than assumed.
    assert header.latt is None
    assert header.symm == ()
    assert header.space_group_confidence is SpaceGroupConfidence.UNKNOWN
    # FACE and ABSG are not instructions this reader claims to understand, and are ignored cleanly.
    assert header.sfac == ()
    assert header.hklf_format is HklfFormat.HKLF4


@needs_platon_data
def test_an_ins_with_no_hklf_record_reports_none_rather_than_assuming_four():
    """``asym.ins`` has ``CELL`` and PLATON's ``SPGR`` and nothing else."""
    header = read_ins_header(_fixture("ASYM_EXAMPLE/asym.ins"))

    assert header.title == "s521b"
    assert header.hklf_format is None, "absence must be reported, not defaulted"
    assert header.cell is not None
    assert header.cell.beta == pytest.approx(115.70)
    # PLATON's SPGR extension is not a SHELX instruction and is not parsed today. The reader must
    # therefore still say it does not know the space group -- not silently ignore the file either.
    assert header.space_group_confidence is SpaceGroupConfidence.UNKNOWN


# --------------------------------------------------------------------------------------
# radiation, on real files
# --------------------------------------------------------------------------------------


@needs_platon_data
def test_a_real_copper_file_is_classified_as_copper():
    """1.5418 is how Cu K-alpha is usually written; the table value is 1.54184."""
    header = read_ins_header(_fixture("ABSTOMPA_TEST/abstest.ins"))
    assumption = RadiationAssumption.from_wavelength(header.wavelength_a)

    assert header.wavelength_a == pytest.approx(1.5418)
    assert assumption.source is RadiationSource.CU_KA
    assert not assumption.is_assumed, "the file states it; nothing is being assumed"


@needs_platon_data
def test_a_real_molybdenum_file_is_classified_as_molybdenum():
    header = read_ins_header(_fixture("ABSTOMPA_EXAMPLE/s555b.ins"))
    assumption = RadiationAssumption.from_wavelength(header.wavelength_a)

    assert "MOKA" in header.title
    assert assumption.source is RadiationSource.MO_KA
    assert not assumption.is_assumed


@needs_platon_data
def test_merging_a_real_copper_file_with_a_real_molybdenum_file_is_refused():
    """The merge guard, exercised on two genuine third-party datasets rather than fixtures."""
    cu = read_ins_header(_fixture("ABSTOMPA_TEST/abstest.ins"))
    mo = read_ins_header(_fixture("ABSTOMPA_EXAMPLE/s555b.ins"))

    def fingerprint(label, header):
        cell = header.cell
        return DatasetFingerprint(
            label=label,
            sha256="",
            radiation=RadiationAssumption.from_wavelength(header.wavelength_a).source,
            wavelength_a=header.wavelength_a,
            cell=(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma) if cell else None,
            hklf=header.hklf_format.value if header.hklf_format else None,
            latt=header.latt,
            n_symm=len(header.symm),
            has_unmerged_friedel_pairs=False,
        )

    report = assess_merge_compatibility(
        (fingerprint("abstest", cu), fingerprint("s555b", mo))
    )

    assert not report.may_merge
    kinds = {b.kind for b in report.blockers}
    assert MergeBlockerKind.RADIATION_MISMATCH in kinds


# --------------------------------------------------------------------------------------
# reflection files
# --------------------------------------------------------------------------------------


@needs_platon_data
def test_a_real_direction_cosine_file_is_not_reported_as_twinned():
    """Regression, on the file that exposed it.

    ``PSICOR_EXAMPLE/s588a.ins`` carries no ``HKLF``, so the dialect has to be inferred from the
    records. They are twelve fields wide -- ``h k l F^2 sigma batch`` plus six direction cosines --
    which is the psi-scan absorption layout, not an HKLF 5 twin.
    """
    header = read_ins_header(_fixture("PSICOR_EXAMPLE/s588a.ins"))
    assert header.hklf_format is None

    summary = scan_hkl(_fixture("PSICOR_EXAMPLE/s588a.hkl"), cell=header.cell)

    assert summary.n_reflections == 3566
    assert summary.hklf_format is HklfFormat.HKLF4
    assert not summary.hklf_was_declared
    assert any("direction cosines" in w for w in summary.warnings)
    assert summary.d_min_a is not None and 0.7 < summary.d_min_a < 0.8


@needs_platon_data
def test_two_stray_mates_in_a_real_merged_file_do_not_unlock_experimental_phasing():
    """The same file contains exactly two accidental Friedel coincidences in 3566 records.

    Under the original ``n_pairs > 0`` test this dataset would have been declared capable of
    experimental phasing, which would have put SHELXC/D/E in front of the user on data that
    cannot support it.
    """
    header = read_ins_header(_fixture("PSICOR_EXAMPLE/s588a.ins"))
    summary = scan_hkl(_fixture("PSICOR_EXAMPLE/s588a.hkl"), cell=header.cell)

    assert summary.n_friedel_pairs == 2
    assert summary.friedel_coverage < 0.01
    assert summary.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_SPARSE
    assert not summary.has_unmerged_friedel_pairs


@needs_platon_data
def test_a_declared_hklf_beats_the_inference():
    """``abstest`` has the same twelve-field records but declares ``HKLF 4`` in the instruction file."""
    header = read_ins_header(_fixture("ABSTOMPA_TEST/abstest.ins"))
    summary = scan_hkl(
        _fixture("ABSTOMPA_TEST/abstest.hkl"), cell=header.cell, hklf_hint=header.hklf_format
    )

    assert summary.hklf_format is HklfFormat.HKLF4
    assert summary.hklf_was_declared
    assert not any("Declare HKLF" in w for w in summary.warnings)
