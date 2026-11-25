"""SHELX diagnostic readers.

The regression that matters most: instruction recognition is **token-anchored**. The imported engine
starts reading atoms at ``if "PLAN" in line``, which a title like ``TITL ligand PLANar isomer``
breaks. Several tests below exist purely to keep that class of bug out.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from conftest import CUBIC_A, write_hkl, write_ins
from crystalx_core.domain.data import AnomalousAvailability, HklfFormat, SpaceGroupConfidence, UnitCell
from crystalx_core.errors import ParseError
from crystalx_core.io.shelx_reader import logical_lines, read_ins_header, scan_hkl


# --------------------------------------------------------------------------------------
# d-spacing: checked against an analytically known case
# --------------------------------------------------------------------------------------


def test_cubic_d_spacing_matches_closed_form():
    """For a cubic cell, d = a / sqrt(h^2 + k^2 + l^2). Verifiable by hand."""
    cell = UnitCell(CUBIC_A, CUBIC_A, CUBIC_A, 90.0, 90.0, 90.0)
    for h, k, l in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (3, 2, 1)]:
        expected = CUBIC_A / math.sqrt(h * h + k * k + l * l)
        assert cell.d_spacing(h, k, l) == pytest.approx(expected, rel=1e-12)


def test_cubic_cell_volume():
    cell = UnitCell(CUBIC_A, CUBIC_A, CUBIC_A, 90.0, 90.0, 90.0)
    assert cell.volume == pytest.approx(CUBIC_A**3)


def test_triclinic_reciprocal_is_self_consistent():
    """For any cell, d(1 0 0) must equal 1/a*, which is an independent route to the same number."""
    cell = UnitCell(8.5, 10.25, 13.75, 72.0, 81.0, 96.0)
    a_star, _, _, _, _, _ = cell.reciprocal()
    assert cell.d_spacing(1, 0, 0) == pytest.approx(1.0 / a_star, rel=1e-12)


def test_zero_reflection_has_no_d_spacing():
    cell = UnitCell(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    with pytest.raises(ValueError):
        cell.d_spacing(0, 0, 0)


def test_invalid_cell_rejected():
    with pytest.raises(ValueError):
        UnitCell(0.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    with pytest.raises(ValueError):
        UnitCell(5.0, 5.0, 5.0, 0.0, 90.0, 90.0)


# --------------------------------------------------------------------------------------
# instruction file
# --------------------------------------------------------------------------------------


def test_reads_a_conventional_ins(tmp_path: Path):
    path = write_ins(tmp_path / "case.ins")
    header = read_ins_header(path)

    assert header.title == "test structure"
    assert header.wavelength_a == pytest.approx(0.71073)
    assert header.cell is not None and header.cell.a == pytest.approx(CUBIC_A)
    assert header.latt == 1
    assert header.symm == ("-X, -Y, Z",)
    assert header.sfac == ("C", "N", "O", "Ag")
    assert header.unit == (24.0, 4.0, 8.0, 2.0)
    assert header.hklf_format is HklfFormat.HKLF4
    assert header.space_group_confidence is SpaceGroupConfidence.FROM_INS
    assert header.non_hydrogen_unit_total == pytest.approx(38.0)
    assert header.is_centrosymmetric_latt is True
    assert header.sha256


def test_title_containing_instruction_words_is_not_misparsed(tmp_path: Path):
    """The exact failure mode substring matching produces.

    A title mentioning PLAN, UNIT, SFAC or HKLF must not be treated as those instructions.
    """
    path = write_ins(
        tmp_path / "tricky.ins",
        title="PLANar UNIT cell SFAC study HKLF variant END of title",
    )
    header = read_ins_header(path)

    assert header.title == "PLANar UNIT cell SFAC study HKLF variant END of title"
    assert header.sfac == ("C", "N", "O", "Ag")
    assert header.unit == (24.0, 4.0, 8.0, 2.0)
    assert header.hklf_format is HklfFormat.HKLF4
    assert header.cell is not None


def test_continuation_lines_are_joined(tmp_path: Path):
    """SHELX continues an instruction when the line ends with '='.

    Ignoring that silently truncates long SFAC/UNIT/SYMM records.
    """
    path = tmp_path / "cont.ins"
    path.write_text(
        "TITL continuation test\n"
        "CELL 1.54178 10 10 10 90 90 90\n"
        "LATT 1\n"
        "SFAC C N O =\n"
        "     Ag Cl\n"
        "UNIT 24 4 8 =\n"
        "     2 2\n"
        "HKLF 4\n"
        "END\n",
        encoding="utf-8",
    )
    header = read_ins_header(path)
    assert header.sfac == ("C", "N", "O", "Ag", "Cl")
    assert header.unit == (24.0, 4.0, 8.0, 2.0, 2.0)


def test_missing_space_group_is_reported_not_guessed(tmp_path: Path):
    path = write_ins(tmp_path / "nosg.ins", latt=None, symm=())
    header = read_ins_header(path)
    assert header.space_group_confidence is SpaceGroupConfidence.UNKNOWN
    assert header.latt is None


def test_sfac_unit_length_mismatch_warns(tmp_path: Path):
    path = write_ins(tmp_path / "mismatch.ins", sfac=("C", "N", "O"), unit=(24, 4))
    header = read_ins_header(path)
    assert any("inconsistent" in w for w in header.warnings)


def test_long_form_sfac_is_recognised(tmp_path: Path):
    """SFAC has two shapes; the long form declares one element plus scattering coefficients."""
    path = tmp_path / "long.ins"
    path.write_text(
        "TITL long sfac\n"
        "CELL 0.71073 10 10 10 90 90 90\n"
        "LATT 1\n"
        "SFAC Ag 19.2808 0.6446 16.6885 7.4726 4.8045 24.6605 1.0463 99.8156 5.1790 "
        "0.1585 4.2820 0.0 0.0 107.868\n"
        "UNIT 2\n"
        "HKLF 4\n"
        "END\n",
        encoding="utf-8",
    )
    header = read_ins_header(path)
    assert header.sfac == ("Ag",)
    assert header.unit == (2.0,)


def test_malformed_cell_raises_parse_error_with_location(tmp_path: Path):
    path = tmp_path / "bad.ins"
    path.write_text("TITL bad\nCELL 0.71073 10 10\nEND\n", encoding="utf-8")
    with pytest.raises(ParseError) as excinfo:
        read_ins_header(path)
    assert excinfo.value.line_no == 2
    assert "CELL" in str(excinfo.value)


def test_non_numeric_cell_raises_parse_error(tmp_path: Path):
    path = tmp_path / "bad2.ins"
    path.write_text("TITL bad\nCELL 0.71073 ten 10 10 90 90 90\nEND\n", encoding="utf-8")
    with pytest.raises(ParseError):
        read_ins_header(path)


def test_missing_file_raises_parse_error(tmp_path: Path):
    with pytest.raises(ParseError):
        read_ins_header(tmp_path / "nope.ins")


def test_logical_lines_reports_first_line_number(tmp_path: Path):
    path = tmp_path / "lines.ins"
    path.write_text("TITL a\nSFAC C =\n     N\nUNIT 1 1\n", encoding="utf-8")
    lines = list(logical_lines(path))
    assert [line.instruction for line in lines] == ["TITL", "SFAC", "UNIT"]
    assert lines[1].first_line_no == 2
    assert lines[1].text == "SFAC C N"
    assert lines[1].raw_lines == ("SFAC C =", "     N")


# --------------------------------------------------------------------------------------
# reflection file
# --------------------------------------------------------------------------------------


def test_merged_file_reports_no_friedel_pairs(tmp_path: Path):
    path = write_hkl(tmp_path / "merged.hkl", include_friedel=False)
    summary = scan_hkl(path)
    assert summary.anomalous is AnomalousAvailability.MERGED_NO_PAIRS
    assert summary.n_friedel_pairs == 0
    assert not summary.has_unmerged_friedel_pairs


def test_unmerged_file_reports_friedel_pairs(tmp_path: Path):
    path = write_hkl(tmp_path / "anom.hkl", include_friedel=True)
    summary = scan_hkl(path)
    assert summary.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_PRESENT
    assert summary.n_friedel_pairs > 0
    assert summary.has_unmerged_friedel_pairs


def test_friedel_pair_count_is_exact(tmp_path: Path):
    """Two pairs plus one unpaired reflection must count as exactly two pairs."""
    path = write_hkl(
        tmp_path / "pairs.hkl",
        reflections=[
            (1, 0, 0, 10.0, 1.0),
            (-1, 0, 0, 9.8, 1.0),
            (2, 1, 0, 8.0, 1.0),
            (-2, -1, 0, 7.9, 1.0),
            (3, 0, 0, 5.0, 1.0),
        ],
    )
    summary = scan_hkl(path)
    assert summary.n_friedel_pairs == 2
    # Counting them is not the same as believing them; see the sparse-coverage tests below.
    assert summary.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_SPARSE


def test_resolution_computed_when_cell_supplied(tmp_path: Path):
    cell = UnitCell(CUBIC_A, CUBIC_A, CUBIC_A, 90.0, 90.0, 90.0)
    path = write_hkl(
        tmp_path / "res.hkl",
        reflections=[(1, 0, 0, 100.0, 1.0), (5, 5, 5, 10.0, 1.0)],
    )
    summary = scan_hkl(path, cell=cell)
    assert summary.d_max_a == pytest.approx(CUBIC_A)
    assert summary.d_min_a == pytest.approx(CUBIC_A / math.sqrt(75))
    assert summary.mean_i_over_sigma == pytest.approx((100.0 + 10.0) / 2)


def test_resolution_absent_without_cell(tmp_path: Path):
    path = write_hkl(tmp_path / "nores.hkl")
    summary = scan_hkl(path)
    assert summary.d_min_a is None


def test_completeness_is_declared_unavailable_not_guessed(tmp_path: Path):
    """Completeness needs symmetry we do not have. It must say so rather than invent a number."""
    path = write_hkl(tmp_path / "c.hkl")
    summary = scan_hkl(path)
    assert summary.completeness is None
    assert summary.completeness_unavailable is not None
    assert "space-group symmetry" in summary.completeness_unavailable.reason


def test_terminator_stops_the_scan(tmp_path: Path):
    path = tmp_path / "term.hkl"
    path.write_text(
        f"{1:4d}{0:4d}{0:4d}{10.0:8.2f}{1.0:8.2f}\n"
        f"{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}\n"
        f"{9:4d}{9:4d}{9:4d}{99.0:8.2f}{1.0:8.2f}\n",
        encoding="utf-8",
    )
    summary = scan_hkl(path)
    assert summary.n_reflections == 1
    assert summary.truncated_at_terminator


def test_free_format_records_are_accepted(tmp_path: Path):
    path = tmp_path / "free.hkl"
    path.write_text("1 0 0 100.0 2.0\n2 0 0 50.0 2.0\n0 0 0 0.0 0.0\n", encoding="utf-8")
    summary = scan_hkl(path)
    assert summary.n_reflections == 2


def test_a_negative_sixth_column_reads_as_hklf_5(tmp_path: Path):
    """HKLF 5 marks every non-final member of an overlap group with a negative component."""
    path = tmp_path / "twin.hkl"
    path.write_text(
        "   1   0   0  100.00    2.00      -1\n   1   0   0   40.00    2.00       2\n",
        encoding="utf-8",
    )
    summary = scan_hkl(path)
    assert summary.hklf_format is HklfFormat.HKLF5
    assert not summary.hklf_was_declared
    assert any("HKLF 5" in w for w in summary.warnings)


def test_an_all_positive_sixth_column_reads_as_a_batch_number_not_a_twin(tmp_path: Path):
    """A batch number is not a twin component, and calling it one sends the run down the wrong route."""
    path = tmp_path / "batched.hkl"
    path.write_text(
        "   1   0   0  100.00    2.00       1\n   2   0   0   50.00    2.00       1\n",
        encoding="utf-8",
    )
    summary = scan_hkl(path)
    assert summary.hklf_format is HklfFormat.HKLF4
    assert any("batch number" in w for w in summary.warnings)


def test_direction_cosine_records_are_not_mistaken_for_a_twin(tmp_path: Path):
    """Invented twelve-field records exercise direction-cosine detection, not measured data.

    Each row carries five reflection fields, a batch field and two unit-length beam vectors.
    A sixth column alone must not classify this extended format as a twinned reflection file.
    """
    path = tmp_path / "cosines.hkl"
    records = (
        (1, 2, 0, 144.0, 6.0, 1, (0.6, 0.8, 0.0, 0.0, 0.6, 0.8)),
        (2, 0, 1, 225.0, 9.0, 1, (0.8, 0.0, 0.6, -0.6, 0.0, 0.8)),
    )
    lines = [
        f"{h:4d}{k:4d}{l:4d}{intensity:8.2f}{sigma:8.2f}{batch:8d} "
        + " ".join(f"{value:.5f}" for value in cosines)
        for h, k, l, intensity, sigma, batch, cosines in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = scan_hkl(path)
    assert summary.hklf_format is HklfFormat.HKLF4
    assert summary.n_reflections == 2
    assert any("direction cosines" in w for w in summary.warnings)
    assert not any("twinned" in w for w in summary.warnings)


def test_a_handful_of_stray_mates_does_not_unlock_experimental_phasing(tmp_path: Path):
    """Two coincidental pairs in a large merged file are not anomalous data.

    ``FRIEDEL_PAIRS_PRESENT`` is the hard prerequisite for SHELXC/D/E. Before this was fixed,
    any single ``(h,k,l)``/``(-h,-k,-l)`` coincidence satisfied it.
    """
    reflections = [(h, k, 0, 100.0, 2.0) for h in range(1, 21) for k in range(0, 10)]
    reflections += [(-1, 0, 0, 99.0, 2.0), (-2, 0, 0, 98.0, 2.0)]
    path = write_hkl(tmp_path / "mostly_merged.hkl", reflections=reflections)
    summary = scan_hkl(path)

    assert summary.n_reflections == 202
    assert summary.n_friedel_pairs == 2
    assert summary.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_SPARSE
    assert not summary.has_unmerged_friedel_pairs, "sparse must not read as unmerged"
    assert summary.friedel_coverage < 0.05
    assert any("stays closed" in w for w in summary.warnings)
    assert any("merged file" in w for w in summary.warnings)


def test_the_pair_count_floor_is_what_rejects_a_small_but_fully_paired_file(tmp_path: Path):
    """Coverage alone is not enough: two pairs in a four-record file is 100 % coverage.

    Both conditions have to hold, and this is the one the coverage test cannot catch.
    """
    path = write_hkl(
        tmp_path / "tiny.hkl",
        reflections=[
            (1, 0, 0, 10.0, 1.0),
            (-1, 0, 0, 9.8, 1.0),
            (2, 1, 0, 8.0, 1.0),
            (-2, -1, 0, 7.9, 1.0),
        ],
    )
    summary = scan_hkl(path)
    assert summary.friedel_coverage == pytest.approx(1.0)
    assert summary.n_friedel_pairs == 2
    assert summary.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_SPARSE


def test_a_genuinely_unmerged_file_does_unlock_it(tmp_path: Path):
    """The floor must not be so high that real anomalous data trips over it."""
    reflections = []
    for h in range(1, 41):
        reflections.append((h, 0, 0, 100.0, 2.0))
        reflections.append((-h, 0, 0, 99.0, 2.0))
    path = write_hkl(tmp_path / "unmerged.hkl", reflections=reflections)
    summary = scan_hkl(path)

    assert summary.n_friedel_pairs == 40
    assert summary.anomalous is AnomalousAvailability.FRIEDEL_PAIRS_PRESENT
    assert summary.has_unmerged_friedel_pairs
    assert summary.friedel_coverage == pytest.approx(1.0)


def test_explicit_hklf_hint_wins_over_the_guess(tmp_path: Path):
    path = tmp_path / "hint.hkl"
    path.write_text("   1   0   0  100.00    2.00       1\n", encoding="utf-8")
    summary = scan_hkl(path, hklf_hint=HklfFormat.HKLF4)
    assert summary.hklf_format is HklfFormat.HKLF4


def test_empty_reflection_file_raises(tmp_path: Path):
    path = tmp_path / "empty.hkl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ParseError):
        scan_hkl(path)


def test_zero_sigma_reflections_are_counted(tmp_path: Path):
    path = write_hkl(
        tmp_path / "zerosig.hkl",
        reflections=[(1, 0, 0, 10.0, 0.0), (2, 0, 0, 10.0, 1.0)],
    )
    summary = scan_hkl(path)
    assert summary.n_zero_sigma == 1


def test_outer_shell_is_the_high_resolution_end(tmp_path: Path):
    cell = UnitCell(CUBIC_A, CUBIC_A, CUBIC_A, 90.0, 90.0, 90.0)
    reflections = [(h, 0, 0, 100.0 / h, 1.0) for h in range(1, 11)]
    path = write_hkl(tmp_path / "shell.hkl", reflections=reflections)
    summary = scan_hkl(path, cell=cell, outer_shell_fraction=0.2)
    assert summary.outer_shell is not None
    # d = a/h, so the smallest d belongs to the largest h.
    assert summary.outer_shell.d_min_a == pytest.approx(CUBIC_A / 10)
    assert summary.outer_shell.n_reflections == 2
