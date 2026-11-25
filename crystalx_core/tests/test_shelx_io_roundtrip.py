"""The ``.ins`` writer and the ``.res`` reader, including a full round trip.

The round-trip test is the important one: writing a candidate and reading it back must give the same
atoms, elements and coordinates. If it does not, the workflow is silently altering a model between
stages, which is the failure this project exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_ins
from crystalx_core.domain.data import HklfFormat, UnitCell
from crystalx_core.domain.hypothesis import LockDirective, LockMode, format_fixed_shelxl_value
from crystalx_core.errors import InputValidationError
from crystalx_core.io.shelx_reader import read_ins_header
from crystalx_core.io.shelx_res import read_res
from crystalx_core.io.shelx_writer import (
    SHELX_LINE_LIMIT,
    AtomRecord,
    InsDocument,
    format_atom_record,
    render_ins,
    wrap_shelx_line,
    write_ins as write_ins_document,
)

CELL = UnitCell(10.0, 11.0, 12.0, 90.0, 100.0, 90.0)


def document(**overrides) -> InsDocument:
    base = dict(
        title="Ag test",
        cell=CELL,
        wavelength_a=1.54184,
        z=2.0,
        sfac=("C", "N", "O", "Ag"),
        unit=(24.0, 4.0, 8.0, 2.0),
        latt=1,
        symm=("-X, -Y, Z",),
        hklf=HklfFormat.HKLF4,
    )
    base.update(overrides)
    return InsDocument(**base)


# --------------------------------------------------------------------------------------
# atom rows and the fixed-coordinate convention
# --------------------------------------------------------------------------------------


def test_a_free_atom_row_has_plain_coordinates():
    row = format_atom_record(AtomRecord("Ag1", 4, 0.25, 0.5, 0.75, u_values=(0.02,)))
    assert "0.25000" in row
    assert "11.00000" in row  # occupancy fixed at 1.0
    assert row.startswith("Ag1")


def test_a_hard_locked_atom_row_uses_the_free_variable_form():
    row = format_atom_record(
        AtomRecord("Ag1", 4, 0.25, 0.5, 0.75, u_values=(0.02,), fix_coordinates=True)
    )
    assert "10.25000" in row
    assert "10.50000" in row
    assert "10.75000" in row


def test_a_negative_locked_coordinate_wraps_correctly():
    assert format_fixed_shelxl_value(-0.125) == "9.87500"


def test_a_free_occupancy_is_written_without_the_code():
    row = format_atom_record(
        AtomRecord("Q1", 1, 0.1, 0.2, 0.3, occupancy=0.65, fix_occupancy=False)
    )
    assert "0.65000" in row
    assert "10.65000" not in row


def test_an_anisotropic_atom_carries_six_u_values():
    atom = AtomRecord("C1", 1, 0.1, 0.2, 0.3, u_values=(0.02, 0.03, 0.04, 0.001, 0.002, 0.003))
    assert atom.is_anisotropic
    assert format_atom_record(atom).count("0.0") >= 6


def test_a_bad_sfac_index_is_refused():
    with pytest.raises(InputValidationError):
        AtomRecord("C1", 0, 0.1, 0.2, 0.3)


def test_a_wrong_number_of_u_values_is_refused():
    with pytest.raises(InputValidationError):
        AtomRecord("C1", 1, 0.1, 0.2, 0.3, u_values=(0.01, 0.02, 0.03))


# --------------------------------------------------------------------------------------
# line wrapping
# --------------------------------------------------------------------------------------


def test_a_short_line_is_left_alone():
    assert wrap_shelx_line("SFAC C N O") == ["SFAC C N O"]


def test_a_long_line_is_split_with_the_continuation_marker():
    line = "UNIT " + " ".join(str(i) for i in range(60))
    wrapped = wrap_shelx_line(line)
    assert len(wrapped) > 1
    assert all(len(part) <= SHELX_LINE_LIMIT for part in wrapped)
    assert all(part.endswith("=") for part in wrapped[:-1])
    assert not wrapped[-1].endswith("=")


def test_wrapping_never_splits_a_token():
    line = "SYMM " + " ".join(f"0.{i:04d}+X" for i in range(30))
    wrapped = wrap_shelx_line(line)
    rejoined = " ".join(part.rstrip("=").strip() for part in wrapped)
    assert rejoined.split() == line.split()


def test_a_wrapped_line_survives_the_reader(tmp_path: Path):
    """Wrapping and the reader's continuation joining must be exact inverses."""
    elements = tuple(f"C" for _ in range(1)) + ("N", "O", "Ag", "Cl", "Br", "S", "P", "F", "I")
    doc = document(sfac=elements, unit=tuple(float(i + 1) for i in range(len(elements))))
    doc.symm = tuple(f"0.5+X, 0.5-Y, {i}+Z" for i in range(8))
    path = write_ins_document(doc, tmp_path / "wrapped.ins")

    header = read_ins_header(path)
    assert header.sfac == tuple(e.capitalize() for e in elements)
    assert header.unit == tuple(float(i + 1) for i in range(len(elements)))
    assert len(header.symm) == 8


# --------------------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------------------


def test_sfac_and_unit_must_agree():
    with pytest.raises(InputValidationError):
        document(sfac=("C", "N"), unit=(4.0,))


def test_sfac_lookup_is_case_insensitive_and_one_based():
    doc = document()
    assert doc.sfac_index("Ag") == 4
    assert doc.sfac_index("ag") == 4
    assert doc.sfac_index("C") == 1


def test_an_unknown_element_is_refused_rather_than_guessed():
    with pytest.raises(InputValidationError) as excinfo:
        document().sfac_index("Pt")
    assert "not in the SFAC list" in str(excinfo.value)


def test_rendering_includes_the_required_records():
    doc = document()
    doc.add_atom(AtomRecord("Ag1", 4, 0.25, 0.25, 0.25, u_values=(0.02,)))
    text = render_ins(doc)
    for keyword in ("TITL", "CELL", "ZERR", "LATT", "SYMM", "SFAC", "UNIT", "HKLF 4", "END"):
        assert keyword in text
    assert text.rstrip().endswith("END")


def test_from_header_preserves_cell_symmetry_and_composition(tmp_path: Path):
    source = write_ins(tmp_path / "source.ins")
    header = read_ins_header(source)
    doc = InsDocument.from_header(header, title="derived")
    assert doc.sfac == header.sfac
    assert doc.unit == header.unit
    assert doc.latt == header.latt
    assert doc.symm == header.symm
    assert doc.cell is not None and doc.cell.a == pytest.approx(header.cell.a)


def test_from_header_refuses_a_file_with_no_cell(tmp_path: Path):
    path = tmp_path / "nocell.ins"
    path.write_text("TITL nothing\nSFAC C\nUNIT 1\nEND\n", encoding="utf-8")
    with pytest.raises(InputValidationError):
        InsDocument.from_header(read_ins_header(path))


# --------------------------------------------------------------------------------------
# lock directives in the written file
# --------------------------------------------------------------------------------------


def test_a_hard_lock_fixes_the_named_atoms_and_states_why(tmp_path: Path):
    doc = document()
    doc.add_atom(AtomRecord("Ag1", 4, 0.25, 0.5, 0.75, u_values=(0.02,)))
    doc.add_atom(AtomRecord("C1", 1, 0.10, 0.20, 0.30, u_values=(0.03,)))
    doc.apply_lock(
        LockDirective(
            mode=LockMode.HARD_LOCK,
            atom_ids=("Ag1",),
            reason="SHELXD CC and an ANODE anomalous peak both place this silver",
        )
    )
    text = render_ins(doc)

    assert "10.25000" in text  # Ag1 fixed
    assert "REM lock hard_lock on Ag1" in text
    assert "ANODE anomalous peak" in text
    # C1 is untouched.
    assert "0.10000" in text and "10.10000" not in text


def test_a_soft_restraint_emits_its_restraint_lines(tmp_path: Path):
    doc = document()
    doc.add_atom(AtomRecord("Ag1", 4, 0.25, 0.5, 0.75, u_values=(0.02,)))
    doc.apply_lock(
        LockDirective(
            mode=LockMode.SOFT_RESTRAINT,
            atom_ids=("Ag1", "N1"),
            reason="hold the Ag-N distance near the expected window while the shell grows",
            instruction_lines=("DFIX 2.15 0.02 Ag1 N1",),
        )
    )
    text = render_ins(doc)
    assert "DFIX 2.15 0.02 Ag1 N1" in text
    assert "10.25000" not in text  # a restraint does not fix coordinates


def test_exploratory_mode_frees_the_occupancy_so_the_data_can_vote():
    doc = document()
    doc.add_atom(AtomRecord("Ag2", 4, 0.4, 0.4, 0.4, occupancy=0.5, u_values=(0.05,)))
    doc.apply_lock(
        LockDirective(
            mode=LockMode.EXPLORATORY,
            atom_ids=("Ag2",),
            reason="second silver site is unconfirmed; let the occupancy decide",
        )
    )
    text = render_ins(doc)
    assert "0.50000" in text
    assert "10.50000" not in text
    assert "REM lock exploratory on Ag2" in text


def test_a_lock_reason_is_always_present_in_the_file():
    doc = document()
    doc.add_atom(AtomRecord("Ag1", 4, 0.25, 0.5, 0.75))
    doc.apply_lock(
        LockDirective(LockMode.HARD_LOCK, ("Ag1",), "confirmed by two independent sources")
    )
    assert "confirmed by two independent sources" in render_ins(doc)


# --------------------------------------------------------------------------------------
# never clobber a user file
# --------------------------------------------------------------------------------------


def test_writing_refuses_to_overwrite_by_default(tmp_path: Path):
    target = tmp_path / "existing.ins"
    target.write_text("PRECIOUS USER DATA\n", encoding="utf-8")
    with pytest.raises(InputValidationError) as excinfo:
        write_ins_document(document(), target)
    assert "Refusing to overwrite" in str(excinfo.value)
    assert target.read_text() == "PRECIOUS USER DATA\n"


def test_overwrite_is_possible_when_explicitly_requested(tmp_path: Path):
    target = tmp_path / "mine.ins"
    write_ins_document(document(), target)
    write_ins_document(document(title="second"), target, allow_overwrite=True)
    assert "TITL second" in target.read_text()


# --------------------------------------------------------------------------------------
# .res reading
# --------------------------------------------------------------------------------------


SHELXL_RES = """TITL agpep_AI in P2(1)/c
CELL 1.54184 10.0000 11.0000 12.0000 90.000 100.000 90.000
ZERR 2.00 0.0010 0.0010 0.0010 0.000 0.000 0.000
LATT 1
SYMM -X, 0.5+Y, 0.5-Z
SFAC C N O Ag
UNIT 24 4 8 2
REM agpep_AI.res created by SHELXL-2018/3 at 12:00:00 on 18-Aug-2026
L.S. 10
PLAN 20
FVAR 0.31459
AG1   4  0.25000  0.50000  0.75000  11.00000  0.02000
N1    2  0.31000  0.42000  0.66000  11.00000  0.03000  0.03100 =
         0.03200  0.00100  0.00200  0.00300
C1    1  0.40000  0.35000  0.60000  11.00000  0.04000
PART 1
O1    3  0.55000  0.25000  0.50000  10.50000  0.06000
PART 0
HKLF 4

END

WGHT      0.0500      0.0000
REM wR2 = 0.1103, GooF = S = 1.042, Restrained GooF = 1.038 for all data
REM R1 = 0.0412 for 3456 Fo > 4sig(Fo) and 0.0567 for all 4567 data
REM 310 parameters refined using 6 restraints
REM Highest difference peak 1.230, deepest hole -0.812, 1-sigma level 0.101
Q1    1  0.12000  0.13000  0.14000  11.00000  0.05000  1.23
Q2    1  0.22000  0.23000  0.24000  11.00000  0.05000  0.87
"""


def test_reads_a_shelxl_res(tmp_path: Path):
    path = tmp_path / "agpep_AI.res"
    path.write_text(SHELXL_RES, encoding="utf-8")
    res = read_res(path)

    labels = [a.label for a in res.atoms]
    assert labels == ["AG1", "N1", "C1", "O1"]
    assert res.header.sfac == ("C", "N", "O", "Ag")
    assert res.sha256


def test_element_comes_from_the_sfac_index_not_the_label(tmp_path: Path):
    path = tmp_path / "case.res"
    path.write_text(SHELXL_RES, encoding="utf-8")
    res = read_res(path)
    silver = next(a for a in res.atoms if a.label == "AG1")
    assert res.element_of(silver) == "Ag"
    assert silver.element_prefix == "AG"


def test_a_label_that_contradicts_its_sfac_index_is_surfaced(tmp_path: Path):
    path = tmp_path / "wrong.res"
    path.write_text(
        SHELXL_RES.replace(
            "C1    1  0.40000", "N9    1  0.40000"  # label says N, SFAC index 1 says C
        ),
        encoding="utf-8",
    )
    res = read_res(path)
    disagreements = res.label_element_disagreements()
    assert ("N9", "C", "N") in disagreements


def test_anisotropic_continuation_gives_six_u_values(tmp_path: Path):
    path = tmp_path / "aniso.res"
    path.write_text(SHELXL_RES, encoding="utf-8")
    res = read_res(path)
    nitrogen = next(a for a in res.atoms if a.label == "N1")
    assert nitrogen.is_anisotropic
    assert len(nitrogen.u_values) == 6


def test_occupancy_codes_are_decoded(tmp_path: Path):
    path = tmp_path / "occ.res"
    path.write_text(SHELXL_RES, encoding="utf-8")
    res = read_res(path)
    silver = next(a for a in res.atoms if a.label == "AG1")
    oxygen = next(a for a in res.atoms if a.label == "O1")
    assert silver.occupancy == pytest.approx(1.0)
    assert silver.occupancy_is_fixed
    assert oxygen.occupancy == pytest.approx(0.5)
    assert oxygen.occupancy_is_fixed


def test_part_membership_is_tracked(tmp_path: Path):
    path = tmp_path / "part.res"
    path.write_text(SHELXL_RES, encoding="utf-8")
    res = read_res(path)
    oxygen = next(a for a in res.atoms if a.label == "O1")
    carbon = next(a for a in res.atoms if a.label == "C1")
    assert oxygen.part == 1
    assert carbon.part is None


def test_q_peaks_are_separated_from_atoms_and_ranked(tmp_path: Path):
    path = tmp_path / "q.res"
    path.write_text(SHELXL_RES, encoding="utf-8")
    res = read_res(path)
    assert [q.label for q in res.q_peaks] == ["Q1", "Q2"]
    assert res.q_peaks[0].height == pytest.approx(1.23)
    assert res.strongest_q_peaks(1)[0].label == "Q1"
    assert "Q1" not in [a.label for a in res.atoms]


def test_r_factors_are_recovered_from_the_rem_lines(tmp_path: Path):
    path = tmp_path / "r.res"
    path.write_text(SHELXL_RES, encoding="utf-8")
    r = read_res(path).r_factors
    assert r.wr2 == pytest.approx(0.1103)
    assert r.goof == pytest.approx(1.042)
    assert r.restrained_goof == pytest.approx(1.038)
    assert r.r1_observed == pytest.approx(0.0412)
    assert r.r1_all == pytest.approx(0.0567)
    assert r.n_observed == 3456
    assert r.n_all == 4567
    assert r.n_parameters == 310
    assert r.n_restraints == 6
    assert r.highest_peak == pytest.approx(1.230)
    assert r.deepest_hole == pytest.approx(-0.812)
    assert not r.is_empty


def test_a_shelxt_style_res_has_no_r_factors_and_says_so(tmp_path: Path):
    """Absent is absent, never zero."""
    path = tmp_path / "shelxt_a.res"
    path.write_text(
        "TITL sol\nCELL 1.54184 10 11 12 90 100 90\nLATT 1\nSFAC C Ag\nUNIT 20 2\n"
        "FVAR 0.5\nAG1  2  0.25 0.5 0.75 11.00000 0.02\nHKLF 4\nEND\n",
        encoding="utf-8",
    )
    r = read_res(path).r_factors
    assert r.is_empty
    assert r.r1_observed is None
    assert r.wr2 is None


def test_instructions_are_never_mistaken_for_atoms(tmp_path: Path):
    path = tmp_path / "instr.res"
    path.write_text(
        "TITL x\nCELL 1.54184 10 11 12 90 100 90\nLATT 1\nSFAC C\nUNIT 4\n"
        "WGHT 0.0500 0.0000\nFVAR 0.5\nBOND $H\nCONF\nHTAB\n"
        "C1   1  0.1 0.2 0.3 11.00000 0.02\nHKLF 4\nEND\n",
        encoding="utf-8",
    )
    res = read_res(path)
    assert [a.label for a in res.atoms] == ["C1"]


def test_out_of_range_sfac_index_produces_a_warning_not_a_wrong_element(tmp_path: Path):
    path = tmp_path / "badsfac.res"
    path.write_text(
        "TITL x\nCELL 1.54184 10 11 12 90 100 90\nLATT 1\nSFAC C\nUNIT 4\nFVAR 0.5\n"
        "AG1  7  0.25 0.5 0.75 11.00000 0.02\nHKLF 4\nEND\n",
        encoding="utf-8",
    )
    res = read_res(path)
    assert any("outside the" in w for w in res.warnings)
    assert res.element_of(res.atoms[0]) == "AG"  # falls back to the label, and said so


def test_missing_res_file_raises_parse_error(tmp_path: Path):
    from crystalx_core.errors import ParseError

    with pytest.raises(ParseError):
        read_res(tmp_path / "absent.res")


# --------------------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------------------


def test_write_then_read_preserves_the_model(tmp_path: Path):
    """The core guarantee: a candidate written out and read back is the same model."""
    doc = document()
    atoms = [
        AtomRecord("Ag1", 4, 0.25000, 0.50000, 0.75000, u_values=(0.02,)),
        AtomRecord("N1", 2, 0.31000, 0.42000, 0.66000, u_values=(0.03,)),
        AtomRecord("C1", 1, 0.40000, 0.35000, 0.60000, u_values=(0.04,)),
        AtomRecord("O1", 3, 0.55000, 0.25000, 0.50000, u_values=(0.05,)),
    ]
    for atom in atoms:
        doc.add_atom(atom)
    path = write_ins_document(doc, tmp_path / "candidate.ins")

    res = read_res(path)
    assert [a.label for a in res.atoms] == ["Ag1", "N1", "C1", "O1"]
    for original, parsed in zip(atoms, res.atoms):
        assert parsed.x == pytest.approx(original.x)
        assert parsed.y == pytest.approx(original.y)
        assert parsed.z == pytest.approx(original.z)
        assert parsed.sfac_index == original.sfac_index
        assert res.element_of(parsed) == doc.sfac[original.sfac_index - 1]
        assert parsed.occupancy == pytest.approx(1.0)


def test_round_trip_preserves_a_hard_lock(tmp_path: Path):
    doc = document()
    doc.add_atom(AtomRecord("Ag1", 4, 0.25, 0.5, 0.75, u_values=(0.02,)))
    doc.apply_lock(LockDirective(LockMode.HARD_LOCK, ("Ag1",), "validated anchor"))
    path = write_ins_document(doc, tmp_path / "locked.ins")

    res = read_res(path)
    silver = res.atoms[0]
    # The reader sees the raw 10.xxxxx code; decoding it back gives the original coordinate.
    assert silver.x == pytest.approx(10.25)
    assert silver.x - 10.0 == pytest.approx(0.25)
    assert "REM lock hard_lock on Ag1" in path.read_text()
