"""SHELXL preparation, listing and PLATON parsing, and display geometry.

All inputs are synthetic text in SHELX's own layout. Nothing here runs SHELXL or PLATON, and
nothing reads the project's real cases.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from crystalx_core.geometry.model import (
    build_viewer_payload,
    frac_to_cart,
    orthogonalisation_matrix,
    u_equivalent,
)
from crystalx_core.io.shelx_res import read_res
from crystalx_core.io.shelxl_lst import parse_lst, parse_lst_text
from crystalx_core.refinement.shelxl import (
    RefinementSettings,
    prepare_refinement_ins,
    stage_refinement,
)
from crystalx_core.validation.platon import parse_chk_text, validation_prerequisites

SHELXT_LIKE = """\
TITL demo_a.res in P1
REM SHELXT solution in P1: R1 0.180, Rweak 0.040, Alpha 0.000
CELL 1.54184 10.0 11.0 12.0 90 100 90
ZERR 2 0.001 0.001 0.001 0 0.002 0
LATT -1
SFAC C N O ZN
UNIT 40 4 8 2
L.S. 10
PLAN 20
FMAP 2
WGHT 0.1
FVAR 1.0000
ZN1 4 0.10000 0.20000 0.30000 11.00000 0.03
N1  2 0.18000 0.22000 0.31000 11.00000 0.05
C1  1 0.25000 0.24000 0.33000 11.00000 0.05 =
    0.06
Q1  1 0.40000 0.40000 0.40000 11.00000 0.05 1.23
HKLF 4
END
REM trailing summary that must not be copied
"""


# --------------------------------------------------------------------------------------
# SHELXL instruction rewrite
# --------------------------------------------------------------------------------------


def test_the_control_block_replaces_the_solution_programs_own():
    text, changes, atoms, q = prepare_refinement_ins(SHELXT_LIKE, RefinementSettings(cycles=4))
    lines = text.splitlines()
    assert "L.S. 4" in lines and "L.S. 10" not in lines
    assert lines.count("PLAN 20") == 1 and lines.count("FMAP 2") == 1
    assert "LIST 4" in lines, "PLATON needs the FCF this writes"
    assert "ACTA" in lines, "PLATON validates the CIF this writes"
    assert atoms == 3 and q == 1
    assert not any(line.startswith("Q1") for line in lines)
    assert lines[-1] == "END" and "trailing summary" not in text
    assert any("removed 1 existing L.S." in c for c in changes)


def test_everything_the_stage_does_not_own_is_kept_verbatim():
    text, *_ = prepare_refinement_ins(SHELXT_LIKE, RefinementSettings())
    assert "WGHT 0.1" in text and "FVAR 1.0000" in text and "UNIT 40 4 8 2" in text
    # A continued atom record keeps both of its physical lines, unchanged.
    assert "C1  1 0.25000 0.24000 0.33000 11.00000 0.05 =\n    0.06" in text


def test_the_control_block_sits_before_the_scale_factor_and_atoms():
    text, *_ = prepare_refinement_ins(SHELXT_LIKE, RefinementSettings())
    lines = text.splitlines()
    assert lines.index("L.S. 4") < lines.index("FVAR 1.0000") < lines.index(
        "ZN1 4 0.10000 0.20000 0.30000 11.00000 0.03"
    )


def test_a_missing_scale_factor_and_hklf_are_supplied_and_reported():
    bare = "TITL x\nCELL 1.54 10 10 10 90 90 90\nSFAC C\nUNIT 4\nC1 1 0.1 0.1 0.1 11 0.05\n"
    text, changes, *_ = prepare_refinement_ins(bare, RefinementSettings())
    lines = text.splitlines()
    assert lines.index("FVAR 1.00000") < lines.index("C1 1 0.1 0.1 0.1 11 0.05")
    assert lines[-2:] == ["HKLF 4", "END"]
    assert any("FVAR" in c for c in changes) and any("HKLF 4" in c for c in changes)


def test_isotropic_refinement_drops_anis():
    text, changes, *_ = prepare_refinement_ins(
        SHELXT_LIKE.replace("FVAR 1.0000", "ANIS\nFVAR 1.0000"), RefinementSettings()
    )
    assert "ANIS" not in text.split()
    assert any("ANIS" in c for c in changes)


def test_a_solution_without_atoms_is_flagged():
    _, changes, atoms, _ = prepare_refinement_ins(
        "TITL x\nCELL 1.54 10 10 10 90 90 90\nSFAC C\nUNIT 4\nHKLF 4\nEND\n", RefinementSettings()
    )
    assert atoms == 0 and any("no atoms" in c for c in changes)


def test_staging_copies_the_reflections_and_refuses_to_overwrite(tmp_path: Path):
    res = tmp_path / "demo_a.res"
    res.write_text(SHELXT_LIKE)
    hkl = tmp_path / "demo.hkl"
    hkl.write_text("   1   0   0  100.00    2.00\n   0   0   0    0.00    0.00\n")
    work = tmp_path / "refine"
    prepared = stage_refinement(
        solution_res=res, hkl=hkl, workdir=work, stem="demo", settings=RefinementSettings()
    )
    assert prepared.hkl_path.read_text() == hkl.read_text()
    assert not prepared.hkl_path.is_symlink()
    assert prepared.expected_outputs == ("demo.res", "demo.lst")
    with pytest.raises(FileExistsError):
        stage_refinement(solution_res=res, hkl=hkl, workdir=work, stem="demo", settings=RefinementSettings())


@pytest.mark.parametrize("stem", ["../escape", "a b", "x;rm", ""])
def test_unsafe_stems_are_refused(tmp_path: Path, stem: str):
    res = tmp_path / "s.res"
    res.write_text(SHELXT_LIKE)
    with pytest.raises(ValueError):
        stage_refinement(
            solution_res=res, hkl=res, workdir=tmp_path / "w", stem=stem, settings=RefinementSettings()
        )


def test_refinement_settings_are_bounded():
    with pytest.raises(ValueError):
        RefinementSettings(cycles=0)
    with pytest.raises(ValueError):
        RefinementSettings(cycles=500)


# --------------------------------------------------------------------------------------
# SHELXL listing
# --------------------------------------------------------------------------------------

LISTING = """\
 Least-squares cycle  1   Maximum vector length =  511
 Mean shift/su =  0.412  Maximum =  3.112 for  U C1
 Least-squares cycle  4   Maximum vector length =  511
 R1 =  0.1210 for    2766 Fo > 4sig(Fo)  and  0.1434 for all    3069 data
 wR2 =  0.3139,  GooF = S =   1.946,  Restrained GooF =    1.946  for all data
 Mean shift/su =  0.084  Maximum =  0.731 for  x N1
 Highest peak    2.27  at  0.4588  0.2513  0.3349  [  0.81 A from ZN1 ]
 Deepest hole   -1.20  at  0.1946  0.1196  0.0849  [  0.68 A from N1 ]
 Warning: 12 reflections with Fo^2 >> Fc^2 were omitted
"""


def test_the_final_values_are_taken_from_the_last_cycle():
    listing = parse_lst_text(LISTING)
    assert listing.r1_observed == pytest.approx(0.1210)
    assert listing.n_all == 3069 and listing.wr2 == pytest.approx(0.3139)
    assert listing.goof == pytest.approx(1.946)
    assert listing.mean_shift_su == pytest.approx(0.084), "last cycle, not the first"
    assert listing.max_shift_for == "x N1"
    assert listing.cycles_run == 4
    assert listing.highest_peak == pytest.approx(2.27) and listing.deepest_hole == pytest.approx(-1.20)
    assert listing.converged is None  # not assessed: raw mean and maximum are reported instead
    assert listing.max_shift_su == pytest.approx(0.731)
    assert listing.to_record()["convergence"] == "not_assessed"
    assert any("omitted" in w for w in listing.warnings)
    assert listing.format_verified is False


def test_absent_figures_are_none_never_zero():
    listing = parse_lst_text(" nothing recognisable here\n")
    assert listing.r1_observed is None and listing.wr2 is None and listing.goof is None
    assert listing.converged is None
    assert listing.matched == ()


def test_a_small_mean_shift_never_hides_a_large_maximum():
    """Codex R4: 'Mean shift/esd = 0.001 Maximum = 9.000' was reported as converged."""
    listing = parse_lst_text(" Mean shift/esd =  0.001  Maximum =  9.000 for  Zn1\n")
    assert listing.mean_shift_su == pytest.approx(0.001)
    assert listing.max_shift_su == pytest.approx(9.0)
    assert listing.converged is None
    missing_max = parse_lst_text(" Mean shift/su =  0.001\n")
    assert missing_max.converged is None and missing_max.max_shift_su is None


def test_shelxl_errors_are_captured():
    listing = parse_lst_text(" ** CANNOT OPEN FILE demo.hkl **\n")
    assert listing.has_fatal_error and "CANNOT OPEN FILE demo.hkl" in listing.errors


def test_absolute_structure_uncertainty_is_review_only_without_weakening_cannot_errors():
    message = "Absolute structure cannot be determined reliably"
    listing = parse_lst_text(f" ** {message} **\n")
    assert not listing.has_fatal_error and listing.errors == ()
    assert listing.warnings == (message,)
    assert listing.converged is None and listing.format_verified is False
    mixed = parse_lst_text(f" ** {message} **\n ** CANNOT OPEN FILE invented.hkl **\n")
    assert mixed.errors == ("CANNOT OPEN FILE invented.hkl",)
    assert mixed.warnings == (message,)
    extended = parse_lst_text(f" ** {message}; required file cannot be opened **\n")
    assert extended.has_fatal_error


@pytest.mark.parametrize(
    "message", ("ARRAYS TOO SMALL - INCREASE -b", "Arrays  too  small - increase -b")
)
def test_full_matrix_array_limit_is_fatal_without_a_final_r1(message: str):
    listing = parse_lst_text(
        " 8101 l.s. parameters for cycle 1\n"
        f" ** {message} **\n"
        " ** Extinction (EXTI) or solvent water (SWAT) correction may be required **\n"
    )
    assert listing.has_fatal_error and listing.errors == (message,)
    assert listing.r1_observed is None and listing.r1_all is None
    assert listing.notices == ()
    assert len(listing.warnings) == 1
    assert listing.to_record()["errors"] == [message]
    assert listing.format_verified is False


def test_a_missing_listing_is_an_error_not_an_exception(tmp_path: Path):
    listing = parse_lst(tmp_path / "absent.lst")
    assert listing.has_fatal_error


# --------------------------------------------------------------------------------------
# PLATON
# --------------------------------------------------------------------------------------

CHK = """\
# Validation report
PLAT029_ALERT_3_A _diffrn_measured_fraction_theta_full value Low .      0.900 Why?
PLAT220_ALERT_2_C NonSolvent Resd 1 C Ueq(max)/Ueq(min) Range      3.4 Ratio
THETM01_ALERT_3_C The value of sine(theta_max)/wavelength is less than 0.575
PLAT910_ALERT_3_G Missing # of FCF Reflection(s) Below Theta(Min).      3 Note
Some ALERT-shaped line this parser does not understand
"""


def test_alerts_are_read_with_code_type_and_level():
    report = parse_chk_text(CHK)
    assert report.counts == {"A": 1, "B": 0, "C": 2, "G": 1}
    first = report.alerts[0]
    assert (first.code, first.alert_type, first.level) == ("PLAT029", 3, "A")
    assert first.token == "PLAT029_ALERT_3_A"
    assert any(a.code == "THETM01" for a in report.alerts), "non-PLAT checkCIF codes count too"


def test_unrecognised_alert_lines_are_kept_for_inspection():
    report = parse_chk_text(CHK)
    assert any("does not understand" in line for line in report.unrecognised_sample)
    assert report.format_verified is False


def test_local_platon_numeric_alert_codes_are_retained_without_inventing_a_prefix():
    report=parse_chk_text("020_ALERT_3_A Synthetic precision concern\n"
                          "203_ALERT_2_B Synthetic displacement concern\n"
                          "904_ALERT_3_G Synthetic reflection note\n"
                          "20_ALERT_3_A Malformed two-digit code\n")
    assert report.counts=={"A":1,"B":1,"C":0,"G":1}
    assert report.alerts[0].code=="020" and report.alerts[0].token=="020_ALERT_3_A"
    assert any("two-digit" in line for line in report.unrecognised_sample)
    assert not report.format_verified


def test_missing_cif_and_fcf_are_explained(tmp_path: Path):
    reasons = validation_prerequisites(tmp_path, "demo")
    assert len(reasons) == 2
    assert any("ACTA" in r for r in reasons)
    assert any("external SHELXL" in r for r in reasons), "the FCF requirement must say why"
    (tmp_path / "demo.cif").write_text("data_demo\n")
    (tmp_path / "demo.fcf").write_text("data_demo\n")
    assert validation_prerequisites(tmp_path, "demo") == []


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def test_a_cubic_cell_is_a_scaled_identity():
    m = orthogonalisation_matrix((10, 10, 10, 90, 90, 90))
    assert frac_to_cart(m, (0.5, 0.25, 0.1)) == pytest.approx((5.0, 2.5, 1.0))


def test_a_monoclinic_c_axis_tilts_in_the_xz_plane():
    m = orthogonalisation_matrix((10, 11, 12, 90, 107, 90))
    x, y, z = frac_to_cart(m, (0, 0, 1))
    assert x == pytest.approx(12 * math.cos(math.radians(107)))
    assert y == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(12 * math.sin(math.radians(107)))


@pytest.mark.parametrize("cell", [(9.1, 10.3, 11.7, 81.0, 97.0, 104.0), (19.66, 30.15, 31.49, 95.8, 107.0, 99.7)])
def test_a_triclinic_matrix_reproduces_every_length_and_angle(cell):
    a, b, c, alpha, beta, gamma = cell
    m = orthogonalisation_matrix(cell)
    va, vb, vc = (frac_to_cart(m, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def angle(u, v):
        return math.degrees(math.acos(sum(p * q for p, q in zip(u, v)) / (norm(u) * norm(v))))

    assert (norm(va), norm(vb), norm(vc)) == pytest.approx((a, b, c))
    assert angle(vb, vc) == pytest.approx(alpha)
    assert angle(va, vc) == pytest.approx(beta)
    assert angle(va, vb) == pytest.approx(gamma)


def test_u_equivalent_of_an_isotropic_tensor_in_an_orthogonal_cell():
    assert u_equivalent((10, 11, 12, 90, 90, 90), (0.04, 0.04, 0.04, 0, 0, 0)) == pytest.approx(0.04)
    assert u_equivalent((10, 11, 12, 90, 90, 90), (0.05,)) is None


MODEL = """\
TITL demo in P1
CELL 1.54184 10.0 10.0 10.0 90 90 90
LATT -1
SFAC C N XX
UNIT 4 1 1
FVAR 1.0
C1 1 0.10000 0.10000 0.10000 11.00000 0.05
C2 1 0.24000 0.10000 0.10000 11.00000 0.06
C3 1 0.05000 0.50000 0.50000 11.00000 0.05
C4 1 0.95000 0.50000 0.50000 11.00000 0.05
X1 3 0.70000 0.70000 0.70000 11.00000 0.05
HKLF 4
END
Q1 1 0.40000 0.40000 0.40000 11.00000 0.05 1.23
"""


@pytest.fixture
def payload(tmp_path: Path) -> dict:
    path = tmp_path / "demo.res"
    path.write_text(MODEL)
    return build_viewer_payload(read_res(path), stage="test")


def test_a_bond_within_the_cell_is_drawn(payload):
    labels = [a["label"] for a in payload["atoms"]]
    pairs = {(labels[i], labels[j]) for i, j, _ in payload["bonds"]}
    assert ("C1", "C2") in pairs


def test_a_bond_across_the_cell_boundary_is_listed_not_drawn(payload):
    labels = [a["label"] for a in payload["atoms"]]
    pairs = {(labels[i], labels[j]) for i, j, _ in payload["bonds"]}
    assert ("C3", "C4") not in pairs, "a line through the middle of the cell would be a fake bond"
    crossing = [b for b in payload["boundary_bonds"] if {b["a"], b["b"]} == {"C3", "C4"}]
    assert crossing and crossing[0]["distance"] == pytest.approx(1.0)
    assert crossing[0]["translation"] in ([-1, 0, 0], [1, 0, 0])
    assert payload["counts"]["boundary_bonds_not_drawn"] == len(payload["boundary_bonds"])


def test_q_peaks_and_unknown_elements_are_styled_as_what_they_are(payload):
    by_label = {a["label"]: a for a in payload["atoms"]}
    assert by_label["Q1"]["kind"] == "q_peak" and by_label["Q1"]["height"] == pytest.approx(1.23)
    assert by_label["X1"]["kind"] == "unknown"
    labels = [a["label"] for a in payload["atoms"]]
    q_index = labels.index("Q1")
    assert all(q_index not in (i, j) for i, j, _ in payload["bonds"]), "peaks are not bonded"


def test_confidence_is_reported_unavailable_rather_than_invented(payload):
    assert payload["confidence"]["available"] is False
    assert "not as confidence" in payload["confidence"]["reason"]
    c1 = next(a for a in payload["atoms"] if a["label"] == "C1")
    assert c1["occupancy"] == pytest.approx(1.0) and c1["u_iso"] == pytest.approx(0.05)
    assert "confidence" not in c1


def test_the_display_mode_is_stated(payload):
    assert payload["display"] == "asymmetric_unit"
    assert "No symmetry expansion" in payload["display_note"]
    assert len(payload["cell_edges"]) == 12


def test_a_skew_cell_contact_is_found_by_cartesian_distance(tmp_path):
    """Codex review: component-wise rounding picked translation 0 and missed this 1.30 A contact."""
    res = tmp_path / "skew.res"
    res.write_text(
        "TITL skew\nCELL 0.71073 5 5 10 90 90 30\nZERR 1 0 0 0 0 0 0\nLATT -1\nSFAC C\nUNIT 2\n"
        "C1 1 0.0 0.0 0.0 11.0 0.05\nC2 1 0.49 0.49 0.0 11.0 0.05\nHKLF 4\nEND\n",
        encoding="utf-8",
    )
    payload = build_viewer_payload(read_res(res), stage="t")
    assert payload["bonds"] == []
    contacts = {(b["a"], b["b"], tuple(b["translation"])): b["distance"] for b in payload["boundary_bonds"]}
    assert contacts[("C1", "C2", (-1, 0, 0))] == pytest.approx(1.298, abs=1e-3)
    assert "display heuristic" in payload["display_note"]


def test_contact_search_stays_fast_for_a_large_asymmetric_unit(tmp_path):
    import random
    import time

    rng = random.Random(1)
    atoms = "\n".join(
        f"C{i} 1 {rng.random():.4f} {rng.random():.4f} {rng.random():.4f} 11.0 0.05" for i in range(800)
    )
    res = tmp_path / "big.res"
    res.write_text(
        "TITL big\nCELL 0.71073 19.7 30.1 31.5 95.8 107.0 99.7\nZERR 1 0 0 0 0 0 0\nLATT -1\n"
        f"SFAC C\nUNIT 800\n{atoms}\nHKLF 4\nEND\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    build_viewer_payload(read_res(res), stage="t")
    assert time.monotonic() - started < 20
