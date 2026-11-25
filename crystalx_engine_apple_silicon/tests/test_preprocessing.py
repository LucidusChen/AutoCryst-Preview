"""Hand-authored geometry and failure contracts; no real structures/weights."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import pytest

from crystalx_core.domain.data import UnitCell
from crystalx_core.domain.symmetry import SymmetryOperator, parse_symm
from crystalx_core.io.shelx_res import read_res
from crystalx_engine_apple_silicon.preprocessing import (
    CompleteSymmetry,
    PreprocessingBlocked,
    prepare_initial_solution,
)


def operators(*texts):
    parsed = tuple(parse_symm(text) for text in texts)
    assert all(op is not None for op in parsed)
    return CompleteSymmetry(parsed, "hand-authored test operators")


P1 = operators("x,y,z")


def model(tmp_path, *, cell="10 10 10 90 90 90", latt=-1, symm=(),
          atoms="C1 1 0 0 0 11 0.05", peaks="", sfac="C O", unit="1 1"):
    path = tmp_path / "synthetic.res"
    path.write_text("\n".join(("TITL hand-authored synthetic test", f"CELL 0.71073 {cell}",
                              f"LATT {latt}", *("SYMM " + s for s in symm),
                              f"SFAC {sfac}", f"UNIT {unit}", atoms,
                              "HKLF 4", "END", peaks, "")))
    return read_res(path)


def prep(res, **kwargs):
    return prepare_initial_solution(res, symmetry=P1, context_radius_angstrom=1, **kwargs)


def test_orthogonal_fractional_coordinates_and_metadata_are_unchanged(tmp_path):
    res = model(tmp_path, cell="10 20 30 90 90 90", atoms="C1 1 .2 .3 .4 11 .05")
    result = prep(res)
    assert result.request.positions_angstrom[0] == pytest.approx((2, 6, 12))
    assert result.request.atomic_numbers == (6,)
    assert result.main_mask == (True,)
    assert result.request.main_site_count == 1
    assert result.source_sha256 == res.sha256
    assert result.cell_parameters == (10, 20, 30, 90, 90, 90)
    assert result.sites[0].fractional == (.2, .3, .4)
    assert result.sites[0].occupancy_code == 11
    assert result.sites[0].u_values == (.05,)
    assert result.symmetry.source == P1.source
    assert not result.upstream_preprocessing_parity_verified


def test_hand_known_hexagonal_cell_and_periodic_metric(tmp_path):
    res = model(tmp_path, cell="2 2 4 90 90 60", atoms="C1 1 0 0 0 11 .05")
    result = prepare_initial_solution(res, symmetry=P1, context_radius_angstrom=2)
    assert len(result.sites) == 7
    context = {s.lattice_translation: p for s, p in zip(result.sites[1:], result.request.positions_angstrom[1:])}
    assert set(context) == {(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 1, 0), (1, -1, 0)}
    assert context[(0, 1, 0)] == pytest.approx((1, 3 ** .5, 0))
    assert context[(1, -1, 0)] == pytest.approx((1, -(3 ** .5), 0))
    assert result.main_mask == (True,) + (False,) * 6


def test_general_skew_cell_known_cartesian_coordinates(tmp_path):
    # a=(2,0,0), b=(1,sqrt(3),0), c=(0,0,4).
    res = model(tmp_path, cell="2 2 4 90 90 60", atoms="C1 1 .25 .5 .75 11 .05")
    assert prep(res).request.positions_angstrom[0] == pytest.approx((1, 3 ** .5 / 2, 3))


def test_reciprocal_bounds_find_images_beyond_one_unit_in_skew_cell(tmp_path):
    res = model(tmp_path, cell="2 2 10 90 90 10")
    result = prepare_initial_solution(res, symmetry=P1, context_radius_angstrom=.7)
    translations = {s.lattice_translation for s in result.sites}
    assert (2, -2, 0) in translations
    assert (-2, 2, 0) in translations
    assert (3, -3, 0) not in translations


def test_periodic_cross_boundary_context_is_masked_not_output(tmp_path):
    res = model(tmp_path, atoms="C1 1 .99 .5 .5 11 .05\nO1 2 .01 .5 .5 11 .05")
    result = prepare_initial_solution(res, symmetry=P1, context_radius_angstrom=.5)
    assert result.request.main_site_count == 2
    assert result.main_mask == (True, True, False, False)
    assert [p[0] for p in result.request.positions_angstrom] == pytest.approx((9.9, .1, -.1, 10.1))
    assert result.request.atomic_numbers == (6, 8, 6, 8)
    assert [s.source_site_id for s in result.sites[2:]] == [s.site_id for s in result.sites[:2]]


def test_boundary_radius_is_inclusive_with_roundoff_tolerance(tmp_path):
    result = prepare_initial_solution(model(tmp_path), symmetry=P1, context_radius_angstrom=10)
    assert len(result.sites) == 7
    smaller = prepare_initial_solution(model(tmp_path), symmetry=P1, context_radius_angstrom=9.999)
    assert len(smaller.sites) == 1


def test_screw_operator_and_inversion_match_declared_group(tmp_path):
    screw = "-x,y+1/2,-z"
    res = model(tmp_path, cell="10 8 6 90 100 90", symm=(screw,), atoms="C1 1 .1 .1 .1 11 .05")
    result = prepare_initial_solution(res, symmetry=operators("x,y,z", screw), context_radius_angstrom=5)
    assert len(result.symmetry.operators) == 2
    assert any(s.symmetry_operation_index == 1 for s in result.sites[1:])
    # LATT positive adds inversion even when no SYMM record is present.
    inverted = model(tmp_path, latt=1, atoms="C1 1 .1 .2 .3 11 .05")
    result = prepare_initial_solution(inverted, symmetry=operators("x,y,z", "-x,-y,-z"), context_radius_angstrom=8)
    assert any(p == pytest.approx((-1, -2, -3)) for p in result.request.positions_angstrom)


def test_body_centring_expands_all_eight_nearest_images(tmp_path):
    res = model(tmp_path, latt=-2)
    result = prepare_initial_solution(res, symmetry=operators("x,y,z", "x+1/2,y+1/2,z+1/2"), context_radius_angstrom=9)
    assert len(result.sites) == 9
    assert {tuple(round(x) for x in p) for p in result.request.positions_angstrom[1:]} == {
        (x, y, z) for x in (-5, 5) for y in (-5, 5) for z in (-5, 5)}


@pytest.mark.parametrize("latt,texts", [
    (-3, ("x,y,z", "x+2/3,y+1/3,z+1/3", "x+1/3,y+2/3,z+2/3")),
    (-4, ("x,y,z", "x,y+1/2,z+1/2", "x+1/2,y,z+1/2", "x+1/2,y+1/2,z")),
    (-5, ("x,y,z", "x,y+1/2,z+1/2")),
    (-6, ("x,y,z", "x+1/2,y,z+1/2")),
    (-7, ("x,y,z", "x+1/2,y+1/2,z")),
])
def test_centring_translations_match_known_shelx_latt_codes(tmp_path, latt, texts):
    result = prepare_initial_solution(model(tmp_path, latt=latt), symmetry=operators(*texts), context_radius_angstrom=1)
    assert len(result.symmetry.operators) == len(texts)


def test_special_position_aliases_are_not_duplicate_model_nodes(tmp_path):
    res = model(tmp_path, latt=1)
    result = prepare_initial_solution(res, symmetry=operators("x,y,z", "-x,-y,-z"), context_radius_angstrom=1)
    assert len(result.sites) == 1
    assert result.special_position_aliases == 1


def test_all_peaks_preserve_input_order_and_require_hypothesis_descriptors(tmp_path):
    res = model(tmp_path, atoms="O1 1 .1 .1 .1 11 .05\nC1 2 .4 .4 .4 11 .05", unit="100 200",
                peaks="Q2 1 .7 .7 .7 11 .05 2\nQ1 1 .9 .9 .9 11 .05 8")
    before = res
    with pytest.raises(PreprocessingBlocked, match="Every Q peak"):
        prep(res)
    result = prep(res, q_peak_atomic_numbers=(6, 8), candidate_atomic_numbers=(6, 8))
    assert [s.label for s in result.sites] == ["O1", "C1", "Q2", "Q1"]
    # SFAC indices, not misleading labels or UNIT ratios, control atom input.
    assert result.request.atomic_numbers == (6, 8, 6, 8)
    assert result.request.candidate_atomic_numbers == (6, 8)
    assert [s.peak_rank for s in result.sites[2:]] == [1, 2]
    assert [s.peak_height for s in result.sites[2:]] == [2, 8]
    assert [s.source_index for s in result.sites] == [0, 1, 2, 3]
    assert result.request.main_site_count == 4
    assert "hypotheses" in result.warnings[-1]
    assert res == before


def test_repeated_labels_have_stable_unique_site_ids(tmp_path):
    res = model(tmp_path, atoms="C1 1 .1 .1 .1 11 .05\nRESI 2\nC1 1 .4 .4 .4 11 .05")
    first, second = prep(res), prep(res)
    assert first.request == second.request
    assert first.sites[1].residue == 2
    assert first.request.site_ids == ("atom/000001/C1", "atom/000002/C1")


def test_hydrogen_is_retained_without_a_hidden_heavy_filter(tmp_path):
    res = model(tmp_path, sfac="H C", atoms="H1 1 .1 .1 .1 11 .05\nC1 2 .4 .4 .4 11 .05")
    assert prep(res).request.atomic_numbers == (1, 6)


def test_no_file_read_or_optional_import_at_preparation_boundary(tmp_path, monkeypatch):
    res = model(tmp_path)
    def refuse(*args, **kwargs):
        raise AssertionError("preprocessing must consume the parsed snapshot only")
    monkeypatch.setattr("builtins.open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    assert prep(res).source_sha256 == res.sha256


def test_import_does_not_load_torch_or_symmetry_extensions():
    script = "import sys; import crystalx_engine_apple_silicon.preprocessing; assert not any(x in sys.modules for x in ('torch', 'gemmi', 'cctbx', 'numpy'))"
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("bad", [None, CompleteSymmetry((), "test"), CompleteSymmetry(P1.operators, "")])
def test_missing_or_unprovenanced_symmetry_is_a_visible_block(tmp_path, bad):
    with pytest.raises(PreprocessingBlocked, match="symmetry|Symmetry"):
        prepare_initial_solution(model(tmp_path), symmetry=bad, context_radius_angstrom=1)


@pytest.mark.parametrize("latt,symm,supplied", [
    (1, (), P1), (-2, (), P1), (-1, ("-x,y+1/2,-z",), P1),
    (-1, (), operators("x,y,z", "-x,-y,-z")),
])
def test_incomplete_symmetry_and_extra_supergroup_are_rejected(tmp_path, latt, symm, supplied):
    with pytest.raises(PreprocessingBlocked, match="differs"):
        prepare_initial_solution(model(tmp_path, latt=latt, symm=symm), symmetry=supplied, context_radius_angstrom=1)


def test_operator_order_is_canonical_and_integer_translations_are_equivalent(tmp_path):
    res = model(tmp_path, latt=1, atoms="C1 1 .1 .1 .1 11 .05")
    a = prepare_initial_solution(res, symmetry=operators("-x,-y,-z", "x+1,y,z"), context_radius_angstrom=4)
    b = prepare_initial_solution(res, symmetry=operators("x,y,z", "-x,-y,-z"), context_radius_angstrom=4)
    assert a.request == b.request
    assert a.sites == b.sites


@pytest.mark.parametrize("text", ["x+0.333,y,z", "bad input", "x+1/0,y,z", "x+y,y,z"])
def test_unsupported_or_nonclosing_declared_symmetry_is_blocked(tmp_path, text):
    with pytest.raises(PreprocessingBlocked):
        prep(model(tmp_path, symm=(text,)))


def test_cell_metric_must_agree_with_rotation(tmp_path):
    res = model(tmp_path, cell="10 20 30 90 90 90", symm=("y,x,-z",))
    with pytest.raises(PreprocessingBlocked, match="metric"):
        prepare_initial_solution(res, symmetry=operators("x,y,z", "y,x,-z"), context_radius_angstrom=1)


def test_long_unrelated_axis_cannot_hide_short_axis_metric_disagreement(tmp_path):
    res = model(tmp_path, cell="1000000 1 2 90 90 90", symm=("x,z,y",))
    with pytest.raises(PreprocessingBlocked, match="metric"):
        prepare_initial_solution(res, symmetry=operators("x,y,z", "x,z,y"), context_radius_angstrom=1)


def test_malformed_explicit_rotations_and_float_translations_rejected(tmp_path):
    bad = [SymmetryOperator("bad", ((1, 0),), (Fraction(0),) * 3),
           SymmetryOperator("bad", ((0, 0, 0),) * 3, (Fraction(0),) * 3),
           SymmetryOperator("bad", P1.operators[0].rotation, (0.5, 0, 0))]
    for op in bad:
        with pytest.raises(PreprocessingBlocked):
            prepare_initial_solution(model(tmp_path), symmetry=CompleteSymmetry((op,), "test"), context_radius_angstrom=1)


def test_duplicate_operators_are_rejected(tmp_path):
    with pytest.raises(PreprocessingBlocked, match="Duplicate"):
        prepare_initial_solution(model(tmp_path), symmetry=operators("x,y,z", "x+1,y,z"), context_radius_angstrom=1)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), True])
def test_invalid_context_radius(tmp_path, radius):
    with pytest.raises(PreprocessingBlocked, match="radius"):
        prepare_initial_solution(model(tmp_path), symmetry=P1, context_radius_angstrom=radius)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_resource_limits(tmp_path, limit):
    with pytest.raises(PreprocessingBlocked, match="limits"):
        prep(model(tmp_path), max_sites=limit)


def test_resource_limits_block_without_partial_or_truncated_output(tmp_path):
    res = model(tmp_path)
    with pytest.raises(PreprocessingBlocked, match="max_sites"):
        prepare_initial_solution(res, symmetry=P1, context_radius_angstrom=10, max_sites=2)
    with pytest.raises(PreprocessingBlocked, match="max_image_candidates"):
        prepare_initial_solution(res, symmetry=P1, context_radius_angstrom=10, max_image_candidates=2)


def test_extreme_finite_radius_is_a_controlled_resource_block(tmp_path):
    with pytest.raises(PreprocessingBlocked, match="max_image_candidates"):
        prepare_initial_solution(model(tmp_path), symmetry=P1, context_radius_angstrom=1e308)


@pytest.mark.parametrize("change", [
    {"cell": None}, {"latt": None}, {"latt": 0},
    {"cell": UnitCell(float("nan"), 10, 10, 90, 90, 90)},
    {"cell": UnitCell(float("inf"), 10, 10, 90, 90, 90)},
    {"cell": UnitCell(10, 10, 10, 10, 10, 170)},
])
def test_missing_nonfinite_and_degenerate_cell_or_latt(tmp_path, change):
    res = model(tmp_path)
    with pytest.raises(PreprocessingBlocked):
        prep(replace(res, header=replace(res.header, **change)))


def test_parse_warnings_and_provenance_mismatch_are_blocking(tmp_path):
    res = model(tmp_path, atoms="C1 1 0 0 0 11 .05\nBROKEN 1 bad 0 0")
    assert res.warnings
    with pytest.raises(PreprocessingBlocked, match="warnings"):
        prep(res)
    res = model(tmp_path)
    with pytest.raises(PreprocessingBlocked, match="snapshot"):
        prep(replace(res, sha256="f" * 64))


@pytest.mark.parametrize("atoms", [
    "C1 3 0 0 0 11 .05", "C1 1 nan 0 0 11 .05", "C1 1 0 0 0 inf .05",
    "C1 1 10.2 0 0 11 .05", "C1 1 0 0 0 11 nan",
    "PART 1\nC1 1 0 0 0 11 .05", "C1 1 0 0 0 11 .05\nO1 2 0 0 0 11 .05",
])
def test_unsupported_site_information_cannot_be_silently_rewritten(tmp_path, atoms):
    with pytest.raises(PreprocessingBlocked):
        prep(model(tmp_path, atoms=atoms))


def test_symmetry_equivalent_distinct_source_sites_cannot_be_merged(tmp_path):
    res = model(tmp_path, atoms="C1 1 0 0 0 11 .05\nC2 1 1 0 0 11 .05")
    with pytest.raises(PreprocessingBlocked, match="Different source sites"):
        prep(res)


def test_no_sites_or_unknown_sfac_is_blocked(tmp_path):
    with pytest.raises(PreprocessingBlocked, match="at least one"):
        prep(model(tmp_path, atoms=""))
    with pytest.raises(PreprocessingBlocked, match="Unknown SFAC"):
        prep(model(tmp_path, sfac="Xx"))


@pytest.mark.parametrize("descriptors", [(), (6, 8), (0,), (119,), (True,), (6.0,)])
def test_q_descriptor_coverage_and_values_are_strict(tmp_path, descriptors):
    res = model(tmp_path, peaks="Q1 1 .3 .3 .3 11 .05 1")
    with pytest.raises(PreprocessingBlocked, match="Every Q peak"):
        prep(res, q_peak_atomic_numbers=descriptors)
