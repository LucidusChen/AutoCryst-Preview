"""Invented small RES structures only; no private files or expert models."""

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import crystalx_core.app.model_comparison as module
from crystalx_core.app.model_comparison import ComparisonPolicy, compare_sealed_res
from crystalx_core.private_acceptance.sealing import seal_run


ATOMS = (("Zn1", 1, (.11, .21, .31)), ("C1", 2, (.27, .12, .39)), ("N1", 3, (.43, .36, .14)),
         ("O1", 4, (.62, .73, .48)), ("H1", 5, (.28, .13, .4)))


def structure(atoms=ATOMS, *, latt=-1, cell=(10, 11, 12, 90, 90, 90), symm=()):
    lines = ["TITL hand-authored comparison fixture", "CELL 0.7 " + " ".join(map(str, cell)), f"LATT {latt}"]
    lines.extend("SYMM " + value for value in symm)
    lines += ["SFAC Zn C N O H", "UNIT 1 1 1 1 1", "WGHT 0.1 0.2", "FVAR 0.5"]
    lines.extend(f"{label} {sfac} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} 11 0.05" for label, sfac, xyz in atoms)
    lines += ["HKLF 4", "END", "REM R1 = 0.1234 for 20 Fo > 4sig(Fo) and 0.1357 for all 25 data",
              "REM 17 parameters refined using 0 restraints"]
    return "\n".join(lines) + "\n"


def inputs(tmp_path, *, candidate=None, expert=None):
    private = tmp_path.resolve() / "runs"
    run = private / "invented-candidate"
    run.mkdir(parents=True)
    (run / "candidate.res").write_text(candidate or structure())
    seal = seal_run(run, private_root=private, case_id="invented", track="de_novo", outputs={"model": "candidate.res"})
    expert_root = tmp_path.resolve() / "invented-expert"
    expert_root.mkdir()
    expert_path = expert_root / "held-out.res"
    expert_path.write_text(expert or structure())
    return dict(run_root=run, private_root=private, expected_seal_sha256=seal.seal_sha256,
                expert_path=expert_path, expert_root=expert_root,
                policy=ComparisonPolicy(.35, ("Zn",), ("C1", "N1"), ("C1", "N1")))


def optional_geometry():
    pytest.importorskip("gemmi")
    pytest.importorskip("scipy.optimize")


def test_expert_access_trap_when_candidate_seal_is_invalid(tmp_path, monkeypatch):
    arguments = inputs(tmp_path)
    arguments["expected_seal_sha256"] = "f" * 64

    def trap(*args, **kwargs):
        pytest.fail("Expert must not be opened before the candidate seal is verified")

    monkeypatch.setattr(module, "read_private_reference", trap)
    assert compare_sealed_res(**arguments).status == "NOT_EVALUATED"


def test_reordered_renamed_sites_match_by_geometry_not_names(tmp_path):
    optional_geometry()
    atoms = tuple((label + "X", sfac, xyz) for label, sfac, xyz in reversed(ATOMS))
    arguments = inputs(tmp_path, candidate=structure(atoms))
    arguments["policy"] = replace(arguments["policy"], framework_candidate_site_ids=("C1X", "N1X"))
    result = compare_sealed_res(**arguments)
    assert result.status == "MEASURED_REVIEW_REQUIRED", result.reasons
    assert len(result.matches) == len(ATOMS)
    assert max(match.distance_angstrom for match in result.matches) < 1e-8
    assert {category.category: category.expert_count for category in result.categories} == {
        "metal": 1, "framework": 2, "light": 1, "hydrogen": 1}
    assert not result.scientifically_reviewed and not result.expert_used_for_solver
    assert result.framework_topology_status == "NOT_EVALUATED"


def test_origin_shift_and_periodic_wrapping_are_supported(tmp_path):
    optional_geometry()
    shift = (.17, -.09, .13)
    moved = tuple((label, sfac, tuple((v + shift[i]) % 1 for i, v in enumerate(xyz))) for label, sfac, xyz in ATOMS)
    result = compare_sealed_res(**inputs(tmp_path, candidate=structure(moved)))
    assert len(result.matches) == len(ATOMS)
    assert max(match.distance_angstrom for match in result.matches) < 1e-7
    assert result.origin_shift_fractional is not None


def test_symmetry_equivalent_individual_asu_images_do_not_need_same_coordinates(tmp_path):
    optional_geometry()
    moved = tuple((label, sfac, tuple((-value) % 1 for value in xyz) if i % 2 else xyz)
                  for i, (label, sfac, xyz) in enumerate(ATOMS))
    result = compare_sealed_res(**inputs(tmp_path, candidate=structure(moved, latt=1), expert=structure(latt=1)))
    assert len(result.matches) == len(ATOMS)
    assert max(match.distance_angstrom for match in result.matches) < 1e-7


def test_wrong_element_hypothesis_is_reported_despite_correct_geometric_site(tmp_path):
    optional_geometry()
    atoms = (("Zn1", 2, ATOMS[0][2]),) + ATOMS[1:]
    result = compare_sealed_res(**inputs(tmp_path, candidate=structure(atoms)))
    metal = next(row for row in result.categories if row.category == "metal")
    assert metal.recovered_expert_sites == 1
    assert metal.element_mismatches == 1 and metal.same_element_matches == 0
    assert result.inventory["missing_by_element_inventory"]["Zn"] == 1


def test_mismatched_cell_is_partial_without_fabricated_rmsd(tmp_path):
    arguments = inputs(tmp_path, candidate=structure(cell=(13, 11, 12, 90, 90, 90)))
    result = compare_sealed_res(**arguments)
    assert result.status == "PARTIAL"
    assert result.origin_shift_fractional is None
    assert all(row.rmsd_angstrom is None and row.recovered_expert_sites is None for row in result.categories)
    assert result.inventory["candidate_site_counts"]["Zn"] == 1


def test_mislabeled_symmetry_is_not_overridden_by_coordinate_agreement(tmp_path):
    optional_geometry()
    result = compare_sealed_res(**inputs(tmp_path, candidate=structure(latt=-1), expert=structure(latt=1)))
    assert result.status == "PARTIAL" and not result.matches
    assert any("symmetries" in reason for reason in result.reasons)


def test_metric_periodic_search_does_not_use_fractional_rounding_in_skew_cells():
    optional_geometry()
    import gemmi
    matcher = module._Matcher(gemmi.UnitCell(10, 10, 11, 90, 90, 20), ComparisonPolicy(2.0, ("Zn",)), float("inf"))
    found = matcher.distance((.48, .48, 0), (0, 0, 0))
    assert found is not None and found[0] < 2.0
    assert found[1] != (0, 0, 0)


def test_no_framework_declaration_is_explicitly_not_evaluated(tmp_path):
    optional_geometry()
    arguments = inputs(tmp_path)
    arguments["policy"] = ComparisonPolicy(.35, ("Zn",))
    result = compare_sealed_res(**arguments)
    framework = next(row for row in result.categories if row.category == "framework")
    assert framework.expert_count is None and framework.rmsd_angstrom is None
    assert framework.role_status.startswith("NOT_EVALUATED")


def test_missing_optional_provider_retains_inventory_without_false_failure_scores(tmp_path, monkeypatch):
    arguments = inputs(tmp_path)

    def absent(name):
        raise ImportError("deliberately absent optional provider")

    monkeypatch.setattr(module.importlib, "import_module", absent)
    result = compare_sealed_res(**arguments)
    assert result.status == "PARTIAL" and not result.matches
    assert result.inventory["expert_site_counts"]["Zn"] == 1
    assert all(category.missing_expert_sites is None for category in result.categories)


def test_expert_pin_and_path_boundary_are_enforced_after_seal(tmp_path):
    arguments = inputs(tmp_path)
    result = compare_sealed_res(**arguments, expected_expert_sha256="f" * 64)
    assert result.status == "NOT_EVALUATED"
    outside = arguments["expert_root"].parent / "outside.res"
    outside.write_text(structure())
    arguments["expert_path"] = outside
    assert compare_sealed_res(**arguments).status == "NOT_EVALUATED"


def test_reflection_and_parameter_compatibility_are_required_for_r_comparison(tmp_path):
    arguments = inputs(tmp_path)
    result = compare_sealed_res(**arguments)
    assert not result.r_comparison["comparable"] and result.r_comparison["r1_observed_delta"] is None


def test_distance_budget_returns_partial_not_a_fake_complete_match(tmp_path):
    optional_geometry()
    arguments = inputs(tmp_path)
    arguments["policy"] = replace(arguments["policy"], max_pair_evaluations=1)
    result = compare_sealed_res(**arguments)
    assert result.status == "PARTIAL"
    assert not result.matches and all(row.rmsd_angstrom is None for row in result.categories)


def test_missing_and_extra_sites_are_separate_from_formula_completeness(tmp_path):
    optional_geometry()
    atoms = tuple(atom for atom in ATOMS if atom[0] != "O1") + (("C9", 2, (.88, .87, .86)),)
    result = compare_sealed_res(**inputs(tmp_path, candidate=structure(atoms)))
    light = next(row for row in result.categories if row.category == "light")
    assert light.missing_expert_sites == light.extra_candidate_sites == 1
    assert result.inventory["missing_by_element_inventory"] == {"O": 1}
    assert result.inventory["extra_by_element_inventory"] == {"C": 1}
    assert "not_chemical_formula" in result.inventory["composition_completeness_scope"]


def test_r_comparison_requires_matching_real_basis_declarations_and_parameter_count(tmp_path):
    optional_geometry()
    arguments = inputs(tmp_path)
    arguments["policy"] = replace(arguments["policy"], candidate_reflection_sha256="a" * 64,
                                    expert_reflection_sha256="a" * 64)
    assert compare_sealed_res(**arguments).r_comparison["comparable"]
    second = tmp_path / "second"
    second.mkdir()
    mismatch = inputs(second, expert=structure().replace("17 parameters refined", "18 parameters refined"))
    mismatch["policy"] = arguments["policy"]
    assert not compare_sealed_res(**mismatch).r_comparison["comparable"]


def test_centrosymmetric_allowed_origin_shift_is_verified_by_operator_conjugacy(tmp_path):
    optional_geometry()
    moved = tuple((label, sfac, tuple((v + (.5, 0, .5)[i]) % 1 for i, v in enumerate(xyz))) for label, sfac, xyz in ATOMS)
    result = compare_sealed_res(**inputs(tmp_path, candidate=structure(moved, latt=1), expert=structure(latt=1)))
    assert len(result.matches) == len(ATOMS)
    assert max(match.distance_angstrom for match in result.matches) < 1e-7


def test_symmetry_duplicate_disorder_sites_are_not_counted_as_independent_recovery(tmp_path):
    optional_geometry()
    atoms = ATOMS + (("C8", 2, tuple((-v) % 1 for v in ATOMS[1][2])),)
    result = compare_sealed_res(**inputs(tmp_path, candidate=structure(atoms, latt=1), expert=structure(latt=1)))
    assert result.status == "PARTIAL" and not result.matches
    assert any("symmetry_duplicate" in reason for reason in result.reasons)


def test_known_expert_seeded_component_is_not_relabelled_independent_comparison(tmp_path, monkeypatch):
    arguments = inputs(tmp_path)
    old = arguments["run_root"]
    run = old.parent / "component-only"
    run.mkdir()
    (run / "model.res").write_text(structure())
    seal = seal_run(run, private_root=arguments["private_root"], case_id="invented", track="component_regression",
                    outputs={"model": "model.res"})
    arguments.update(run_root=run, expected_seal_sha256=seal.seal_sha256)

    def trap(*args, **kwargs):
        pytest.fail("A known expert-seeded component cannot enter the independent comparison route")

    monkeypatch.setattr(module, "read_private_reference", trap)
    result = compare_sealed_res(**arguments)
    assert result.status == "NOT_EVALUATED" and result.expert_used_for_solver


def test_scoped_residue_coordinate_view_namespaces_sites_without_touching_raw_models(tmp_path):
    optional_geometry()
    expert=structure().replace("FVAR 0.5","DFIX_ALA 1.40 C1 N1\nFVAR 0.5")
    expert=expert.replace("C1 2 ","RESI ALA A:1\nC1 2 ").replace("N1 3 ","RESI GLY A:2\nC1 3 ")
    arguments=inputs(tmp_path,expert=expert)
    arguments['policy']=ComparisonPolicy(.35,('Zn',),coordinate_identity_mode='scoped_residues')
    before=arguments['expert_path'].read_bytes()
    result=compare_sealed_res(**arguments)
    assert len(result.matches)==5,result.reasons
    assert {'A:1/C1','A:2/C1'} <= {match.expert_site_id for match in result.matches}
    assert max(match.distance_angstrom for match in result.matches)<1e-8
    assert arguments['expert_path'].read_bytes()==before
    receipt=result.settings['coordinate_identity_views']['expert']
    assert not receipt['coordinates_modified'] and receipt['unassessed_scoped_restraint_commands']==['DFIX_ALA']


def test_partial_inventory_counts_repeated_raw_labels_as_distinct_unmatched_rows(tmp_path):
    expert=structure().replace('N1 3 ','C1 3 ')
    arguments=inputs(tmp_path,expert=expert);arguments['policy']=ComparisonPolicy(.35,('Zn',))
    result=compare_sealed_res(**arguments)
    light=next(row for row in result.categories if row.category=='light')
    assert result.status=='PARTIAL' and light.expert_count==3
    assert light.recovered_expert_sites is None and not result.matches


def test_invalid_scoped_residue_does_not_fall_back_to_flat_identity(tmp_path):
    arguments=inputs(tmp_path,expert=structure().replace('C1 2 ','RESI ALA A:?\nC1 2 '))
    arguments['policy']=ComparisonPolicy(.35,('Zn',),coordinate_identity_mode='scoped_residues')
    result=compare_sealed_res(**arguments)
    assert result.status=='PARTIAL' and not result.matches
    assert 'explicit_residue_or_chain_number_required_for_coordinate_identity' in result.reasons


def test_metal_origin_selection_evaluates_full_site_assignment_once(tmp_path):
    optional_geometry()
    moved=tuple((label,sfac,tuple((v+(.17,.21,.13)[i])%1 for i,v in enumerate(xyz))) for label,sfac,xyz in ATOMS)
    arguments=inputs(tmp_path,candidate=structure(moved))
    arguments['policy']=replace(arguments['policy'],origin_selection='metal_anchors')
    result=compare_sealed_res(**arguments)
    assert len(result.matches)==len(ATOMS),result.reasons
    assert max(match.distance_angstrom for match in result.matches)<1e-7
    assert result.settings['full_site_assignments_evaluated']==1
    assert result.settings['comparison_policy']['origin_selection']=='metal_anchors'


def test_exhausted_metal_origin_search_does_not_report_unperformed_full_matching(tmp_path):
    optional_geometry()
    arguments=inputs(tmp_path)
    arguments['policy']=replace(arguments['policy'],origin_selection='metal_anchors',max_pair_evaluations=3)
    result=compare_sealed_res(**arguments)
    assert result.status=='PARTIAL' and result.origin_shift_fractional is None
    assert all(row.recovered_expert_sites is None for row in result.categories)
