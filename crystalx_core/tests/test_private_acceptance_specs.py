"""Only generated, synthetic private-case contracts are used here."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from crystalx_core.app.models import RunMode
from crystalx_core.app.staging import apply_composition_override
from crystalx_core.private_acceptance.specs import (
    AcceptanceTrack, SpecError, contained_path, load_grader_spec, load_solver_spec,
    parse_grader_spec, parse_solver_spec, synthetic_solver_template,
)


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "User original directory"
    root.mkdir()
    (root / "Original input.ins").write_text(
        "TITL Synthetic input\nCELL 0.71073 10 10 10 90 90 90\n"
        "LATT -1\nSFAC C O\nUNIT 2 2\nHKLF 4\nEND\n", encoding="utf-8",
    )
    (root / "Original intensities.hkl").write_text("   1   0   0  100.00    1.00\n", encoding="utf-8")
    return root


@pytest.fixture
def solver_json():
    return {"schema_version": 1, "case_id": "local Case α", "track": "DE_NOVO",
            "inputs": {"ins": "Original input.ins", "hkl": "Original intensities.hkl"},
            "priors": {}, "preparation": {"policy": "preserve_input"}}


def declaration(value, provenance="owner_confirmed"):
    return {"value": value, "provenance": provenance, "source": "Synthetic owner statement"}


@pytest.mark.parametrize("track", [AcceptanceTrack.DE_NOVO, AcceptanceTrack.COMPONENT_REGRESSION,
                                  AcceptanceTrack.UNSOLVED])
def test_generic_case_identifier_and_original_names_survive(case, solver_json, track):
    solver_json["track"] = track.value
    spec = parse_solver_spec(solver_json, case_root=case)
    assert spec.case_id == "local Case α"
    assert spec.ins_path.name == "Original input.ins"
    assert spec.hkl_path.name == "Original intensities.hkl"
    request = spec.to_solve_request()
    assert request.mode is RunMode.REAL
    assert request.config.strict_native
    assert request.ins_path == spec.ins_path
    assert request.hkl_path == spec.hkl_path
    assert request.label == spec.case_id
    assert spec.is_initial_solution_track == (track is not AcceptanceTrack.COMPONENT_REGRESSION)


def test_loader_reads_only_declared_solver_spec_and_instruction(case, solver_json, monkeypatch):
    path = case / "solver-input.json"
    path.write_text(json.dumps(solver_json), encoding="utf-8")
    answer = case / "held-out.res"
    answer.write_text("DO NOT READ", encoding="utf-8")
    original_open = __import__("builtins").open
    original_read_text = Path.read_text

    def guarded_open(file, *args, **kwargs):
        assert Path(file) != answer
        return original_open(file, *args, **kwargs)

    def guarded_read_text(file, *args, **kwargs):
        assert file != answer
        return original_read_text(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    spec = load_solver_spec(path)
    assert spec.to_solve_request().ins_path.name == "Original input.ins"
    assert not hasattr(spec, "answer_path")


@pytest.mark.parametrize("location", ["root", "inputs", "priors", "element", "preparation", "composition"])
@pytest.mark.parametrize("key", ["expected_answer", "known_chemistry"])
def test_answer_fields_rejected_at_every_input_depth(case, solver_json, location, key):
    if location == "root":
        solver_json[key] = {"path": "held-out.res"}
    elif location == "inputs":
        solver_json["inputs"][key] = "held-out.res"
    elif location == "priors":
        solver_json["priors"][key] = {"value": "C"}
    elif location == "element":
        item = declaration("C")
        item[key] = "held-out"
        solver_json["priors"]["elements"] = [item]
    elif location == "preparation":
        solver_json["preparation"][key] = "held-out"
    else:
        solver_json["preparation"]["composition"] = [{"element": "C", "count": 2, key: 5}]
    with pytest.raises(SpecError, match="grader-only|archive"):
        parse_solver_spec(solver_json, case_root=case)


@pytest.mark.parametrize("extra", ["notes", "expected_atom_counts", "grader", "config", "outputs", "references"])
def test_overfull_archive_is_not_a_solver_spec(case, solver_json, extra):
    solver_json[extra] = {}
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


@pytest.mark.parametrize("relative", ["../outside.ins", "/tmp/outside.ins", "folder/../../outside.ins",
                                     "folder/../Original input.ins", "./Original input.ins",
                                     "C:\\outside.ins", "C:outside.ins", "\\\\server\\file.ins",
                                     "Original\x00input.ins", "Original\ninput.ins"])
def test_paths_reject_traversal_absolute_and_alternate_separators(case, relative):
    with pytest.raises(SpecError):
        contained_path(case, relative)


def test_paths_reject_symlink_escape_even_when_final_file_missing(case, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "input.ins").write_text("synthetic", encoding="utf-8")
    (case / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SpecError, match="escapes"):
        contained_path(case, "escape/input.ins")
    with pytest.raises(SpecError, match="escapes"):
        contained_path(case, "escape/missing.ins", must_exist=False)


@pytest.mark.parametrize("relative", ["missing.ins", "folder"])
def test_missing_or_directory_inputs_rejected(case, relative):
    (case / "folder").mkdir()
    with pytest.raises(SpecError):
        contained_path(case, relative)


def test_filename_is_never_globbed(case, solver_json):
    solver_json["inputs"]["ins"] = "*.ins"
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


def test_owner_and_source_prior_provenance_are_retained(case, solver_json):
    solver_json["priors"] = {"elements": [declaration("C", "source_ins"), declaration("Cu")],
                             "metals": [declaration("Cu")], "peptide": declaration(True)}
    spec = parse_solver_spec(solver_json, case_root=case)
    assert spec.prior.elements == ("C", "Cu")
    assert spec.prior.metals == ("Cu",)
    assert spec.prior.peptide
    assert spec.prior_records[0].provenance == "source_ins"
    assert not spec.prior.composition  # Declared labels do not pretend to change SFAC/UNIT.
    assert spec.to_solve_request().prior == spec.prior


@pytest.mark.parametrize("provenance", ["expert_model", "expert_derived", "reference_structure", "unknown", None])
def test_expert_derived_and_unsourced_prior_refused(case, solver_json, provenance):
    solver_json["priors"]["elements"] = [declaration("C", provenance)]
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


def test_source_ins_prior_must_really_exist_in_sfac(case, solver_json):
    solver_json["priors"]["elements"] = [declaration("Cu", "source_ins")]
    with pytest.raises(SpecError, match="absent"):
        parse_solver_spec(solver_json, case_root=case)


@pytest.mark.parametrize("field", ["peptide", "racemic"])
def test_boolean_science_prior_needs_independent_owner_confirmation(case, solver_json, field):
    solver_json["priors"][field] = declaration(True, "source_ins")
    with pytest.raises(SpecError, match="independent owner"):
        parse_solver_spec(solver_json, case_root=case)


@pytest.mark.parametrize("prior", [
    {"elements": [declaration("cu")]},
    {"elements": [declaration("C"), declaration("C")]},
    {"metals": "Cu"},
    {"racemic": declaration("false")},
    {"peptide": {"value": True, "provenance": "owner_confirmed"}},
    {"elements": [dict(declaration("C"), extra="hidden answer")]},
])
def test_malformed_or_ambiguous_priors_fail(case, solver_json, prior):
    solver_json["priors"] = prior
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


def reviewed_preparation():
    return {"policy": "explicit_reviewed_composition", "composition": [
        {"element": "C", "count": 4}, {"element": "Cu", "count": 2},
    ], "owner_review_reference": "Synthetic full cell composition approval",
        "official_semantics_reference": "Synthetic review of official SFAC/UNIT semantics"}


def test_explicit_full_composition_reaches_staging_without_changing_original(case, solver_json):
    original = case / "Original input.ins"
    before = original.read_bytes()
    solver_json["preparation"] = reviewed_preparation()
    spec = parse_solver_spec(solver_json, case_root=case)
    request = spec.to_solve_request()
    derived, changes = apply_composition_override(original.read_text(), request.prior.composition)
    assert "SFAC C Cu\nUNIT 4 2\n" in derived
    assert "LATT -1" in derived
    assert changes
    assert original.read_bytes() == before
    assert spec.preparation.owner_review_reference
    assert spec.preparation.official_semantics_reference


@pytest.mark.parametrize("count", [0, -1, True, "unknown", None, float("inf"), float("nan")])
def test_unknown_nonfinite_or_invented_composition_counts_refused(case, solver_json, count):
    preparation = reviewed_preparation()
    preparation["composition"][0]["count"] = count
    solver_json["preparation"] = preparation
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


@pytest.mark.parametrize("missing", ["owner_review_reference", "official_semantics_reference", "composition"])
def test_composition_policy_requires_both_reviews_and_full_values(case, solver_json, missing):
    preparation = reviewed_preparation()
    del preparation[missing]
    solver_json["preparation"] = preparation
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


def test_preserve_input_refuses_smuggled_composition(case, solver_json):
    solver_json["preparation"]["composition"] = [{"element": "Cu", "count": 2}]
    with pytest.raises(SpecError, match="must not include"):
        parse_solver_spec(solver_json, case_root=case)


@pytest.mark.parametrize("track", ["DE_NOVO", "UNSOLVED"])
def test_blind_track_rejects_model_atoms_even_with_ins_extension(case, solver_json, track):
    solver_json["track"] = track
    path = case / "Original input.ins"
    path.write_text(path.read_text().replace("HKLF", "C1 1 0.1 0.2 0.3 11.0 0.05\nHKLF"))
    with pytest.raises(SpecError, match="model atom records"):
        parse_solver_spec(solver_json, case_root=case)


def test_component_track_explicitly_allows_model_but_refuses_composition_reindexing(case, solver_json):
    model = case / "expert-copy.res"
    model.write_text((case / "Original input.ins").read_text().replace("HKLF", "C1 1 0.1 0.2 0.3 11.0 0.05\nHKLF"))
    solver_json["track"] = "COMPONENT_REGRESSION"
    solver_json["inputs"]["ins"] = model.name
    spec = parse_solver_spec(solver_json, case_root=case)
    assert not spec.is_initial_solution_track
    solver_json["preparation"] = reviewed_preparation()
    with pytest.raises(SpecError, match="preserve"):
        parse_solver_spec(solver_json, case_root=case)


@pytest.mark.parametrize("directive", ["+../held-out.res", "++held-out.res"])
def test_instruction_file_cannot_bypass_two_file_boundary_with_include(case, solver_json, directive):
    path = case / "Original input.ins"
    path.write_text(directive + "\n" + path.read_text(), encoding="utf-8")
    with pytest.raises(SpecError, match="file-inclusion"):
        parse_solver_spec(solver_json, case_root=case)


def test_bad_instruction_parse_is_a_safe_spec_error(case, solver_json):
    path = case / "Original input.ins"
    path.write_text("CELL definitely not a cell\n", encoding="utf-8")
    with pytest.raises(SpecError, match="could not be checked"):
        parse_solver_spec(solver_json, case_root=case)


def test_nonscalar_provenance_is_rejected_as_spec_error(case, solver_json):
    solver_json["priors"] = {"elements": [declaration("C", {"expert": True})]}
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


def test_separate_grader_loading_never_reads_answer_content(case, monkeypatch):
    answer = case / "held-out.res"
    answer.write_text("unread synthetic model", encoding="utf-8")
    grader = {"schema_version": 1, "case_id": "local Case α", "track": "DE_NOVO", "answer": answer.name}
    config = case / "grader-only.json"
    config.write_text(json.dumps(grader), encoding="utf-8")
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert path != answer
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    loaded = load_grader_spec(config)
    assert loaded.answer_path == answer
    assert loaded.site_match_tolerance_angstrom is None
    assert not loaded.thresholds_owner_approved


def test_unsolved_has_no_answer_but_may_label_advisory_reference(case):
    reference = case / "advisory-only.cif"
    reference.write_text("synthetic", encoding="utf-8")
    data = {"schema_version": 1, "case_id": "unsolved_local", "track": "UNSOLVED", "answer": reference.name}
    with pytest.raises(SpecError, match="no ground truth"):
        parse_grader_spec(data, case_root=case)
    del data["answer"]
    data["references"] = [reference.name]
    spec = parse_grader_spec(data, case_root=case)
    assert spec.answer_path is None
    assert spec.reference_paths == (reference,)


@pytest.mark.parametrize("thresholds", [
    {"owner_approved": True},
    {"owner_approved": True, "approval_reference": "Owner review"},
    {"owner_approved": True, "site_match_tolerance_angstrom": 0.2},
    {"owner_approved": "true"},
    {"heavy_atomic_number_min": True},
    {"heavy_atomic_number_min": 119},
    {"site_match_tolerance_angstrom": 0},
    {"expected_answer": 2},
])
def test_no_default_or_malformed_scientific_threshold_approval(case, thresholds):
    data = {"schema_version": 1, "case_id": "synthetic", "track": "DE_NOVO", "thresholds": thresholds}
    with pytest.raises(SpecError):
        parse_grader_spec(data, case_root=case)


def test_explicit_owner_threshold_evidence_retained(case):
    data = {"schema_version": 1, "case_id": "synthetic", "track": "DE_NOVO", "thresholds": {
        "owner_approved": True, "approval_reference": "Synthetic owner approval",
        "site_match_tolerance_angstrom": 0.2, "heavy_atomic_number_min": 21,
    }}
    spec = parse_grader_spec(data, case_root=case)
    assert spec.thresholds_owner_approved
    assert spec.thresholds_approval_reference == "Synthetic owner approval"
    assert spec.heavy_atomic_number_min == 21


@pytest.mark.parametrize("document", [
    '{"schema_version":1,"schema_version":1}',
    '{"schema_version":1,"priors":{"elements":[],"elements":[]}}',
    '{"schema_version":NaN}',
    '{"schema_version":Infinity}',
    '[]',
    '{broken}',
])
def test_document_loader_rejects_duplicate_fields_and_nonjson_numbers(case, document):
    path = case / "solver-input.json"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(SpecError):
        load_solver_spec(path)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("schema_version", 2),
                                         ("case_id", "../case"), ("track", "solved")])
def test_invalid_version_id_and_track(case, solver_json, field, value):
    solver_json[field] = value
    with pytest.raises(SpecError):
        parse_solver_spec(solver_json, case_root=case)


def test_recursive_nonjson_python_input_is_a_spec_error(case, solver_json):
    solver_json["priors"]["recursive"] = solver_json
    with pytest.raises(SpecError, match="nesting"):
        parse_solver_spec(solver_json, case_root=case)


def test_document_loader_enforces_size_limit(case):
    path = case / "overfull-input.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(SpecError, match="document limit"):
        load_solver_spec(path)


def test_public_template_is_fictional_and_has_no_answer_or_research_metadata():
    example = synthetic_solver_template()
    assert example["case_id"] == "synthetic_example"
    assert set(example["inputs"]) == {"ins", "hkl"}
    assert example["priors"] == {}
    assert "answer" not in json.dumps(example)
    changed = copy.deepcopy(example)
    changed["inputs"]["ins"] = "another.ins"
    assert synthetic_solver_template() == example
