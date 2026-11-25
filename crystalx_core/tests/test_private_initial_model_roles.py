"""Automatic seed roles and pins use invented files; no private archive or tool is read."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from crystalx_core.private_acceptance.grading import GradingRequest, GradingStatus, grade_sealed_run
from crystalx_core.private_acceptance.sealing import seal_run, verify_seal
from crystalx_core.private_acceptance.specs import (
    AcceptanceTrack, FileRole, SpecError, parse_grader_spec, parse_solver_spec,
)


@pytest.fixture
def seed_case(tmp_path):
    case = tmp_path / "synthetic owner originals"
    case.mkdir()
    instructions = "TITL Synthetic original\nCELL 0.71073 10 11 12 90 90 90\nLATT -1\nSFAC C O\nUNIT 2 2\nHKLF 4\nEND\n"
    (case / "original.ins").write_text(instructions)
    (case / "measured.hkl").write_text("   1   0   0  100.00    1.00\n")
    (case / "automatic.res").write_text(instructions.replace("HKLF 4", "FVAR 1.0\nC1 1 0.1 0.2 0.3 11.0 0.04\nHKLF 4"))
    (case / "held-out.res").write_text("Synthetic held-out answer trap, never a seed")
    return case


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recovery_json(seed_case):
    return {
        "schema_version": 1, "case_id": "synthetic owner identifier", "track": "INITIAL_MODEL_RECOVERY",
        "inputs": {"ins": "original.ins", "hkl": "measured.hkl"}, "priors": {},
        "preparation": {"policy": "preserve_input"},
        "seed": {"path": "automatic.res", "role": "AUTOMATIC_INITIAL_MODEL",
                 "sha256": digest(seed_case / "automatic.res"),
                 "compatible_hkl_sha256": digest(seed_case / "measured.hkl"),
                 "provenance": {"kind": "owner_confirmed_automatic_initial_model",
                                "source": "Synthetic automatic solver checkpoint provenance",
                                "owner_declaration": "Synthetic owner declaration: automatic initial output, not an expert final model"}},
    }


def protect_answer(monkeypatch, answer):
    original_open = __import__("builtins").open
    original_path_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert Path(path) != answer, "a held-out answer must not be read for seed selection"
        return original_open(path, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        assert path != answer, "a held-out answer must not be read for seed selection"
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)


def test_explicit_automatic_seed_has_four_role_boundary_and_input_pins(seed_case, recovery_json, monkeypatch):
    protect_answer(monkeypatch, seed_case / "held-out.res")
    spec = parse_solver_spec(recovery_json, case_root=seed_case)
    assert spec.track is AcceptanceTrack.INITIAL_MODEL_RECOVERY
    assert not spec.is_initial_solution_track
    assert not spec.blockers
    assert spec.seed.available
    assert spec.seed.sha256 == spec.seed.actual_sha256
    assert spec.seed.compatible_hkl_sha256 == recovery_json["seed"]["compatible_hkl_sha256"]
    assert spec.seed.role is FileRole.AUTOMATIC_INITIAL_MODEL
    assert spec.seed.compatibility_scientifically_verified is False
    assert [file.role for file in spec.input_roles] == [FileRole.INSTRUMENT_INSTRUCTIONS,
                                                     FileRole.MEASURED_REFLECTIONS,
                                                     FileRole.AUTOMATIC_INITIAL_MODEL]
    assert spec.ins_path.name == "original.ins"
    assert spec.seed.path.name == "automatic.res"
    assert FileRole.EXPERT_FINAL_ANSWER not in {file.role for file in spec.input_roles}
    with pytest.raises(SpecError, match="explicit iterative seed interface"):
        spec.to_solve_request()


def test_missing_seed_object_blocks_only_seed_track_without_discovering_answers(seed_case, recovery_json, monkeypatch):
    del recovery_json["seed"]
    protect_answer(monkeypatch, seed_case / "held-out.res")
    monkeypatch.setattr(Path, "glob", lambda *a, **k: pytest.fail("never discover a replacement seed"))
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: pytest.fail("never discover a replacement seed"))
    spec = parse_solver_spec(recovery_json, case_root=seed_case)
    assert spec.seed is None
    assert spec.blockers == ("automatic_initial_model_missing",)
    recovery_json["track"] = "DE_NOVO"
    baseline = parse_solver_spec(recovery_json, case_root=seed_case)
    assert baseline.seed is None
    assert baseline.blockers == ()
    assert baseline.to_solve_request().ins_path == seed_case / "original.ins"


def test_missing_declared_seed_preserves_path_does_not_use_expert_or_other_model(seed_case, recovery_json, monkeypatch):
    recovery_json["seed"]["path"] = "not-yet-supplied.res"
    protect_answer(monkeypatch, seed_case / "held-out.res")
    spec = parse_solver_spec(recovery_json, case_root=seed_case)
    assert spec.blockers == ("automatic_initial_model_missing",)
    assert spec.seed.path.name == "not-yet-supplied.res"
    assert not spec.seed.available
    assert spec.seed.actual_sha256 is None
    assert spec.seed.sha256 == recovery_json["seed"]["sha256"]


@pytest.mark.parametrize("track", ["DE_NOVO", "UNSOLVED", "COMPONENT_REGRESSION"])
def test_seed_never_silently_changes_another_track(seed_case, recovery_json, track, monkeypatch):
    recovery_json["track"] = track
    protect_answer(monkeypatch, seed_case / "held-out.res")
    with pytest.raises(SpecError, match="only INITIAL_MODEL_RECOVERY"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_de_novo_still_rejects_atom_bearing_model_without_seed(seed_case, recovery_json):
    recovery_json["track"] = "DE_NOVO"
    del recovery_json["seed"]
    (seed_case / "original.ins").write_text((seed_case / "automatic.res").read_text())
    with pytest.raises(SpecError, match="model atom records"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_recovery_instrument_input_cannot_be_relabelled_as_seed(seed_case, recovery_json):
    (seed_case / "original.ins").write_text((seed_case / "automatic.res").read_text())
    with pytest.raises(SpecError, match="model atom records"):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("role", ["EXPERT_FINAL_ANSWER", "INSTRUMENT_INSTRUCTIONS", "MEASURED_REFLECTIONS", "unknown", None])
def test_automatic_seed_role_must_be_explicit_and_cannot_be_expert_answer(seed_case, recovery_json, role, monkeypatch):
    recovery_json["seed"]["role"] = role
    recovery_json["seed"]["path"] = "held-out.res"
    protect_answer(monkeypatch, seed_case / "held-out.res")
    with pytest.raises(SpecError, match="AUTOMATIC_INITIAL_MODEL"):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("kind", ["expert_final_answer", "component_regression", "source_ins", "automatic", None])
def test_automatic_seed_provenance_requires_owner_confirmation(seed_case, recovery_json, kind):
    recovery_json["seed"]["provenance"]["kind"] = kind
    with pytest.raises(SpecError, match="owner-confirmed"):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("missing", ["role", "sha256", "compatible_hkl_sha256", "provenance"])
def test_incomplete_seed_metadata_rejected_rather_than_guessed(seed_case, recovery_json, missing):
    del recovery_json["seed"][missing]
    with pytest.raises(SpecError, match="missing fields"):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("missing", ["kind", "source", "owner_declaration"])
def test_complete_owner_seed_declaration_is_required(seed_case, recovery_json, missing):
    del recovery_json["seed"]["provenance"][missing]
    with pytest.raises(SpecError):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("field", ["source", "owner_declaration"])
def test_empty_owner_seed_declaration_is_not_confirmation(seed_case, recovery_json, field):
    recovery_json["seed"]["provenance"][field] = " "
    with pytest.raises(SpecError):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("where", ["seed", "provenance"])
def test_answer_metadata_cannot_be_embedded_in_automatic_seed(seed_case, recovery_json, where):
    target = recovery_json["seed"] if where == "seed" else recovery_json["seed"]["provenance"]
    target["expected_answer"] = "held-out.res"
    with pytest.raises(SpecError, match="grader-only"):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("field", ["sha256", "compatible_hkl_sha256"])
@pytest.mark.parametrize("value", ["", "f" * 63, "Z" * 64, 123, None])
def test_seed_and_reflection_identity_pins_are_required_sha256(seed_case, recovery_json, field, value):
    recovery_json["seed"][field] = value
    with pytest.raises(SpecError, match="SHA-256"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_digest_mismatch_rejected_without_fallback(seed_case, recovery_json):
    recovery_json["seed"]["sha256"] = "0" * 64
    with pytest.raises(SpecError, match="pinned identity"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_reflection_pin_must_match_actual_selected_hkl(seed_case, recovery_json):
    recovery_json["seed"]["compatible_hkl_sha256"] = "0" * 64
    with pytest.raises(SpecError, match="reflection identity"):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("path", ["../automatic.res", "/tmp/automatic.res", "C:\\automatic.res"])
def test_seed_path_cannot_escape_private_case(seed_case, recovery_json, path):
    recovery_json["seed"]["path"] = path
    with pytest.raises(SpecError):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_symlink_escape_rejected_before_external_read(seed_case, recovery_json, tmp_path, monkeypatch):
    outside = tmp_path / "outside-answer.res"
    outside.write_text("never read")
    (seed_case / "escape.res").symlink_to(outside)
    recovery_json["seed"]["path"] = "escape.res"
    protect_answer(monkeypatch, outside)
    with pytest.raises(SpecError, match="escapes"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_cannot_alias_instruction_role(seed_case, recovery_json):
    recovery_json["seed"]["path"] = "original.ins"
    with pytest.raises(SpecError, match="distinct declared file"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_instruction_hardlink_cannot_become_automatic_model(seed_case, recovery_json):
    (seed_case / "linked.res").hardlink_to(seed_case / "original.ins")
    recovery_json["seed"]["path"] = "linked.res"
    recovery_json["seed"]["sha256"] = digest(seed_case / "linked.res")
    with pytest.raises(SpecError, match="aliases"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_renamed_atomless_instructions_are_not_an_automatic_model(seed_case, recovery_json):
    (seed_case / "automatic.res").write_text((seed_case / "original.ins").read_text())
    recovery_json["seed"]["sha256"] = digest(seed_case / "automatic.res")
    with pytest.raises(SpecError, match="no model atoms"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_role_is_not_inferred_from_extension(seed_case, recovery_json):
    (seed_case / "automatic-output.ins").write_text((seed_case / "automatic.res").read_text())
    recovery_json["seed"]["path"] = "automatic-output.ins"
    spec = parse_solver_spec(recovery_json, case_root=seed_case)
    assert spec.seed.role is FileRole.AUTOMATIC_INITIAL_MODEL
    assert spec.input_roles[0].role is FileRole.INSTRUMENT_INSTRUCTIONS


@pytest.mark.parametrize("remove", ["LATT -1\n", "HKLF 4\n", "CELL 0.71073 10 11 12 90 90 90\n"])
def test_seed_must_preserve_explicit_source_setting(seed_case, recovery_json, remove):
    model = seed_case / "automatic.res"
    model.write_text(model.read_text().replace(remove, ""))
    recovery_json["seed"]["sha256"] = digest(model)
    with pytest.raises(SpecError, match="source CELL, LATT and HKLF"):
        parse_solver_spec(recovery_json, case_root=seed_case)


@pytest.mark.parametrize("replace", [("C1 1", "C1 99"), ("C1 1 0.1", "C1 1 nan")])
def test_seed_bad_atom_data_is_blocked(seed_case, recovery_json, replace):
    model = seed_case / "automatic.res"
    model.write_text(model.read_text().replace(*replace))
    recovery_json["seed"]["sha256"] = digest(model)
    with pytest.raises(SpecError):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_external_include_forbidden(seed_case, recovery_json):
    model = seed_case / "automatic.res"
    model.write_text("+held-out.res\n" + model.read_text())
    recovery_json["seed"]["sha256"] = digest(model)
    with pytest.raises(SpecError, match="file-inclusion"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_nonfinite_source_setting_is_not_replayable(seed_case, recovery_json):
    model = seed_case / "automatic.res"
    model.write_text(model.read_text().replace("CELL 0.71073 10", "CELL 0.71073 nan"))
    recovery_json["seed"]["sha256"] = digest(model)
    with pytest.raises(SpecError, match="nonfinite source CELL"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_seed_track_refuses_unreviewed_index_reassignment(seed_case, recovery_json):
    recovery_json["preparation"] = {"policy": "explicit_reviewed_composition", "composition": [{"element": "O", "count": 4}],
                                     "owner_review_reference": "Synthetic", "official_semantics_reference": "Synthetic"}
    with pytest.raises(SpecError, match="preserve supplied model SFAC"):
        parse_solver_spec(recovery_json, case_root=seed_case)


def test_component_role_stays_expert_and_is_not_automatic_recovery(seed_case, recovery_json):
    # This is a separate, explicitly labelled synthetic expert component input. The seed object
    # is not converted into that role or passed to the component route.
    (seed_case / "expert-component.res").write_text((seed_case / "automatic.res").read_text())
    recovery_json["track"] = "COMPONENT_REGRESSION"
    recovery_json["inputs"]["ins"] = "expert-component.res"
    del recovery_json["seed"]
    spec = parse_solver_spec(recovery_json, case_root=seed_case)
    assert spec.input_roles[0].role is FileRole.EXPERT_FINAL_ANSWER
    assert spec.seed is None
    assert not spec.is_initial_solution_track


def test_recovery_grader_has_no_default_ground_truth(seed_case):
    value = {"schema_version": 1, "case_id": "synthetic", "track": "INITIAL_MODEL_RECOVERY"}
    config = parse_grader_spec(value, case_root=seed_case)
    assert config.answer_path is None
    assert config.answer_role is None
    value["answer"] = "held-out.res"
    config = parse_grader_spec(value, case_root=seed_case)
    assert config.answer_role is FileRole.EXPERT_FINAL_ANSWER


def test_recovery_seals_normalize_track_and_missing_grading_configuration_keeps_answer_closed(tmp_path, seed_case, monkeypatch):
    private = tmp_path.resolve() / "private-runs"
    run = private / "synthetic-recovery"
    run.mkdir(parents=True)
    (run / "model.res").write_text("synthetic produced model")
    seal = seal_run(run, private_root=private, outputs={"model": "model.res"},
                    case_id="synthetic", track="INITIAL_MODEL_RECOVERY")
    assert seal.track == "initial_model_recovery"
    assert verify_seal(run, private_root=private, expected_seal_sha256=seal.seal_sha256).track == seal.track
    request = GradingRequest("synthetic", "INITIAL_MODEL_RECOVERY", seed_case, seed_case / "held-out.res")
    protect_answer(monkeypatch, seed_case / "held-out.res")
    not_pinned = grade_sealed_run(run, private_root=private, specification=request)
    assert not_pinned.status is GradingStatus.NOT_EVALUATED
    assert not_pinned.answer_sha256 is None
    no_provider = grade_sealed_run(run, private_root=private, specification=request, expected_seal_sha256=seal.seal_sha256)
    assert no_provider.status is GradingStatus.NOT_EVALUATED
    assert no_provider.answer_sha256 is None
    assert not no_provider.scientifically_reviewed
    wrong_track = GradingRequest("synthetic", "DE_NOVO", seed_case, seed_case / "held-out.res")
    blocked = grade_sealed_run(run, private_root=private, specification=wrong_track, expected_seal_sha256=seal.seal_sha256)
    assert blocked.status is GradingStatus.BLOCKED
    unsealed = private / "unsealed"
    unsealed.mkdir()
    report = grade_sealed_run(unsealed, private_root=private, specification=request, expected_seal_sha256=seal.seal_sha256)
    assert report.status is GradingStatus.BLOCKED
