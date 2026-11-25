"""Read-only receipts for an automatic SHELXT checkpoint, not full-run acceptance.

The caller names one retained branch and invocation explicitly. No answer, reference, arbitrary
model path or file discovery interface is accepted. A receipt pins original retained inputs,
the inputs SHELXT actually recorded reading, and its new model/matching-reflection outputs.
Only matching HKL outputs are supported: the original-orientation REM dialect remains unverified.

Receipts are frozen integrity records, not signatures or launch authority. A service may use the
verified paths only after staging fresh copies and rechecking the returned hashes. This module
never writes, launches a program, changes security settings or relaxes ``verify_run``. It does
not re-hash installed executables or establish scientific correctness/complete workflow success.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from crystalx_core.app.reflections import ReflectionSource, solution_model_name, solution_reflections_name
from crystalx_core.app.staging import apply_composition_override
from crystalx_core.private_acceptance.sealing import SealError, _directory, _read
from crystalx_core.private_acceptance.specs import FileRole
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins

__all__ = ["HandoffError", "ArtifactPin", "InitialModelReceipt", "VerifiedInitialModelHandoff",
           "create_initial_model_receipt", "verify_initial_model_receipt"]

_JSON_LIMIT = 4 * 1024 * 1024
_TEXT_LIMIT = 64 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_BRANCH = re.compile(r"stages/[A-Za-z0-9_-]+/branches/[A-Za-z0-9_-]+-shelxt\Z")
_TERMINAL = {"completed", "failed", "cancelled", "blocked"}


class HandoffError(ValueError):
    """The explicit checkpoint cannot be safely retained or independently rechecked."""


def _require(test: bool, message: str) -> None:
    if not test:
        raise HandoffError(message)


def _relative(value: Any) -> str:
    _require(isinstance(value, str) and bool(value) and len(value) <= 4096,
             "evidence path must be a bounded relative path")
    _require(not PurePosixPath(value).is_absolute() and "\\" not in value and ":" not in value
             and not any(ord(c) < 32 or ord(c) == 127 for c in value)
             and not any(part in {"", ".", ".."} for part in value.split("/")),
             "evidence path traversal or alternate separators are forbidden")
    return value


def _sha(value: Any) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None,
             "evidence requires a canonical SHA-256 digest")
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ArtifactPin:
    role: str
    relative_path: str
    sha256: str
    size_bytes: int

    def to_record(self) -> dict[str, Any]:
        return {"role": self.role, "relative_path": self.relative_path,
                "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class InitialModelReceipt:
    run_root: Path
    run_id: str
    branch: str
    provenance: str
    stem: str
    solution_letter: str
    require_real: bool
    synthetic: bool
    source_duration_s: float
    original_instructions: ArtifactPin
    measured_reflections: ArtifactPin
    shelxt_instructions: ArtifactPin
    shelxt_reflections: ArtifactPin
    automatic_model: ArtifactPin
    paired_reflections: ArtifactPin
    records: tuple[ArtifactPin, ...]
    executable_identity: tuple[tuple[str, str], ...]

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "crystalx.automatic_initial_model_receipt/1", "run_root": str(self.run_root),
            "run_id": self.run_id, "branch": self.branch, "provenance": self.provenance,
            "stem": self.stem, "solution_letter": self.solution_letter,
            "require_real": self.require_real, "synthetic": self.synthetic,
            "source_tool_calls": 1, "source_duration_s": self.source_duration_s,
            "source_cost_scope": "selected_shelxt_invocation_only",
            "full_source_run_cost_included": False,
            "original_instructions": self.original_instructions.to_record(),
            "measured_reflections": self.measured_reflections.to_record(),
            "shelxt_instructions": self.shelxt_instructions.to_record(),
            "shelxt_reflections": self.shelxt_reflections.to_record(),
            "automatic_model": self.automatic_model.to_record(),
            "paired_reflections": self.paired_reflections.to_record(),
            "records": [pin.to_record() for pin in self.records],
            "executable_identity": dict(self.executable_identity),
            "reflection_source": ReflectionSource.SHELXT_MATCHING.value,
            "real_dispatch_allowed": False, "full_workflow_verified": False,
            "scientifically_reviewed": False,
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(_canonical(self._payload())).hexdigest()

    def to_record(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> InitialModelReceipt:
        """Restore the exact receipt schema; metadata never discovers additional inputs."""
        keys = {"schema", "run_root", "run_id", "branch", "provenance", "stem", "solution_letter",
                "require_real", "synthetic", "original_instructions", "measured_reflections",
                "source_tool_calls", "source_duration_s", "source_cost_scope", "full_source_run_cost_included",
                "shelxt_instructions", "shelxt_reflections", "automatic_model", "paired_reflections",
                "records", "executable_identity", "reflection_source", "real_dispatch_allowed",
                "full_workflow_verified", "scientifically_reviewed", "receipt_sha256"}
        _require(isinstance(value, Mapping) and set(value) == keys, "unsupported receipt fields")
        _require(value["schema"] == "crystalx.automatic_initial_model_receipt/1"
                 and value["reflection_source"] == ReflectionSource.SHELXT_MATCHING.value,
                 "unsupported receipt/reflection contract")
        _require(all(value[field] is False for field in
                     ("real_dispatch_allowed", "full_workflow_verified", "scientifically_reviewed")),
                 "receipt cannot grant dispatch, full acceptance or scientific review")
        _require(type(value["require_real"]) is bool and type(value["synthetic"]) is bool,
                 "receipt program mode must be explicit")
        _require(isinstance(value["run_root"], str) and Path(value["run_root"]).is_absolute()
                 and isinstance(value["run_id"], str) and bool(value["run_id"]),
                 "receipt requires its absolute source root and source identity")
        _require(_BRANCH.fullmatch(_relative(value["branch"])) is not None
                 and isinstance(value["stem"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value["stem"]) is not None
                 and isinstance(value["solution_letter"], str) and re.fullmatch(r"[a-z]", value["solution_letter"]) is not None,
                 "receipt selected branch/stem/solution is invalid")
        _relative(value["provenance"])
        _require(type(value["source_tool_calls"]) is int and value["source_tool_calls"] == 1
                 and value["source_cost_scope"] == "selected_shelxt_invocation_only"
                 and value["full_source_run_cost_included"] is False,
                 "receipt cost covers only the selected SHELXT invocation")
        source_duration = _duration(value["source_duration_s"])

        def pin(data: Any) -> ArtifactPin:
            _require(isinstance(data, dict) and set(data) == {"role", "relative_path", "sha256", "size_bytes"},
                     "unsupported artifact pin")
            _require(isinstance(data["role"], str) and type(data["size_bytes"]) is int and data["size_bytes"] >= 0,
                     "invalid artifact pin fields")
            return ArtifactPin(data["role"], _relative(data["relative_path"]), _sha(data["sha256"]), data["size_bytes"])

        _require(isinstance(value["records"], list) and len(value["records"]) == 6,
                 "receipt requires its six explicit source records")
        identity = value["executable_identity"]
        _require(isinstance(identity, dict) and set(identity) == {"path", "sha256", "arch", "source"}
                 and all(isinstance(item, str) for item in identity.values()), "invalid executable identity")
        _sha(identity["sha256"])
        receipt = cls(Path(value["run_root"]), value["run_id"], value["branch"], value["provenance"],
                      value["stem"], value["solution_letter"], value["require_real"], value["synthetic"],
                      source_duration,
                      *(pin(value[name]) for name in ("original_instructions", "measured_reflections",
                        "shelxt_instructions", "shelxt_reflections", "automatic_model", "paired_reflections")),
                      tuple(pin(item) for item in value["records"]), tuple(sorted(identity.items())))
        _require(receipt.receipt_sha256 == _sha(value["receipt_sha256"]), "receipt digest does not match its fields")
        return receipt


@dataclass(frozen=True)
class VerifiedInitialModelHandoff:
    source_run_id: str
    receipt_sha256: str
    model_path: Path
    paired_hkl_path: Path
    instrument_ins_path: Path
    measured_hkl_path: Path
    model_sha256: str
    paired_hkl_sha256: str
    measured_hkl_sha256: str
    source_duration_s: float
    synthetic: bool
    source_tool_calls: int = 1
    source_cost_scope: str = "selected_shelxt_invocation_only"
    full_source_run_cost_included: bool = False
    role: FileRole = FileRole.AUTOMATIC_INITIAL_MODEL
    real_dispatch_allowed: bool = False
    full_workflow_verified: bool = False
    scientifically_reviewed: bool = False


class _EvidenceReader:
    def __init__(self, root: Path, directory_fd: int):
        self.root, self.fd = root, directory_fd
        self.pins: list[ArtifactPin] = []

    def read(self, relative: str, role: str, *, collect: bool = False,
             max_bytes: int | None = None) -> tuple[ArtifactPin, bytes]:
        relative = _relative(relative)
        sha, size, data = _read(self.fd, relative, max_bytes=max_bytes, collect=collect)
        pin = ArtifactPin(role, relative, sha, size)
        self.pins.append(pin)
        return pin, data

    def json(self, relative: str, role: str) -> tuple[ArtifactPin, dict[str, Any]]:
        pin, data = self.read(relative, role, collect=True, max_bytes=_JSON_LIMIT)

        def pairs(items):
            result = {}
            for key, val in items:
                _require(key not in result, "duplicate source JSON field")
                result[key] = val
            return result

        def constant(_token):
            raise HandoffError("nonfinite source JSON number")

        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
        _require(isinstance(value, dict), "source record must be a JSON object")
        return pin, value

    def recheck(self) -> None:
        for pin in self.pins:
            sha, size, _ = _read(self.fd, pin.relative_path)
            _require(sha == pin.sha256 and size == pin.size_bytes, "source evidence changed while receipt was built")


def _entries(record: Mapping[str, Any], key: str, required_role: str) -> dict[str, dict[str, Any]]:
    values = record.get(key)
    _require(isinstance(values, list), "program input/output provenance is missing")
    result = {}
    for item in values:
        _require(isinstance(item, dict), "invalid program artifact record")
        relative = _relative(item.get("relative_path"))
        _require(relative not in result, "duplicate program artifact name")
        if required_role == "input":
            _require(item.get("role") == "input", "program input has a different declared role")
        result[relative] = item
    return result


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="strict")


def _atom_count(text: str, *, include_peaks: bool = True) -> int:
    _require(not any(line.split() and line.split()[0].startswith("+") for line in text.splitlines()),
             "external file includes are forbidden in retained checkpoint instructions")
    # Reuse the existing text-based atom grammar without writing or applying the returned edits.
    _, _, count, peaks = prepare_refinement_ins(text, RefinementSettings())
    return count + peaks if include_peaks else count


def _duration(value: Any) -> float:
    _require(type(value) in {int, float},
             "selected invocation duration must be finite and nonnegative")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise HandoffError("selected invocation duration must be finite and nonnegative") from exc
    _require(math.isfinite(result) and result >= 0, "selected invocation duration must be finite and nonnegative")
    return result


def _create(run_root: Path, *, branch: str, provenance: str, solution_letter: str,
            expected_inputs: Mapping[str, str], require_real: bool) -> InitialModelReceipt:
    _require(type(require_real) is bool, "require_real must be an explicit boolean")
    root = Path(run_root).absolute()
    branch = _relative(branch)
    provenance = _relative(provenance)
    _require(_BRANCH.fullmatch(branch) is not None, "an explicit SHELXT branch is required")
    _require(PurePosixPath(provenance).parent.as_posix() == branch
             and re.fullmatch(r"provenance_[0-9]+\.json", PurePosixPath(provenance).name) is not None,
             "invocation provenance must be explicitly inside the chosen SHELXT branch")
    _require(isinstance(solution_letter, str) and re.fullmatch(r"[a-z]", solution_letter) is not None,
             "an explicit lowercase SHELXT solution letter is required")
    _require(isinstance(expected_inputs, Mapping) and set(expected_inputs) == {"ins", "hkl"},
             "both expected original INS and measured HKL identities are required")
    expected = {role: _sha(value) for role, value in expected_inputs.items()}
    with _directory(root) as fd:
        reader = _EvidenceReader(root, fd)
        state_pin, state = reader.json("state.json", "RUN_STATE")
        summary_pin, summary = reader.json("summary.json", "RUN_SUMMARY")
        branch_pin, branch_record = reader.json(branch + "/branch.json", "SHELXT_BRANCH")
        outcome_pin, outcome = reader.json(branch + "/outcome.json", "SHELXT_OUTCOME")
        staging_pin, staging = reader.json(branch + "/staging.json", "SHELXT_STAGING")
        provenance_pin, invocation = reader.json(provenance, "SHELXT_INVOCATION")
        records = (state_pin, summary_pin, branch_pin, outcome_pin, staging_pin, provenance_pin)
        _require(state.get("phase") in _TERMINAL and (summary.get("process") or {}).get("phase") == state["phase"],
                 "source run must be terminal; this receipt does not accept live or interrupted work")
        _require(isinstance(state.get("run_id"), str) and state["run_id"] == summary.get("run_id"),
                 "source run identity disagrees")
        stages = state.get("stages")
        _require(isinstance(stages, list), "source stage state is missing")
        selected_stages = [item for item in stages if isinstance(item, dict) and item.get("name") == "shelxt"]
        _require(len(selected_stages) == 1 and selected_stages[0].get("state") == "succeeded",
                 "SHELXT stage did not succeed")
        _require(branch_record.get("kind") == "shelxt" and branch_record.get("state") == "succeeded"
                 and outcome.get("state") == "succeeded" and branch_record.get("branch_id") == outcome.get("branch_id")
                 and branch_record.get("branch_id") == PurePosixPath(branch).name
                 and branch_record.get("stage_id") == PurePosixPath(branch).parts[1],
                 "selected SHELXT branch/outcome did not succeed with this identity")
        _require(summary.get("acceptance_track") != "COMPONENT_REGRESSION"
                 and not summary.get("component_regression"), "expert component results are never automatic initial checkpoints")
        _require(invocation.get("tool") == "shelxt" and invocation.get("succeeded") is True
                 and type(invocation.get("exit_code")) is int and invocation["exit_code"] == 0
                 and invocation.get("cancelled") is False and not invocation.get("missing_outputs")
                 and invocation.get("diagnostic") in {"ok", "ok_via_translation"},
                 "successful SHELXT invocation evidence is missing")
        source_duration = _duration(invocation.get("duration_s"))
        synthetic = (summary.get("mode") == "mock_demo" or summary.get("mock") is True
                     or invocation.get("executable_source") == "synthetic_demo")
        _require(not require_real or (summary.get("mode") == "real" and summary.get("mock") is False and not synthetic),
                 "synthetic checkpoint is not real-process evidence")
        identity = {"path": invocation.get("executable_path"), "sha256": invocation.get("executable_sha256"),
                    "arch": invocation.get("executable_arch"), "source": invocation.get("executable_source")}
        _require(all(isinstance(value, str) and value for value in identity.values()), "executable identity is not recorded")
        _sha(identity["sha256"])
        _require(not require_real or Path(identity["path"]).is_absolute(), "real executable identity must have an absolute path")
        stem = summary.get("stem")
        _require(isinstance(stem, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", stem) is not None,
                 "source stem is invalid")
        _require(invocation.get("argv") == [identity["path"], stem]
                 and invocation.get("cwd") == str(root / branch), "SHELXT invocation does not address the chosen branch/stem")
        solution = summary.get("solution") or {}
        _require(solution.get("work_dir") == branch and solution.get("branch") == PurePosixPath(branch).name,
                 "source summary selected a different branch")
        model_name = solution_model_name(stem, solution_letter)
        hkl_name = solution_reflections_name(stem, solution_letter)
        choices = [item for item in solution.get("solutions", []) if isinstance(item, dict) and item.get("letter") == solution_letter]
        _require(len(choices) == 1 and choices[0].get("file") == f"{branch}/{model_name}",
                 "source summary did not select this generated solution file")
        source_inputs = summary.get("inputs")
        _require(isinstance(source_inputs, dict) and set(source_inputs) == {"ins", "hkl"},
                 "source must contain exactly instrument instructions and measured reflections")
        originals, original_text = {}, b""
        for role, file_role in (("ins", FileRole.INSTRUMENT_INSTRUCTIONS), ("hkl", FileRole.MEASURED_REFLECTIONS)):
            item = source_inputs[role]
            _require(isinstance(item, dict) and item.get("sha256") == expected[role], "original retained input identity differs from expected input")
            pin, data = reader.read(item.get("staged"), file_role.value, collect=role == "ins",
                                    max_bytes=_TEXT_LIMIT if role == "ins" else None)
            _require(pin.sha256 == expected[role], "retained original input changed")
            originals[role] = pin
            if role == "ins":
                original_text = data
        _require(originals["ins"].relative_path != originals["hkl"].relative_path,
                 "different original file roles cannot alias one file")
        _require(_atom_count(_text(original_text)) == 0,
                 "original instructions contain an existing model; expert or supplied models are not independent SHELXT input")
        read_entries, output_entries = _entries(invocation, "inputs", "input"), _entries(invocation, "outputs", "output")
        _require(set(read_entries) == {stem + ".ins", stem + ".hkl"}, "SHELXT recorded inputs outside the explicit instruction/reflection pair")
        branch_inputs = {}
        instruction_text = b""
        for role, kind in (("ins", "SHELXT_INSTRUCTIONS"), ("hkl", "SHELXT_INPUT_REFLECTIONS")):
            name = stem + "." + role
            pin, data = reader.read(branch + "/" + name, kind, collect=role == "ins",
                                    max_bytes=_TEXT_LIMIT if role == "ins" else None)
            _require(read_entries[name].get("sha256") == pin.sha256, "SHELXT input changed after recorded execution")
            branch_inputs[role] = pin
            if role == "ins":
                instruction_text = data
        _require(branch_inputs["hkl"].sha256 == originals["hkl"].sha256,
                 "SHELXT did not read the expected measured reflection set")
        _require(staging.get("stem") == stem and staging.get("ins") == stem + ".ins"
                 and staging.get("hkl") == stem + ".hkl"
                 and staging.get("ins_source_sha256") == originals["ins"].sha256
                 and staging.get("ins_staged_sha256") == branch_inputs["ins"].sha256
                 and staging.get("hkl_sha256") == originals["hkl"].sha256,
                 "SHELXT staging evidence does not connect original and actual inputs")
        if branch_inputs["ins"].sha256 == originals["ins"].sha256:
            _require(staging.get("ins_is_verbatim") is True and staging.get("changes") == [],
                     "verbatim instruction staging is inconsistently recorded")
        else:
            raw_composition = ((summary.get("request") or {}).get("prior") or {}).get("composition")
            _require(isinstance(raw_composition, list) and bool(raw_composition), "changed instructions lack an explicit composition record")
            composition = []
            for entry in raw_composition:
                _require(isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str)
                         and type(entry[1]) in {int, float} and math.isfinite(entry[1]) and entry[1] > 0,
                         "invalid explicit instruction composition record")
                composition.append((entry[0], entry[1]))
            derived, changes = apply_composition_override(_text(original_text), tuple(composition))
            _require(derived.encode() == instruction_text and staging.get("changes") == list(changes)
                     and staging.get("ins_is_verbatim") is False,
                     "actual instruction edits differ from explicit retained transformation")
        _require(_atom_count(_text(instruction_text)) == 0, "SHELXT instructions contain an existing model")
        pins = {}
        for name, role in ((model_name, FileRole.AUTOMATIC_INITIAL_MODEL.value),
                           (hkl_name, "SHELXT_MATCHING_REFLECTIONS")):
            entry = output_entries.get(name)
            _require(isinstance(entry, dict) and entry.get("role") == "output",
                     "chosen model/matching HKL were not newly written SHELXT outputs; fallback is unverified")
            pin, data = reader.read(branch + "/" + name, role, collect=name == model_name,
                                    max_bytes=_TEXT_LIMIT if name == model_name else None)
            _require(pin.sha256 == entry.get("sha256") and pin.size_bytes == entry.get("size_bytes"),
                     "chosen SHELXT output differs from its written provenance")
            if name == model_name:
                _require(_atom_count(_text(data), include_peaks=False) > 0, "automatic output contains no model atom records")
            pins[name] = pin
        # Do not call the broader decision reader: it can inspect the unverified REM fallback.
        # Both matching filenames above use its shared normal-path naming contract, and both
        # were explicitly recorded as new output and read safely through directory descriptors.
        reader.recheck()
        return InitialModelReceipt(root, state["run_id"], branch, provenance, stem, solution_letter,
                                   require_real, synthetic, source_duration, originals["ins"], originals["hkl"],
                                   branch_inputs["ins"], branch_inputs["hkl"], pins[model_name], pins[hkl_name],
                                   records, tuple(sorted(identity.items())))


def create_initial_model_receipt(run_root: Path, *, branch: str, provenance: str,
                                 solution_letter: str, expected_inputs: Mapping[str, str],
                                 require_real: bool = True) -> InitialModelReceipt:
    """Build a narrow initial-checkpoint receipt without running or accepting the whole workflow."""
    try:
        return _create(run_root, branch=branch, provenance=provenance, solution_letter=solution_letter,
                       expected_inputs=expected_inputs, require_real=require_real)
    except HandoffError:
        raise
    except (SealError, OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, RecursionError) as exc:
        raise HandoffError("automatic checkpoint evidence is missing, malformed, unsafe or changed") from exc


def verify_initial_model_receipt(receipt: InitialModelReceipt | Mapping[str, Any], *,
                                 expected_receipt_sha256: str) -> VerifiedInitialModelHandoff:
    """Rebuild receipt evidence against an independently retained digest before any handoff."""
    try:
        if not isinstance(receipt, InitialModelReceipt):
            receipt = InitialModelReceipt.from_record(receipt)
        _require(receipt.receipt_sha256 == _sha(expected_receipt_sha256), "receipt differs from the independently pinned digest")
        _require(receipt.automatic_model.role == FileRole.AUTOMATIC_INITIAL_MODEL.value
                 and receipt.original_instructions.role == FileRole.INSTRUMENT_INSTRUCTIONS.value
                 and receipt.measured_reflections.role == FileRole.MEASURED_REFLECTIONS.value
                 and receipt.paired_reflections.role == "SHELXT_MATCHING_REFLECTIONS",
                 "receipt file roles differ from the automatic checkpoint contract")
        rebuilt = create_initial_model_receipt(receipt.run_root, branch=receipt.branch,
            provenance=receipt.provenance, solution_letter=receipt.solution_letter,
            expected_inputs={"ins": receipt.original_instructions.sha256, "hkl": receipt.measured_reflections.sha256},
            require_real=receipt.require_real)
        _require(rebuilt == receipt, "retained checkpoint evidence no longer matches the receipt")
        return VerifiedInitialModelHandoff(receipt.run_id, receipt.receipt_sha256,
            receipt.run_root / receipt.automatic_model.relative_path,
            receipt.run_root / receipt.paired_reflections.relative_path,
            receipt.run_root / receipt.original_instructions.relative_path,
            receipt.run_root / receipt.measured_reflections.relative_path,
            receipt.automatic_model.sha256, receipt.paired_reflections.sha256,
            receipt.measured_reflections.sha256, receipt.source_duration_s, receipt.synthetic)
    except HandoffError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, RecursionError) as exc:
        raise HandoffError("automatic checkpoint receipt is malformed or cannot be reverified") from exc
