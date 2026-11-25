"""Strict, local-only input contracts for private acceptance.

Solver and grader JSON documents are deliberately different contracts. An archive's descriptive
``case.json`` is never treated as either. Loading a solver spec opens only that spec and its
explicit instruction input; it never opens grader files, answers or related structures. The
checks constrain this application's inputs, not the filesystem permissions of an external tool.
The caller must stage copies and run in a fresh directory; same-user processes are not sandboxed.
"""

from __future__ import annotations

import enum
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from crystalx_core.app.models import ChemicalPriorInput, RunConfig, RunMode, SolveRequest
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.errors import ParseError
from crystalx_core.io.shelx_reader import logical_lines, read_ins_header
from crystalx_core.io.shelx_res import read_res
from crystalx_core.workspace.hashing import sha256_file

__all__ = [
    "AcceptanceTrack", "FileRole", "DeclaredPrivateFile", "InitialModelSeed", "SpecError",
    "PriorRecord", "PreparationSpec", "SolverSpec",
    "GraderSpec", "contained_path", "load_solver_spec", "load_grader_spec",
    "parse_solver_spec", "parse_grader_spec", "synthetic_solver_template",
]


class SpecError(ValueError):
    """A private acceptance document is not safe or explicit enough to use."""


class AcceptanceTrack(str, enum.Enum):
    DE_NOVO = "DE_NOVO"
    COMPONENT_REGRESSION = "COMPONENT_REGRESSION"
    UNSOLVED = "UNSOLVED"
    INITIAL_MODEL_RECOVERY = "INITIAL_MODEL_RECOVERY"


class FileRole(str, enum.Enum):
    """Declared use, never inferred from a filename or extension."""

    MEASURED_REFLECTIONS = "MEASURED_REFLECTIONS"
    INSTRUMENT_INSTRUCTIONS = "INSTRUMENT_INSTRUCTIONS"
    AUTOMATIC_INITIAL_MODEL = "AUTOMATIC_INITIAL_MODEL"
    EXPERT_FINAL_ANSWER = "EXPERT_FINAL_ANSWER"


@dataclass(frozen=True)
class DeclaredPrivateFile:
    role: FileRole
    path: Path


@dataclass(frozen=True)
class InitialModelSeed:
    """An owner-declared automatic checkpoint; never a final scientific answer.

    The digests pin identities. They do not prove crystallographic compatibility or authenticate
    the owner's provenance declaration. Missing seeds retain the requested path and hashes with
    ``actual_sha256=None`` so that only this track is blocked; no alternative file is discovered.
    """

    path: Path
    sha256: str
    compatible_hkl_sha256: str
    provenance_source: str
    owner_declaration: str
    actual_sha256: str | None = None

    @property
    def role(self) -> FileRole:
        return FileRole.AUTOMATIC_INITIAL_MODEL

    @property
    def available(self) -> bool:
        return self.actual_sha256 is not None

    @property
    def compatibility_scientifically_verified(self) -> bool:
        return False


@dataclass(frozen=True)
class PriorRecord:
    field: str
    value: str | bool
    provenance: str
    source: str


@dataclass(frozen=True)
class PreparationSpec:
    policy: str = "preserve_input"
    composition: tuple[tuple[str, float], ...] = ()
    owner_review_reference: str = ""
    official_semantics_reference: str = ""


@dataclass(frozen=True)
class SolverSpec:
    case_id: str
    track: AcceptanceTrack
    case_root: Path
    ins_path: Path
    hkl_path: Path
    prior: ChemicalPriorInput
    prior_records: tuple[PriorRecord, ...]
    preparation: PreparationSpec
    seed: InitialModelSeed | None = None
    blockers: tuple[str, ...] = ()

    @property
    def input_roles(self) -> tuple[DeclaredPrivateFile, ...]:
        # The contract and explicit track supply roles. File extensions validate grammar only.
        instruction_role = (FileRole.EXPERT_FINAL_ANSWER
                            if self.track is AcceptanceTrack.COMPONENT_REGRESSION
                            else FileRole.INSTRUMENT_INSTRUCTIONS)
        roles = (DeclaredPrivateFile(instruction_role, self.ins_path),
                 DeclaredPrivateFile(FileRole.MEASURED_REFLECTIONS, self.hkl_path))
        if self.seed is not None:
            roles += (DeclaredPrivateFile(FileRole.AUTOMATIC_INITIAL_MODEL, self.seed.path),)
        return roles

    @property
    def is_initial_solution_track(self) -> bool:
        """Route intent only, never a successful-solution claim about any run."""
        return self.track in {AcceptanceTrack.DE_NOVO, AcceptanceTrack.UNSOLVED}

    def to_solve_request(self, config: RunConfig | None = None) -> SolveRequest:
        """Only the two allowlisted files and checked priors cross the solver boundary."""
        if self.track is AcceptanceTrack.INITIAL_MODEL_RECOVERY:
            raise SpecError("INITIAL_MODEL_RECOVERY requires the explicit iterative seed interface; the one-shot initial-solution route is forbidden")
        return SolveRequest(
            ins_path=self.ins_path, hkl_path=self.hkl_path, prior=self.prior,
            config=config or RunConfig(strict_native=True), mode=RunMode.REAL, label=self.case_id,
        )


@dataclass(frozen=True)
class GraderSpec:
    case_id: str
    track: AcceptanceTrack
    case_root: Path
    answer_path: Path | None = None
    reference_paths: tuple[Path, ...] = ()
    thresholds_owner_approved: bool = False
    thresholds_approval_reference: str = ""
    site_match_tolerance_angstrom: float | None = None
    heavy_atomic_number_min: int = 20

    @property
    def answer_role(self) -> FileRole | None:
        return FileRole.EXPERT_FINAL_ANSWER if self.answer_path is not None else None


_FORBIDDEN = frozenset({
    "expected_answer", "known_chemistry", "answer", "answer_path", "reference_paths",
    "expert_model", "expert_structure", "expected_atom_counts", "score", "scores",
})
_CASE_ID = re.compile(r"^[^/\\\x00-\x1f\x7f]{1,128}$")


def _object(value: Any, allowed: set[str], required: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SpecError(f"{where} must be an object with string keys")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise SpecError(f"{where} contains unsupported fields: {', '.join(sorted(unknown))}")
    if missing:
        raise SpecError(f"{where} is missing fields: {', '.join(sorted(missing))}")
    return value


def _text(value: Any, where: str, *, maximum: int = 2048) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SpecError(f"{where} must be a nonempty string of at most {maximum} characters")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise SpecError(f"{where} contains control characters")
    return value


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise SpecError(f"{where} must be a JSON boolean")
    return value


def _positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{where} must be a finite positive number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise SpecError(f"{where} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise SpecError(f"{where} must be a finite positive number")
    return result


def _track(value: Any) -> AcceptanceTrack:
    try:
        return AcceptanceTrack(value)
    except (ValueError, TypeError) as exc:
        raise SpecError("track must be DE_NOVO, COMPONENT_REGRESSION, UNSOLVED or INITIAL_MODEL_RECOVERY") from exc


def _case_id(value: Any) -> str:
    result = _text(value, "case_id", maximum=128)
    if not _CASE_ID.fullmatch(result) or result.strip() in {".", ".."}:
        raise SpecError("case_id must be an existing local identifier without path separators")
    return result


def _root(root: Path | str) -> Path:
    try:
        result = Path(root).resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SpecError("case_root must be an existing local directory") from exc
    if not result.is_dir():
        raise SpecError("case_root must be a directory")
    return result


def contained_path(root: Path | str, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve one explicit relative file and reject traversal and symlink escapes.

    This is a point-in-time path check, not protection against a hostile process changing files
    concurrently. Staging must verify its copies/hashes and keep source originals untouched.
    """
    base = _root(root)
    text = _text(relative, "path", maximum=4096)
    candidate = Path(text)
    if (candidate.is_absolute() or PureWindowsPath(text).drive or "\\" in text
            or any(part in {"..", "."} for part in text.split("/"))):
        raise SpecError("input paths must be relative, without traversal or alternate separators")
    try:
        resolved = (base / candidate).resolve(strict=must_exist)
        resolved.relative_to(base)
    except (ValueError, OSError, RuntimeError) as exc:
        raise SpecError("input path is missing or escapes case_root") from exc
    if must_exist and not resolved.is_file():
        raise SpecError("input path must name a regular file")
    return resolved


def _no_answer_fields(value: Any, where: str = "solver") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SpecError(f"{where} contains a non-string key")
            if key.casefold() in _FORBIDDEN:
                raise SpecError(f"{where} contains a grader-only or archive field: {key}")
            _no_answer_fields(child, f"{where}.{key}")
    elif isinstance(value, list):
        for child in value:
            _no_answer_fields(child, where)


def _version(value: Any) -> None:
    if type(value) is not int or value != 1:
        raise SpecError("schema_version must be the integer 1")


def _element(value: Any, where: str) -> str:
    # Symbols are intentionally strict: do not silently reinterpret owner declarations.
    if not isinstance(value, str) or value not in ATOMIC_NUMBERS:
        raise SpecError(f"{where} must be a canonical element symbol")
    return value


def _declaration(value: Any, field: str) -> PriorRecord:
    item = _object(value, {"value", "provenance", "source"},
                   {"value", "provenance", "source"}, f"priors.{field}")
    provenance = item["provenance"]
    if not isinstance(provenance, str) or provenance not in {"owner_confirmed", "source_ins"}:
        raise SpecError("prior provenance must be owner_confirmed or source_ins; expert-derived priors are excluded")
    source = _text(item["source"], f"priors.{field}.source")
    declared = (_element(item["value"], field) if field in {"elements", "metals"}
                else _boolean(item["value"], field))
    if field in {"peptide", "racemic"} and provenance != "owner_confirmed":
        raise SpecError(f"{field} requires independent owner confirmation")
    return PriorRecord(field=field, value=declared, provenance=provenance, source=source)


def _priors(value: Any, ins_path: Path, track: AcceptanceTrack) -> tuple[ChemicalPriorInput, tuple[PriorRecord, ...]]:
    data = _object(value, {"elements", "metals", "peptide", "racemic"}, set(), "priors")
    records: list[PriorRecord] = []
    values: dict[str, Any] = {"elements": (), "metals": (), "peptide": False, "racemic": False}
    for field in ("elements", "metals"):
        declarations = data.get(field, [])
        if not isinstance(declarations, list) or len(declarations) > len(ATOMIC_NUMBERS):
            raise SpecError(f"priors.{field} must be a bounded array")
        parsed = tuple(_declaration(item, field) for item in declarations)
        symbols = tuple(str(item.value) for item in parsed)
        if len(set(symbols)) != len(symbols):
            raise SpecError(f"priors.{field} contains duplicate elements")
        records.extend(parsed)
        values[field] = symbols
    for field in ("peptide", "racemic"):
        if field in data:
            record = _declaration(data[field], field)
            values[field] = record.value
            records.append(record)
    try:
        # File-inclusion directives defeat an INS/HKL-only input contract. Reject them rather
        # than interpreting or rewriting them; all required instructions must be in the selected
        # local input. This is still not an operating-system sandbox for the eventual binary.
        if any(line.instruction.startswith("+") for line in logical_lines(ins_path)):
            raise SpecError("instruction file contains a file-inclusion directive; only explicit INS/HKL inputs are allowed")
        header = read_ins_header(ins_path)
    except SpecError:
        raise
    except (ValueError, OSError, OverflowError, ParseError) as exc:
        raise SpecError("instruction input could not be checked") from exc
    if track is not AcceptanceTrack.COMPONENT_REGRESSION and header.has_atom_records:
        raise SpecError("blind input contains model atom records; use an explicit component regression track")
    source_elements = set(header.sfac)
    for record in records:
        if record.provenance == "source_ins" and record.value not in source_elements:
            raise SpecError("source_ins element prior is absent from the original SFAC")
    return ChemicalPriorInput(**values), tuple(records)


def _preparation(value: Any, track: AcceptanceTrack) -> PreparationSpec:
    data = _object(value, {"policy", "composition", "owner_review_reference", "official_semantics_reference"},
                   {"policy"}, "preparation")
    policy = data["policy"]
    if policy == "preserve_input":
        if set(data) != {"policy"}:
            raise SpecError("preserve_input must not include composition edits or inferred counts")
        return PreparationSpec()
    if policy != "explicit_reviewed_composition":
        raise SpecError("preparation policy must preserve_input or explicit_reviewed_composition")
    if track in {AcceptanceTrack.COMPONENT_REGRESSION, AcceptanceTrack.INITIAL_MODEL_RECOVERY}:
        raise SpecError("model-based tracks must preserve supplied model SFAC and atom indices")
    _object(data, set(data), {"composition", "owner_review_reference", "official_semantics_reference"}, "preparation")
    owner = _text(data["owner_review_reference"], "owner_review_reference")
    semantics = _text(data["official_semantics_reference"], "official_semantics_reference")
    items = data["composition"]
    if not isinstance(items, list) or not 1 <= len(items) <= len(ATOMIC_NUMBERS):
        raise SpecError("composition must explicitly list the full reviewed SFAC/UNIT replacement")
    composition: list[tuple[str, float]] = []
    for item in items:
        obj = _object(item, {"element", "count"}, {"element", "count"}, "composition entry")
        composition.append((_element(obj["element"], "composition"), _positive(obj["count"], "composition count")))
    if len({symbol for symbol, _ in composition}) != len(composition):
        raise SpecError("composition contains duplicate elements")
    return PreparationSpec(policy, tuple(composition), owner, semantics)


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SpecError(f"{where} must be a canonical SHA-256 digest")
    return value


def _initial_seed(value: Any, *, root: Path, ins: Path, hkl: Path) -> InitialModelSeed:
    data = _object(value, {"path", "role", "sha256", "compatible_hkl_sha256", "provenance"},
                   {"path", "role", "sha256", "compatible_hkl_sha256", "provenance"}, "seed")
    if data["role"] != FileRole.AUTOMATIC_INITIAL_MODEL.value:
        raise SpecError("reconstruction seed must explicitly declare AUTOMATIC_INITIAL_MODEL; expert answers are never substituted")
    provenance = _object(data["provenance"], {"kind", "source", "owner_declaration"},
                         {"kind", "source", "owner_declaration"}, "seed.provenance")
    if provenance["kind"] != "owner_confirmed_automatic_initial_model":
        raise SpecError("seed requires independent owner-confirmed automatic-initial-model provenance")
    source = _text(provenance["source"], "seed.provenance.source")
    declaration = _text(provenance["owner_declaration"], "seed.provenance.owner_declaration")
    digest = _sha256(data["sha256"], "seed.sha256")
    hkl_digest = _sha256(data["compatible_hkl_sha256"], "seed.compatible_hkl_sha256")
    path = contained_path(root, data["path"], must_exist=False)
    if path in {ins, hkl}:
        raise SpecError("automatic seed must be a distinct declared file, not a renamed instruction or HKL input")
    try:
        if sha256_file(hkl) != hkl_digest:
            raise SpecError("seed's compatible reflection identity differs from the selected HKL")
        if not path.exists():
            return InitialModelSeed(path, digest, hkl_digest, source, declaration)
        if not path.is_file():
            raise SpecError("automatic seed must name a regular file")
        if path.samefile(ins) or path.samefile(hkl):
            raise SpecError("automatic seed aliases a different input role")
        actual = sha256_file(path)
        if actual != digest:
            raise SpecError("automatic seed digest differs from its explicitly pinned identity")
        if any(line.instruction.startswith("+") for line in logical_lines(path)):
            raise SpecError("automatic seed contains a file-inclusion directive")
        model = read_res(path)
        if model.n_atoms == 0:
            raise SpecError("automatic seed contains no model atoms; an instruction file is not a generated initial model")
        if model.header.cell is None or model.header.latt is None or model.header.hklf_format is None:
            raise SpecError("automatic seed must explicitly preserve its source CELL, LATT and HKLF setting")
        cell = model.header.cell
        if any(not math.isfinite(value) for value in (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)):
            raise SpecError("automatic seed contains a nonfinite source CELL setting")
        if any(not 1 <= atom.sfac_index <= len(model.header.sfac) for atom in model.atoms):
            raise SpecError("automatic seed contains an invalid SFAC atom index")
        if any(not math.isfinite(v) for atom in model.atoms
               for v in (atom.x, atom.y, atom.z, atom.occupancy_code, *atom.u_values)):
            raise SpecError("automatic seed contains nonfinite model atom values")
        if model.sha256 != digest or sha256_file(path) != digest:
            raise SpecError("automatic seed changed while its identity and model were checked")
    except SpecError:
        raise
    except (OSError, ValueError, OverflowError, ParseError) as exc:
        raise SpecError("automatic seed could not be safely checked") from exc
    return InitialModelSeed(path, digest, hkl_digest, source, declaration, actual)


def parse_solver_spec(data: Any, *, case_root: Path | str) -> SolverSpec:
    """Validate only an allowlisted solver document; no archive object is accepted."""
    try:
        _no_answer_fields(data)
    except RecursionError as exc:
        raise SpecError("solver spec nesting is too deep") from exc
    obj = _object(data, {"schema_version", "case_id", "track", "inputs", "priors", "preparation", "seed"},
                  {"schema_version", "case_id", "track", "inputs"}, "solver spec")
    _version(obj["schema_version"])
    case_id = _case_id(obj["case_id"])
    track = _track(obj["track"])
    if "seed" in obj and track is not AcceptanceTrack.INITIAL_MODEL_RECOVERY:
        raise SpecError("only INITIAL_MODEL_RECOVERY accepts an automatic seed; blind and expert-component routes remain separate")
    inputs = _object(obj["inputs"], {"ins", "hkl"}, {"ins", "hkl"}, "inputs")
    root = _root(case_root)
    ins = contained_path(root, inputs["ins"])
    hkl = contained_path(root, inputs["hkl"])
    suffixes = {".ins", ".res"} if track is AcceptanceTrack.COMPONENT_REGRESSION else {".ins"}
    if ins.suffix.lower() not in suffixes or hkl.suffix.lower() != ".hkl" or ins == hkl:
        raise SpecError("inputs must select an instruction input and a distinct HKL reflection file")
    prior, records = _priors(obj.get("priors", {}), ins, track)
    preparation = _preparation(obj.get("preparation", {"policy": "preserve_input"}), track)
    if preparation.composition:
        prior = ChemicalPriorInput(metals=prior.metals, elements=prior.elements, peptide=prior.peptide,
                                   racemic=prior.racemic, composition=preparation.composition)
    seed = (_initial_seed(obj["seed"], root=root, ins=ins, hkl=hkl) if "seed" in obj else None)
    blockers = (("automatic_initial_model_missing",)
                if track is AcceptanceTrack.INITIAL_MODEL_RECOVERY and (seed is None or not seed.available) else ())
    return SolverSpec(case_id, track, root, ins, hkl, prior, records, preparation, seed, blockers)


def parse_grader_spec(data: Any, *, case_root: Path | str) -> GraderSpec:
    """Resolve held-out paths without reading their contents. Call only in the grading phase."""
    obj = _object(data, {"schema_version", "case_id", "track", "answer", "references", "thresholds"},
                  {"schema_version", "case_id", "track"}, "grader spec")
    _version(obj["schema_version"])
    root = _root(case_root)
    case_id = _case_id(obj["case_id"])
    track = _track(obj["track"])
    answer = contained_path(root, obj["answer"]) if "answer" in obj else None
    if track is AcceptanceTrack.UNSOLVED and answer is not None:
        raise SpecError("UNSOLVED has no ground truth; put an advisory structure in references")
    references = obj.get("references", [])
    if not isinstance(references, list) or len(references) > 32:
        raise SpecError("references must be a bounded array of advisory paths")
    paths = tuple(contained_path(root, path) for path in references)
    if len(set(paths)) != len(paths) or (answer is not None and answer in paths):
        raise SpecError("grader paths must be distinct")
    thresholds = _object(obj.get("thresholds", {}), {
        "owner_approved", "approval_reference", "site_match_tolerance_angstrom", "heavy_atomic_number_min",
    }, set(), "thresholds")
    approved = _boolean(thresholds.get("owner_approved", False), "thresholds.owner_approved")
    approval = _text(thresholds["approval_reference"], "thresholds.approval_reference") if "approval_reference" in thresholds else ""
    tolerance = (_positive(thresholds["site_match_tolerance_angstrom"], "site match tolerance")
                 if "site_match_tolerance_angstrom" in thresholds else None)
    heavy_min = thresholds.get("heavy_atomic_number_min", 20)
    if type(heavy_min) is not int or not 1 <= heavy_min <= 118:
        raise SpecError("heavy_atomic_number_min must be an integer from 1 to 118")
    if approved and (not approval or tolerance is None):
        raise SpecError("approved thresholds need explicit owner evidence and a positive site tolerance")
    return GraderSpec(case_id, track, root, answer, paths, approved, approval, tolerance, heavy_min)


def _read_json(path: Path | str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SpecError(f"JSON repeats a field: {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise SpecError("nonfinite JSON number")

    try:
        target = Path(path)
        if not target.is_file():
            raise SpecError("acceptance spec must be a regular file")
        document_limit = 1024 * 1024
        if target.stat().st_size > document_limit:
            raise SpecError("acceptance spec exceeds the 1 MiB document limit")
        with target.open("rb") as stream:
            payload = stream.read(document_limit + 1)
        if len(payload) > document_limit:
            raise SpecError("acceptance spec exceeds the 1 MiB document limit")
        return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SpecError("acceptance spec is not readable strict JSON") from exc


def load_solver_spec(path: Path | str, *, case_root: Path | str | None = None) -> SolverSpec:
    target = Path(path)
    return parse_solver_spec(_read_json(target), case_root=case_root if case_root is not None else target.parent)


def load_grader_spec(path: Path | str, *, case_root: Path | str | None = None) -> GraderSpec:
    target = Path(path)
    return parse_grader_spec(_read_json(target), case_root=case_root if case_root is not None else target.parent)


def synthetic_solver_template() -> dict[str, Any]:
    """Public example contains fictional identifiers, filenames and no scientific answer."""
    return {"schema_version": 1, "case_id": "synthetic_example", "track": "DE_NOVO",
            "inputs": {"ins": "original.ins", "hkl": "original.hkl"}, "priors": {},
            "preparation": {"policy": "preserve_input"}}
