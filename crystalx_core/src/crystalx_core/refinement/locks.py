"""Apply a deliberately small, reversible subset of explicit SHELXL locks.

Always rebuild from the immutable lock-free baseline. The output normalizer
requires that baseline, its reproduced application and the original ledger;
it never reverse-engineers arbitrary refined RES constraints. Application changes
only approved coordinate fields or adds explicit one-pair DFIX instructions;
occupancy, displacement, composition, symmetry and reflection settings survive.

Primary semantics: https://shelx.uni-goettingen.de/shelxl_html.php (atom records,
DFIX, AFIX, PART, RESI and HKLF). SHELXL encodes a fixed parameter as 10+p;
DFIX uses a target distance and an ESD. This adapter neither infers those values
nor establishes that a restraint or coordinate lock is scientifically justified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import re

from crystalx_core.domain.hypothesis import LockDirective, LockMode
from crystalx_core.io.shelx_res import SHELX_INSTRUCTIONS


__all__ = ["LockApplicationError", "AppliedLockRecord", "LockApplicationResult", "apply_locks",
           "LockOutputNormalization", "normalize_managed_output"]


class LockApplicationError(ValueError):
    """An active lock cannot be applied without guessing or widening its effect."""


@dataclass(frozen=True)
class AppliedLockRecord:
    lock_id: str
    mode: str
    atom_ids: tuple[str, ...]
    effect: str
    instruction_lines: tuple[str, ...]
    reason_sha256: str
    hard_lock_review_approved: bool


@dataclass(frozen=True)
class LockApplicationResult:
    text: str
    applied_ids: tuple[str, ...]
    baseline_sha256: str
    output_sha256: str
    records: tuple[AppliedLockRecord, ...]


@dataclass(frozen=True)
class LockOutputNormalization:
    text: str
    raw_output_sha256: str
    lock_free_sha256: str
    baseline_sha256: str
    application_sha256: str
    removed_lock_ids: tuple[str, ...]
    decoded_atom_ids: tuple[str, ...]
    managed_dialect: str = "intact_contiguous_rem_block_v1"


@dataclass(frozen=True)
class _Atom:
    label: str
    line_index: int
    coordinate_spans: tuple[tuple[int, int], ...]
    coordinate_tokens: tuple[str, ...]
    context_bounded: bool


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}\Z")
_ATOM_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9]{0,3}\Z")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?\Z")
_BEGIN = "REM CrystalX active constraints BEGIN"
_END = "REM CrystalX active constraints END"
_MAX_LOCKS = 128
_MAX_DFIX_LINES = 32
_MAX_BASE_BYTES = 8 * 1024 * 1024
_MAX_ESD = Decimal("1000000")  # A parser/resource bound, not a scientific tolerance.
_COMMANDS = SHELX_INSTRUCTIONS | frozenset({"ABIN", "NEUT", "TWST", "WIGL"})


def _hash(text: str) -> str:
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeError as exc:
        raise LockApplicationError("Lock text must be valid UTF-8") from exc


def _number(token: str) -> Decimal:
    if len(token) > 48 or not _NUMBER.fullmatch(token):
        raise LockApplicationError("Lock effect requires an ordinary finite numeric field")
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise LockApplicationError("Lock numeric field is invalid") from exc
    if not value.is_finite() or value.adjusted() > 12 or value.adjusted() < -12:
        raise LockApplicationError("Lock numeric field is outside the supported representation")
    return value


def _render_number(value: Decimal) -> str:
    # Never round a scientific value merely to fit the instruction width.
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result if result != "-0" else "0"


def _visible(line: str) -> str:
    return line.split("!", 1)[0].split("=", 1)[0]


def _scan(text: str) -> tuple[list[str], dict[str, list[_Atom]], int, set[frozenset[str]]]:
    lines = text.splitlines(keepends=True)
    atoms: dict[str, list[_Atom]] = {}
    hklf_indices: list[int] = []
    prior_pairs: set[frozenset[str]] = set()
    residue = part = afix = 0
    fragment = False
    coupled_coordinates: set[str] = set()
    continuation = False
    reached_hklf = False
    for index, physical in enumerate(lines):
        line = physical.rstrip("\r\n")
        if reached_hklf:
            trailing = _visible(line).split()
            if trailing and not line[0].isspace() and trailing[0].upper() == "HKLF":
                hklf_indices.append(index)
            continue  # RES output comments and Q peaks are not refinement atoms.
        if line.startswith("+"):
            raise LockApplicationError("Included instruction files are outside this bounded lock adapter")
        if continuation:
            if not line or not line[0].isspace():
                raise LockApplicationError("Malformed SHELXL continuation in baseline")
            continuation = "=" in line.split("!", 1)[0]
            continue
        if not line or line[0].isspace():
            continue  # Official SHELXL treats these as comments, except continuations.
        visible = _visible(line)
        spans = list(re.finditer(r"\S+", visible))
        if not spans:
            continue
        tokens = [match.group() for match in spans]
        command = tokens[0].upper()
        continuation = "=" in line.split("!", 1)[0]
        if command == "REM":
            continuation = False
            continue
        if command == "HKLF":
            if residue or part or afix:
                raise LockApplicationError("Baseline must restore neutral RESI, PART and AFIX context before HKLF")
            hklf_indices.append(index)
            reached_hklf = True
            continue
        if command == "END":
            raise LockApplicationError("Baseline terminates before its reflection instruction")
        if command == "MOVE":
            raise LockApplicationError("Coordinate-transforming MOVE is outside this bounded lock adapter")
        if command == "EXYZ":
            if any(not _ATOM_LABEL.fullmatch(token) for token in tokens[1:]):
                raise LockApplicationError("Coordinate equivalence selectors are outside this bounded lock adapter")
            coupled_coordinates.update(token.casefold().split("_", 1)[0] for token in tokens[1:])
        if command.split("_", 1)[0] in {"DFIX", "EXYZ"} and "_" in command:
            raise LockApplicationError("Scoped restraint/equivalence instructions need an audited selector adapter")
        if command in {"RESI", "PART", "AFIX"}:
            try:
                value = int(tokens[1])
            except (IndexError, ValueError) as exc:
                raise LockApplicationError("Unsupported residue, disorder or rigid-group context") from exc
            if command == "RESI":
                residue = value
            elif command == "PART":
                part = value
            else:
                afix = value
            continue
        if command == "FRAG":
            fragment = True
        if command == "FEND":
            fragment = False
        if command == "DFIX":
            if continuation:
                raise LockApplicationError("Existing continued DFIX instructions need an audited restraint adapter")
            # Recognise existing simple numeric distance restraints so a second
            # copy cannot silently double their weight. Keep the original bytes.
            cursor = 1
            while cursor < len(tokens) and _NUMBER.fullmatch(tokens[cursor]):
                cursor += 1
            names = tokens[cursor:]
            if any(not _ATOM_LABEL.fullmatch(token) for token in names) or len(names) % 2:
                raise LockApplicationError("Existing DFIX selectors are outside this bounded lock adapter")
            for first, second in zip(names[::2], names[1::2]):
                prior_pairs.add(frozenset((first.casefold(), second.casefold())))
        if fragment or command in _COMMANDS or not _ATOM_LABEL.fullmatch(tokens[0]):
            continue
        if len(tokens) < 5:
            continue
        try:
            sfac = int(tokens[1])
        except ValueError:
            continue
        if sfac < 1:
            continue
        atoms.setdefault(tokens[0].casefold(), []).append(_Atom(
            tokens[0], index, tuple(match.span() for match in spans[2:5]),
            tuple(tokens[2:5]), residue == part == afix == 0,
        ))
    if continuation or fragment:
        raise LockApplicationError("Unterminated baseline continuation or fragment")
    if len(hklf_indices) != 1:
        raise LockApplicationError("Baseline requires exactly one HKLF instruction")
    for label in coupled_coordinates:
        if label in atoms:
            atoms[label] = [replace(atom, context_bounded=False) for atom in atoms[label]]
    return lines, atoms, hklf_indices[0], prior_pairs


def _resolve(atom_id: object, atoms: dict[str, list[_Atom]], *, require_bounded: bool) -> _Atom:
    if not isinstance(atom_id, str) or not _ATOM_LABEL.fullmatch(atom_id):
        raise LockApplicationError("Locks require explicit supported atom labels, without residue or symmetry selectors")
    matches = atoms.get(atom_id.casefold(), [])
    if len(matches) != 1:
        raise LockApplicationError("Lock target is absent or has an ambiguous atom identity")
    atom = matches[0]
    if require_bounded and not atom.context_bounded:
        raise LockApplicationError("Lock target uses an unsupported residue, disorder or AFIX context")
    for token in atom.coordinate_tokens:
        _number(token)
    return atom


def _dfix(tokens: list[str], named: set[str], atoms: dict[str, list[_Atom]]) -> tuple[str, frozenset[str]]:
    if len(tokens) != 5 or tokens[0].upper() != "DFIX":
        raise LockApplicationError("Only DFIX with explicit distance, ESD and one atom pair is allowed")
    target, esd = _number(tokens[1]), _number(tokens[2])
    if not 0 < target < 15 or not 0 < esd <= _MAX_ESD:
        raise LockApplicationError("DFIX needs a positive direct distance below free-variable encoding and a bounded positive ESD")
    first = _resolve(tokens[3], atoms, require_bounded=True)
    second = _resolve(tokens[4], atoms, require_bounded=True)
    if first.label.casefold() == second.label.casefold():
        raise LockApplicationError("DFIX requires two distinct atom sites")
    pair = frozenset((first.label.casefold(), second.label.casefold()))
    if not pair.issubset(named):
        raise LockApplicationError("DFIX may affect only the directive's explicitly named atom sites")
    instruction = f"DFIX {_render_number(target)} {_render_number(esd)} {first.label} {second.label}"
    if len(instruction) > 80:
        raise LockApplicationError("DFIX exceeds the supported physical instruction width")
    return instruction, pair


def _soft_lines(directive: LockDirective, named: set[str], atoms: dict[str, list[_Atom]]) -> tuple[tuple[str, frozenset[str]], ...]:
    if not isinstance(directive.instruction_lines, tuple) or not isinstance(directive.target_distances, tuple):
        raise LockApplicationError("Restraint instructions and target distances must be explicit immutable sequences")
    lines = []
    for line in directive.instruction_lines:
        if (not isinstance(line, str) or len(line) > 80 or not line or line[0].isspace()
                or any(mark in line for mark in ("\r", "\n", "!", "=", ";"))):
            raise LockApplicationError("Raw restraint instruction is unsafe or unsupported")
        lines.append(_dfix(line.split(), named, atoms))
    targets = []
    for item in directive.target_distances:
        if not isinstance(item, tuple) or len(item) != 4:
            raise LockApplicationError("A distance target requires two atom sites, a target and an ESD")
        first, second, target, esd = item
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (target, esd)):
            raise LockApplicationError("Distance targets must be numeric values, not implicit hypotheses")
        targets.append(_dfix(["DFIX", str(target), str(esd), str(first), str(second)], named, atoms))
    if lines and targets and tuple(lines) != tuple(targets):
        raise LockApplicationError("Raw restraint instructions disagree with their recorded distance targets")
    result = tuple(lines or targets)
    if not 1 <= len(result) <= _MAX_DFIX_LINES:
        raise LockApplicationError("A soft restraint requires a bounded nonempty DFIX list")
    affected = set().union(*(pair for _, pair in result))
    if affected != named:
        raise LockApplicationError("Every declared soft-lock target must have an actual emitted restraint effect")
    return result


def _fix_coordinate(token: str) -> str:
    value = _number(token)
    if -5 < value < 5:
        with localcontext() as context:
            context.prec = 100
            return _render_number(Decimal(10) + value)
    if 5 < value < 15:
        return token  # Already explicitly fixed; never add ten twice.
    raise LockApplicationError("Coordinate uses unsupported free-variable or boundary encoding")


def apply_locks(base_ins_text: str, active_locks: Mapping[str, LockDirective], *,
                approved_hard_lock_ids: frozenset[str] = frozenset()) -> LockApplicationResult:
    """Return an audited derivative without changing the immutable baseline.

    Approval is supplied by the trusted controller using ledger lock IDs, never
    inferred from AI probabilities, reasons or directive metadata. EXPLORATORY
    and FREE add no constraints; existing occupancy/disorder codes are preserved.
    They do not reverse arbitrary constraints already present in the baseline.
    """
    if not isinstance(base_ins_text, str) or "\x00" in base_ins_text:
        raise LockApplicationError("Baseline instruction text is unsupported or exceeds its size bound")
    try:
        if len(base_ins_text.encode("utf-8")) > _MAX_BASE_BYTES:
            raise LockApplicationError("Baseline instruction text exceeds its size bound")
    except UnicodeError as exc:
        raise LockApplicationError("Baseline instruction text must be valid UTF-8") from exc
    if _BEGIN in base_ins_text:
        raise LockApplicationError("Rebuild locks from the original baseline, not a previously managed derivative")
    if not isinstance(active_locks, Mapping) or len(active_locks) > _MAX_LOCKS:
        raise LockApplicationError("Active locks must be a bounded explicit mapping")
    if (not isinstance(approved_hard_lock_ids, frozenset)
            or any(not isinstance(value, str) for value in approved_hard_lock_ids)):
        raise LockApplicationError("Hard-lock review approval must be an explicit immutable set of ledger IDs")
    baseline_digest = _hash(base_ins_text)
    if not active_locks:
        return LockApplicationResult(base_ins_text, (), baseline_digest, baseline_digest, ())
    if any(not isinstance(key, str) or not _ID.fullmatch(key) for key in active_locks):
        raise LockApplicationError("Lock ledger ID is invalid")
    lines, atoms, insertion, seen_pairs = _scan(base_ins_text)
    newline = "\r\n" if "\r\n" in base_ins_text else "\n"
    records = []
    emitted = [_BEGIN]
    hard_atoms: set[str] = set()
    modified_hard_atoms: set[str] = set()
    unconstrained_atoms: set[str] = set()
    for lock_id in sorted(active_locks):
        directive = active_locks[lock_id]
        if not isinstance(directive, LockDirective) or not isinstance(directive.mode, LockMode):
            raise LockApplicationError("Unsupported lock directive object or mode")
        if directive.reversible is not True:
            raise LockApplicationError("This adapter only accepts reversible lock directives")
        if not isinstance(directive.reason, str) or not directive.reason.strip():
            raise LockApplicationError("Lock justification is missing")
        if not isinstance(directive.atom_ids, tuple) or len(directive.atom_ids) > 256:
            raise LockApplicationError("Lock targets must be a bounded explicit tuple")
        resolved = [_resolve(value, atoms, require_bounded=directive.mode in {
            LockMode.HARD_LOCK, LockMode.SOFT_RESTRAINT}) for value in directive.atom_ids]
        named = {atom.label.casefold() for atom in resolved}
        if len(named) != len(resolved) or not named and directive.mode is not LockMode.FREE:
            raise LockApplicationError("Lock targets are empty or repeated")
        effects: tuple[str, ...] = ()
        approved = False
        if directive.mode is LockMode.SOFT_RESTRAINT:
            parsed = _soft_lines(directive, named, atoms)
            for _, pair in parsed:
                if pair in seen_pairs:
                    raise LockApplicationError("Duplicate DFIX pair would silently change restraint weighting")
                seen_pairs.add(pair)
            effects = tuple(line for line, _ in parsed)
            effect = "explicit_distance_restraint"
        else:
            if directive.instruction_lines or directive.target_distances:
                raise LockApplicationError("Raw instructions are not allowed for hard, exploratory or free modes")
            if directive.mode is LockMode.HARD_LOCK:
                if lock_id not in approved_hard_lock_ids:
                    raise LockApplicationError("Hard coordinate fixing requires explicit trusted scientific review approval")
                approved = True
                hard_atoms.update(named)
                effect = "fixed_fractional_coordinates"
                for atom in resolved:
                    if atom.label.casefold() in modified_hard_atoms:
                        continue
                    physical = lines[atom.line_index]
                    replacements = [_fix_coordinate(token) for token in atom.coordinate_tokens]
                    for (start, end), token in reversed(tuple(zip(atom.coordinate_spans, replacements))):
                        physical = physical[:start] + token + physical[end:]
                    if len(physical.rstrip("\r\n")) > 80:
                        raise LockApplicationError("Fixed atom row exceeds supported physical width; regenerate the baseline safely")
                    lines[atom.line_index] = physical
                    modified_hard_atoms.add(atom.label.casefold())
            else:
                unconstrained_atoms.update(named)
                effect = "no_new_constraint"
        emitted.append(f"REM CrystalX lock {lock_id} {directive.mode.value}")
        emitted.extend(effects)
        records.append(AppliedLockRecord(lock_id, directive.mode.value,
            tuple(atom.label for atom in resolved), effect, effects, _hash(directive.reason), approved))
    if hard_atoms.intersection(unconstrained_atoms):
        raise LockApplicationError("An atom cannot be both explicitly hard-locked and declared exploratory/free")
    emitted.append(_END)
    lines.insert(insertion, newline.join(emitted) + newline)
    output = "".join(lines)
    return LockApplicationResult(output, tuple(record.lock_id for record in records),
                                 baseline_digest, _hash(output), tuple(records))


def _managed(line: str) -> bool:
    text = line.strip().upper()
    return text.startswith("REM CRYSTALX ACTIVE CONSTRAINTS") or text.startswith("REM CRYSTALX LOCK ")


def _strip_exact_managed_block(text: str, application: LockApplicationResult) -> str:
    lines = text.splitlines(keepends=True)
    locations = [index for index, line in enumerate(lines) if _managed(line)]
    if not application.applied_ids:
        if locations:
            raise LockApplicationError("Output contains managed constraints absent from the application")
        return text
    expected_lines = application.text.splitlines()
    first = expected_lines.index(_BEGIN)
    last = expected_lines.index(_END, first + 1)
    expected = expected_lines[first:last + 1]
    begins = [index for index in locations if lines[index].rstrip("\r\n") == _BEGIN]
    ends = [index for index in locations if lines[index].rstrip("\r\n") == _END]
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise LockApplicationError("Managed REM boundaries are missing, duplicated or changed")
    first, last = begins[0], ends[0]
    if any(index < first or index > last for index in locations):
        raise LockApplicationError("Output contains additional managed audit instructions")
    # An indented DFIX is a comment in SHELXL, so leading whitespace must not be
    # normalized away. Reordered/dropped REM or changed/wrapped DFIX is currently
    # an unsupported output dialect, not recoverable evidence of an applied lock.
    if [line.rstrip("\r\n").rstrip() for line in lines[first:last + 1]] != expected:
        raise LockApplicationError("Managed instruction block differs from the reproduced application")
    _, _, hklf, _ = _scan(text)
    if last >= hklf or any(line.strip() for line in lines[last + 1:hklf]):
        raise LockApplicationError("Managed block was moved from its supported position before HKLF")
    return "".join(lines[:first] + lines[last + 1:])


def _canonical_token(token: str) -> str:
    return _render_number(_number(token)) if _NUMBER.fullmatch(token) else token.casefold()


def _scientific_records(text: str) -> tuple[tuple[str, ...], ...]:
    lines, atoms, hklf, _ = _scan(text)
    atom_lines = {atom.line_index for rows in atoms.values() for atom in rows}
    records = []
    index = 0
    while index <= hklf:
        line = lines[index].rstrip("\r\n")
        if not line or line[0].isspace():
            index += 1
            continue
        fragments = [_visible(line)]
        continuation = "=" in line.split("!", 1)[0]
        first_index = index
        tokens = fragments[0].split()
        if tokens and tokens[0].upper() == "REM":
            index += 1
            continue
        while continuation:
            index += 1
            fragment = lines[index].rstrip("\r\n")
            fragments.append(_visible(fragment))
            continuation = "=" in fragment.split("!", 1)[0]
        index += 1
        if first_index in atom_lines:
            continue
        tokens = " ".join(fragments).split()
        if not tokens:
            continue
        if tokens[0].upper() == "FVAR":
            values = [_number(token) for token in tokens[1:]]
            if not values or values[0] <= 0:
                raise LockApplicationError("Refinement output has invalid scale/free-variable fields")
            # Values are refinable parameters, not permission to add/remove
            # variables or change coordinate/occupancy free-variable encoding.
            records.append(("fvar", str(len(values))))
        else:
            records.append(tuple(_canonical_token(token) for token in tokens))
    return tuple(records)


def _all_dfix_records(text: str) -> tuple[tuple[str, ...], ...]:
    records = []
    for physical in text.splitlines():
        if not physical or physical[0].isspace():
            continue
        tokens = _visible(physical).split()
        if tokens and tokens[0].upper().split("_", 1)[0] == "DFIX":
            records.append(tuple(_canonical_token(token) for token in tokens))
    return tuple(records)


def _ordered_atoms(atoms: dict[str, list[_Atom]]) -> tuple[_Atom, ...]:
    if any(len(rows) != 1 for rows in atoms.values()):
        raise LockApplicationError("Output normalization requires unambiguous atom identities")
    return tuple(sorted((rows[0] for rows in atoms.values()), key=lambda atom: atom.line_index))


def _atom_parameters(lines: list[str], atom: _Atom) -> tuple[str, ...]:
    index = atom.line_index
    fragments = [_visible(lines[index].rstrip("\r\n"))]
    continuation = "=" in lines[index].split("!", 1)[0]
    while continuation:
        index += 1
        fragments.append(_visible(lines[index].rstrip("\r\n")))
        continuation = "=" in lines[index].split("!", 1)[0]
    fields = " ".join(fragments).split()
    if len(fields) not in {7, 12}:
        raise LockApplicationError("Only explicit SOF and isotropic or anisotropic atom rows can be normalized")
    for token in fields[2:]:
        _number(token)
    return tuple(fields)


def _same_parameter_freedom(before: str, after: str, *, isotropic_u: bool = False) -> bool:
    old, new = _number(before), _number(after)
    if isotropic_u and old < 0:
        return new == old  # A riding U multiplier must not become an independent U.
    if -5 < old < 5:
        return -5 < new < 5 and (not isotropic_u or new >= 0)
    return new == old  # Preserve fixed or free-variable encoded parameter constraints.


def normalize_managed_output(output_text: str, application: LockApplicationResult, *,
        baseline_ins_text: str, active_locks: Mapping[str, LockDirective],
        approved_hard_lock_ids: frozenset[str] = frozenset()) -> LockOutputNormalization:
    """Remove only exactly reproduced managed effects from a retained output.

    This narrow software-tested dialect requires an intact contiguous REM/DFIX
    block immediately before HKLF. Actual SHELXL may reformat or relocate records;
    such output is BLOCKED until that dialect has been independently verified.
    Changed scientific instructions (including WGHT), atom identity/order, SFAC
    indices and parameter freedom are rejected. Free sites and refinable FVAR,
    occupancy and displacement values retain their returned refinement values.
    Only newly approved fixed XYZ fields are decoded; pre-existing constraints
    are preserved. No periodic wrapping, coordinate tolerance or arbitrary
    constraint removal is performed. Appended REM diagnostics/Q peaks survive.
    """
    if not isinstance(application, LockApplicationResult) or not isinstance(output_text, str) or "\x00" in output_text:
        raise LockApplicationError("Output normalization requires a retained lock application and text")
    raw_digest = _hash(output_text)
    if len(output_text.encode("utf-8")) > _MAX_BASE_BYTES:
        raise LockApplicationError("Refinement output exceeds its normalization size bound")
    reproduced = apply_locks(baseline_ins_text, active_locks,
                             approved_hard_lock_ids=approved_hard_lock_ids)
    if application != reproduced or _hash(application.text) != application.output_sha256:
        raise LockApplicationError("Application records or baseline/instruction pins do not reproduce")
    unlocked = _strip_exact_managed_block(output_text, reproduced)
    if _scientific_records(unlocked) != _scientific_records(baseline_ins_text):
        raise LockApplicationError("Refinement output changed pre-existing scientific instructions")
    if _all_dfix_records(unlocked) != _all_dfix_records(baseline_ins_text):
        raise LockApplicationError("Additional or missing unmanaged DFIX instructions remain")
    baseline_lines, baseline_atoms, _, _ = _scan(baseline_ins_text)
    application_lines, application_atoms, _, _ = _scan(reproduced.text)
    output_lines, output_atoms, _, _ = _scan(unlocked)
    before_order = _ordered_atoms(baseline_atoms)
    after_order = _ordered_atoms(output_atoms)
    if [atom.label for atom in before_order] != [atom.label for atom in after_order]:
        raise LockApplicationError("Refinement output changed atom identity or ordering")
    hard_names = {name.casefold() for record in reproduced.records
                  if record.mode == LockMode.HARD_LOCK.value for name in record.atom_ids}
    decoded = []
    for original, returned in zip(before_order, after_order):
        staged = _resolve(original.label, application_atoms, require_bounded=original.label.casefold() in hard_names)
        before = _atom_parameters(baseline_lines, original)
        after = _atom_parameters(output_lines, returned)
        if int(before[1]) != int(after[1]) or len(before) != len(after):
            raise LockApplicationError("Refinement output changed atom scattering factors or displacement parameterization")
        if original.context_bounded != returned.context_bounded:
            raise LockApplicationError("Refinement output changed an atom's constraint context")
        replacements = []
        for old_token, staged_token, new_token in zip(original.coordinate_tokens, staged.coordinate_tokens,
                                                       returned.coordinate_tokens):
            old, staged_value, new = _number(old_token), _number(staged_token), _number(new_token)
            if old != staged_value:
                with localcontext() as context:
                    context.prec = 100
                    expected_fixed = Decimal(10) + old
                if (original.label.casefold() not in hard_names or not -5 < old < 5
                        or staged_value != expected_fixed or new != staged_value):
                    raise LockApplicationError("Managed hard coordinate was missing, recoded or drifted")
                replacements.append(old_token)
            else:
                if -5 < old < 5:
                    if not -5 < new < 5:
                        raise LockApplicationError("Output added an unapproved coordinate constraint")
                elif 5 < old < 15:
                    if new != old:
                        raise LockApplicationError("Output changed a pre-existing fixed coordinate")
                else:
                    raise LockApplicationError("Unsupported baseline coordinate/free-variable encoding")
                replacements.append(new_token)
        if not _same_parameter_freedom(before[5], after[5]):
            raise LockApplicationError("Output changed the pre-existing occupancy constraint")
        for old_token, new_token in zip(before[6:], after[6:]):
            if not _same_parameter_freedom(old_token, new_token, isotropic_u=len(before) == 7):
                raise LockApplicationError("Output changed the pre-existing displacement constraint")
        physical = output_lines[returned.line_index]
        changed = False
        for (start, end), token, previous in reversed(tuple(zip(returned.coordinate_spans, replacements,
                                                                returned.coordinate_tokens))):
            if token != previous:
                physical = physical[:start] + token + physical[end:]
                changed = True
        output_lines[returned.line_index] = physical
        if changed:
            decoded.append(original.label)
    normalized = "".join(output_lines)
    if any(_managed(line) for line in normalized.splitlines()):
        raise LockApplicationError("Managed audit instructions remain after normalization")
    return LockOutputNormalization(normalized, raw_digest, _hash(normalized), reproduced.baseline_sha256,
                                   reproduced.output_sha256, reproduced.applied_ids, tuple(decoded))
