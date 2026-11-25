"""Measured-file operations for an unvalidated CrystalX/SHELXL reconstruction cycle.

These operations never open a reference structure. Technical input/identity failures block;
missing chemical validation stays recorded for a separate checkCIF/review stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import math
import os
from pathlib import Path
import re
import shutil
import stat

from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.symmetry import parse_symm
from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix, u_equivalent
from crystalx_core.io.shelx_reader import logical_lines
from crystalx_core.io.shelx_res import ResFile, read_res
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.tools.adapter import ExternalToolAdapter
from crystalx_core.tools.runner import CancelToken, ResourceLimits
from crystalx_core.workspace.hashing import sha256_file
from crystalx_core.workflow.reconstruction import ReconstructionBlocked, digest


# Explicit chemical classes, not "all high-Z atoms": halogens must not become metal anchors.
METALS = frozenset("Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv".split())
_Q = re.compile(r"Q\d+\Z", re.IGNORECASE)
_MAX_BYTES = 32 * 1024 * 1024
# Real SHELXL-2025/1 prints six decimals after storing the encoded 10+p
# coordinate at single precision. Allow that bounded representation error only.
_FIXED_COORDINATE_PRINT_TOLERANCE = 0.000002


def shelx_continuation_rows(tokens):
    rows=[];row=tokens[0]
    for token in tokens[1:]:
        if len(row)+len(token)+1>76:
            rows.append(row+" =");row=" "+token
        else:row+=" "+token
    return [*rows,row]


def pinned_bytes(path: Path, expected_sha256: str, *, maximum: int = _MAX_BYTES) -> bytes:
    """Do not follow output aliases, including a file exchanged between lstat and open."""
    if path.parent.resolve(strict=True) != path.parent.absolute():
        raise ReconstructionBlocked("artifact_parent_must_be_canonical")
    named = path.lstat()
    if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or named.st_size > maximum:
        raise ReconstructionBlocked("artifact_must_be_bounded_unaliased_regular_file")
    fd = os.open(path, os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
    with os.fdopen(fd,"rb") as source:
        opened=os.fstat(source.fileno())
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if identity(named)!=identity(opened):raise ReconstructionBlocked("artifact_changed_during_open")
        data=source.read(maximum+1)
        after=os.fstat(source.fileno());retained=path.lstat()
    if (len(data)>maximum or identity(named)!=identity(after) or identity(named)!=identity(retained)
            or not stat.S_ISREG(retained.st_mode) or retained.st_nlink!=1
            or hashlib.sha256(data).hexdigest()!=expected_sha256):
        raise ReconstructionBlocked("artifact_digest_or_file_identity_changed")
    return data


def cell_values(cell):
    return (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)


def displacement_value(model: ResFile, atom) -> float | None:
    """Scalar Uiso or cell-metric Ueq; the raw anisotropic tensor is retained."""
    value = atom.u_iso if len(atom.u_values)==1 else u_equivalent(cell_values(model.header.cell),atom.u_values)
    return value if value is not None and math.isfinite(value) else None


def promote_initial_peaks(source: Path, destination: Path, *, source_program: str | None = None) -> dict:
    """Unknown automatic phase maxima become provisional descriptors, never known carbon atoms."""
    parsed = read_res(source)
    atom_ids={atom.label for atom in parsed.atoms}
    density_rows={}
    if source_program=="shelxt":
        for line in logical_lines(source):
            if line.tokens and line.tokens[0] in atom_ids and len(line.tokens)==8:
                try:
                    height=float(line.tokens[7]);u=float(line.tokens[6])
                except ValueError:
                    raise ReconstructionBlocked("invalid_shelxt_typed_peak_height")
                if not math.isfinite(height) or not math.isfinite(u):
                    raise ReconstructionBlocked("nonfinite_shelxt_typed_peak_height")
                density_rows[line.tokens[0]]={"height":height,"u_iso":u}
    hklf_line = next((line.first_line_no for line in logical_lines(source) if line.instruction == "HKLF"), None)
    if not parsed.header.cell or hklf_line is None:
        raise ReconstructionBlocked("automatic_seed_cell_and_hklf_required")
    peak_lines = {line.tokens[0] for line in logical_lines(source)
                  if line.tokens and _Q.fullmatch(line.tokens[0]) and line.first_line_no < hklf_line}
    selected = [peak for peak in parsed.q_peaks if peak.label in peak_lines or not parsed.atoms]
    if not selected and not density_rows:
        shutil.copyfile(source, destination)
        return {"promoted_peaks": [], "placeholder_is_an_element_assignment": False}
    sfac = list(parsed.header.sfac)
    appended = "C" not in sfac
    if appended:
        sfac.append("C")
    labels = {atom.label.casefold() for atom in parsed.atoms}
    rows, report = [], []
    for peak in selected:
        index = 1
        while f"D{index:03d}".casefold() in labels:
            index += 1
        if index > 999:
            raise ReconstructionBlocked("automatic_untyped_site_label_budget_exceeded")
        name = f"D{index:03d}"; labels.add(name.casefold())
        rows.append(f"{name} {sfac.index('C')+1} {peak.x:.8f} {peak.y:.8f} {peak.z:.8f} 11 0.05")
        report.append({"source_peak":peak.label,"site_id":name,"descriptor":"C_placeholder_not_confirmed_element"})
    output = []
    for line in logical_lines(source):
        if line.instruction == "END":
            break
        if line.tokens and _Q.fullmatch(line.tokens[0]):
            continue
        if line.tokens and line.tokens[0] in density_rows:
            output.append(" ".join(line.tokens[:7]))
        elif line.instruction == "SFAC":
            output.append("SFAC " + " ".join(sfac))
        elif line.instruction == "UNIT" and appended:
            output.append(line.text+" 0")
        elif line.instruction == "HKLF":
            output.extend(rows);output.extend(line.raw_lines);break
        else:
            output.extend(line.raw_lines)
    destination.write_text("\n".join(output)+"\nEND\n")
    return {"promoted_peaks":report,"placeholder_is_an_element_assignment":False,
            "shelxt_typed_peak_heights":density_rows,"normalization":"retain_u_iso_and_record_trailing_density_height"}


def checked_res(path: Path, *, allow_anisotropic: bool = False) -> ResFile:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ReconstructionBlocked("model_must_be_a_bounded_regular_file")
    parsed = read_res(path)
    if (not parsed.atoms or not parsed.header.cell or not parsed.header.hklf_format
            or parsed.sha256 != sha256_file(path)):
        raise ReconstructionBlocked("model_parse_or_input_identity_failed")
    labels = [a.label.casefold() for a in parsed.atoms]
    if len(set(labels)) != len(labels):
        raise ReconstructionBlocked("residue_scoped_or_duplicate_labels_require_a_separate_adapter")
    for atom in parsed.atoms:
        if (atom.part not in (None, 0) or atom.residue is not None or atom.afix not in (None, 0)
                or len(atom.u_values) not in ((1,6) if allow_anisotropic else (1,)) or not 1 <= atom.sfac_index <= len(parsed.header.sfac)
                or any(not math.isfinite(x) for x in (atom.x, atom.y, atom.z, atom.occupancy_code, *atom.u_values))
                or any(not -5 < x < 5 for x in (atom.x, atom.y, atom.z))):
            raise ReconstructionBlocked("unsupported_disorder_anisotropic_or_encoded_coordinate_seed")
    return parsed


def setting_key(model: ResFile, hkl_sha256: str) -> str:
    # WGHT and scale are refined parameters/settings; CELL/SYMM/data masks cannot drift silently.
    commands = {"CELL", "LATT", "SYMM", "HKLF", "TWIN", "BASF", "MERG", "OMIT", "SHEL"}
    records=[]
    def canonical(token):
        try:
            number=Decimal(token)
            if not number.is_finite():raise ReconstructionBlocked("nonfinite_reflection_setting")
            return str(number.normalize())
        except InvalidOperation:
            return token.upper()
    for line in logical_lines(model.source):
        if line.instruction not in commands:continue
        if line.instruction=="BASF":
            records.append(["BASF_PARAMETER_COUNT",len(line.tokens)-1])
        elif line.instruction=="SYMM":
            operation=parse_symm("".join(line.tokens[1:]))
            if operation is None:raise ReconstructionBlocked("unreadable_refinement_symmetry")
            records.append(["SYMM",operation.rotation,tuple(str(value%1) for value in operation.translation)])
        else:
            records.append([canonical(token) for token in line.tokens])
    return digest((hkl_sha256, records))


def periodic_distance(cell, first, second) -> float:
    """Exact closest lattice image for an ordinary cell, with an explicit enumeration cap."""
    m = orthogonalisation_matrix(tuple(cell))
    # Upper-triangular inverse of the standard orthogonalisation matrix.
    a, b, c, d, e, f = m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2]
    inverse = ((1/a, -b/(a*d), (b*e-c*d)/(a*d*f)), (0., 1/d, -e/(d*f)), (0., 0., 1/f))
    delta = tuple(second[i] - first[i] for i in range(3))
    center = tuple(-round(x) for x in delta)
    upper = math.dist((0., 0., 0.), frac_to_cart(m, tuple(delta[i] + center[i] for i in range(3))))
    extents = tuple(upper * math.sqrt(sum(v*v for v in row)) + 1e-10 for row in inverse)
    ranges = [range(math.ceil(-delta[i] - extents[i]), math.floor(-delta[i] + extents[i]) + 1)
              for i in range(3)]
    if math.prod(len(r) for r in ranges) > 50_000:
        raise ReconstructionBlocked("skew_cell_periodic_search_budget_exceeded")
    return min((math.dist((0., 0., 0.), frac_to_cart(m, tuple(delta[i]+shift[i] for i in range(3))))
                for shift in itertools.product(*ranges)), default=upper)


def important_movement(previous: ResFile, current: ResFile, site_ids: tuple[str, ...]) -> float | None:
    before, after = {a.label: a for a in previous.atoms}, {a.label: a for a in current.atoms}
    if not site_ids or any(key not in before or key not in after for key in site_ids):
        return None
    if previous.header.cell != current.header.cell:
        return None
    cell = cell_values(previous.header.cell)
    return max(periodic_distance(cell, (before[key].x, before[key].y, before[key].z),
                                 (after[key].x, after[key].y, after[key].z)) for key in site_ids)


def important_sites(model: ResFile, explicit: tuple[str, ...] = ()) -> tuple[str, ...]:
    labels = {a.label for a in model.atoms}
    if explicit:
        if len(set(explicit)) != len(explicit) or not set(explicit) <= labels:
            raise ReconstructionBlocked("important_atom_selection_is_not_in_initial_model")
        return explicit
    heavy = tuple(a.label for a in model.atoms if ATOMIC_NUMBERS[model.element_of(a)] >= 20)
    return heavy or tuple(a.label for a in model.atoms if model.element_of(a) != "H")


def heavy_only_seed(model: ResFile, destination: Path, elements: tuple[str, ...] = ()) -> tuple[str, ...]:
    selected = set(elements) if elements else METALS
    if not selected <= METALS:
        raise ReconstructionBlocked("heavy_only_anchor_elements_must_be_explicit_metals")
    retained = tuple(a.label for a in model.atoms if model.element_of(a) in selected)
    if not retained:
        raise ReconstructionBlocked("no_requested_metal_sites_in_automatic_initial_model")
    all_labels = {a.label for a in model.atoms}
    unsupported = {"PART", "RESI", "AFIX", "HFIX", "DFIX", "DANG", "SADI", "SAME", "EXYZ", "EADP",
                   "SUMP", "CHIV", "FLAT", "DELU", "SIMU", "RIGU", "ISOR"}
    output = []
    for line in logical_lines(model.source):
        if line.instruction == "END":
            break
        if line.instruction in unsupported:
            raise ReconstructionBlocked("heavy_only_pruning_requires_an_unconstrained_initial_model")
        label = line.tokens[0] if line.tokens else ""
        if label in all_labels and label not in retained or _Q.fullmatch(label):
            continue
        output.extend(line.raw_lines)
        if line.instruction == "HKLF":
            break
    destination.write_text("\n".join(output) + "\nEND\n")
    checked_res(destination)
    return retained


def inference_input(model: ResFile, destination: Path, *, max_new_peaks: int,
                    min_peak_height: float, min_distance: float,
                    selected_peak_ids: tuple[str,...] | None = None,
                    excluded_peak_ids:tuple[str,...]=(),decision_records:list|None=None) -> tuple[str, ...]:
    selected = []
    cell = cell_values(model.header.cell)
    for peak in sorted(model.q_peaks, key=lambda p: (-p.height, p.label)):
        row={"peak_id":peak.label,"height":peak.height,"decision":"reject"}
        def reject(reason):
            if decision_records is not None:decision_records.append({**row,"reason":reason})
        if peak.label in excluded_peak_ids:
            reject("metal_ripple_exclusion");continue
        if selected_peak_ids is not None and peak.label not in selected_peak_ids:
            reject("outside_explicit_growth_selection")
            continue
        if len(selected) >= max_new_peaks:
            if decision_records is None:break
            reject("per_round_peak_budget");continue
        position = (peak.x, peak.y, peak.z)
        if not math.isfinite(peak.height) or peak.height < min_peak_height or any(not math.isfinite(x) for x in position):
            reject("absolute_height_or_nonfinite_coordinate")
            continue
        others = [(a.x, a.y, a.z) for a in model.atoms] + [(p.x, p.y, p.z) for p in selected]
        if any(periodic_distance(cell, xyz, position) < min_distance for xyz in others):
            reject("too_close_to_existing_or_selected_site")
            continue
        selected.append(peak)
        if decision_records is not None:decision_records.append({**row,"decision":"select","reason":"legacy_height_and_distance_supported"})
    text = []
    for line in logical_lines(model.source):
        if line.instruction == "END":
            break
        if line.tokens and _Q.fullmatch(line.tokens[0]):
            continue
        text.extend(line.raw_lines)
        if line.instruction == "HKLF":
            break
    text.append("END")
    text.extend(f"{p.label} 1 {p.x:.8f} {p.y:.8f} {p.z:.8f} 11 0.05 {p.height:.8f}" for p in selected)
    destination.write_text("\n".join(text) + "\n")
    return tuple(p.label for p in selected)


def assign_to_model(model: ResFile, predictions, destination: Path, *,
                    candidate_elements: tuple[str, ...], locked_sites: tuple[str, ...] = (),
                    minimum_probability: float = 0.0,
                    element_overrides: dict[str,str] | None = None,
                    reserved_site_ids:tuple[str,...]=(),allow_anisotropic_metals:bool=False) -> dict:
    sites = tuple(a.label for a in model.atoms) + tuple(p.label for p in model.q_peaks)
    by_id = {p.site_id: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != set(sites):
        raise ReconstructionBlocked("inference_output_sites_do_not_match_exact_model_and_selected_peaks")
    atomic_to_symbol = {z: element for element, z in ATOMIC_NUMBERS.items()}
    choices, uncertainty = {}, {}
    for key in sites:
        top = by_id[key].topk
        if (len(top) < 2 or any(p.atomic_number not in [ATOMIC_NUMBERS[e] for e in candidate_elements]
                              or not math.isfinite(p.probability) or not 0 <= p.probability <= 1 for p in top)
                or any(top[i].probability < top[i+1].probability for i in range(len(top)-1))):
            raise ReconstructionBlocked("invalid_pretrained_element_alternatives")
        choices[key] = atomic_to_symbol[top[0].atomic_number]
        uncertainty[key] = [{"element": atomic_to_symbol[p.atomic_number], "probability": p.probability} for p in top]
    existing = {a.label: a for a in model.atoms}
    if element_overrides:
        if (not set(element_overrides)<=set(sites)
                or any(e not in candidate_elements and e!="H" for e in element_overrides.values())):
            raise ReconstructionBlocked("invalid_explicit_assignment_policy_override")
        choices.update(element_overrides)
    # The loaded head classifies non-H types; it must not turn an existing hydrogen into carbon.
    for key, atom in existing.items():
        if model.element_of(atom) == "H":
            choices[key] = "H"
    for key in locked_sites:
        choices[key] = model.element_of(existing[key])
    sfac = list(model.header.sfac)
    # The hypothesis set may be broad; register only types actually used by an
    # accepted assignment, without adding unused zero-content chemical species.
    used_elements={element for key,element in choices.items()
                   if key in locked_sites or by_id[key].topk[0].probability>=minimum_probability}
    for element in candidate_elements:
        if element not in used_elements:continue
        if element not in sfac:
            sfac.append(element)
    def instruction_rows(tokens):
        rows=[];row=tokens[0]
        for token in tokens[1:]:
            if len(row)+len(token)+1>76:
                rows.append(row+" =");row=" "+token
            else:row+=" "+token
        return [*rows,row]
    changes, output, injected = [], [], False
    for line in logical_lines(model.source):
        if line.instruction == "END":
            break
        tokens = line.tokens
        if line.instruction == "SFAC":
            if injected:
                raise ReconstructionBlocked("multiple_sfac_instructions_need_an_explicit_adapter")
            output.extend(instruction_rows(["SFAC",*sfac])); injected = True
        elif tokens and tokens[0] in existing:
            if (len(tokens) != 7 and not (len(tokens)==12 and allow_anisotropic_metals
                    and model.element_of(existing[tokens[0]]) in METALS)):
                raise ReconstructionBlocked("element_assignment_supports_isotropic_initial_models_only")
            key = tokens[0]
            element = choices[key] if by_id[key].topk[0].probability >= minimum_probability or key in locked_sites else model.element_of(existing[key])
            if element != model.element_of(existing[key]):
                changes.append({"site_id": key, "from": model.element_of(existing[key]), "to": element})
            tokens[1] = str(sfac.index(element)+1)
            if len(tokens)==12 and element not in METALS:
                value=displacement_value(model,existing[key])
                if value is None:raise ReconstructionBlocked("anisotropic_retyped_site_has_no_finite_ueq")
                tokens=tokens[:6]+[f"{value:.8f}"]
            output.extend(instruction_rows(tokens))
        elif line.instruction == "HKLF":
            used = {key.casefold() for key in (*existing,*reserved_site_ids)}
            for peak in model.q_peaks:
                if by_id[peak.label].topk[0].probability < minimum_probability:
                    continue
                index = 1
                while f"X{index:03d}".casefold() in used:
                    index += 1
                if index > 999:
                    raise ReconstructionBlocked("generated_site_label_budget_exceeded")
                label = f"X{index:03d}"; used.add(label.casefold())
                element = choices[peak.label]
                output.append(f"{label} {sfac.index(element)+1} {peak.x:.8f} {peak.y:.8f} {peak.z:.8f} 11 0.05")
                changes.append({"site_id": label, "peak_id": peak.label, "from": "Q_PEAK", "to": element})
            output.extend(line.raw_lines)
            break
        elif tokens and _Q.fullmatch(tokens[0]):
            continue
        elif line.instruction == "UNIT" and len(sfac) != len(model.header.sfac):
            # Append explicit zero counts for newly permitted types; never infer composition.
            output.extend(instruction_rows([*line.tokens,*(["0"]*(len(sfac)-len(model.header.sfac)))]))
        else:
            output.extend(line.raw_lines)
    if not injected:
        raise ReconstructionBlocked("sfac_required_for_element_assignment")
    destination.write_text("\n".join(output) + "\nEND\n")
    checked_res(destination,allow_anisotropic=allow_anisotropic_metals)
    return {"changes": changes, "alternatives": uncertainty, "probabilities_calibrated": False,
            "geometry_checked": False, "full_map_used": False,
            "candidate_elements":candidate_elements,"new_sfac_types":tuple(sfac[len(model.header.sfac):]),
            "unit_new_type_counts":"zero_placeholders_not_inferred_composition"}


def prune_model(model:ResFile,destination:Path,site_ids:tuple[str,...],*,protected_site_ids:tuple[str,...]=(),
                allow_anisotropic_metals:bool=False) -> dict:
    """Write a reversible derivative; the immutable source contains every removed row."""
    selected=set(site_ids);known={a.label for a in model.atoms}
    if any(a.is_anisotropic and (not allow_anisotropic_metals or model.element_of(a) not in METALS) for a in model.atoms):
        raise ReconstructionBlocked("pruning_anisotropic_model_requires_explicit_metal_policy")
    if (len(selected)!=len(site_ids) or not selected<=known or selected&set(protected_site_ids)
            or len(selected)>=len(known)):
        raise ReconstructionBlocked("invalid_or_empty_model_pruning_proposal")
    removed=[];rows=[]
    for line in logical_lines(model.source):
        if selected and line.instruction in {"DFIX","DANG","SADI","SAME","FLAT","CHIV","EXYZ","EADP","BIND","FREE","SUMP"}:
            raise ReconstructionBlocked("pruning_requires_flat_unrestrained_model")
        if line.tokens and line.tokens[0] in selected:
            removed.append({"site_id":line.tokens[0],"raw_lines":line.raw_lines,"line_number":line.first_line_no})
        else:rows.extend(line.raw_lines)
    destination.write_text("\n".join(rows)+"\n")
    checked_res(destination,allow_anisotropic=allow_anisotropic_metals)
    return {"source_sha256":model.sha256,"source_path":str(model.source),
            "derivative_sha256":sha256_file(destination),"removed_sites":removed,
            "rollback":"retain_previous_immutable_model_and_policy_state","unit_counts_recomputed":False}


@dataclass(frozen=True)
class FileRefinementResult:
    model_path: Path
    raw_res_path: Path
    cif_path: Path | None
    fcf_path: Path | None
    r1: float
    wr2: float | None
    goof: float | None
    n_reflections: int | None
    n_parameters: int | None
    invocation: object
    artifact_sha256s: tuple[tuple[str,str], ...] = ()
    method: str = "ls"
    fourier_map: int = 2


def refine_file(*, model_path: Path, hkl_path: Path, directory: Path, adapter: ExternalToolAdapter,
                cycles: int, timeout_s: float, cancel: CancelToken, plan_peaks: int = 20,
                method: str = "ls", damp_factor: float | None = None, damp_shift_esd_limit: float = 15.0,
                locked_sites: tuple[str, ...] = (), limits: ResourceLimits | None = None,
                isotropic: bool = True,fourier_map:int=2,anisotropic_metals:bool=False) -> FileRefinementResult:
    from crystalx_core.io.shelxl_lst import parse_lst_text
    directory.mkdir(parents=True, exist_ok=False)
    if type(anisotropic_metals) is not bool or anisotropic_metals and not isotropic:
        raise ValueError("choose global or metal-only anisotropic refinement explicitly")
    source = checked_res(model_path,allow_anisotropic=not isotropic or anisotropic_metals)
    if anisotropic_metals and any(a.is_anisotropic and source.element_of(a) not in METALS for a in source.atoms):
        raise ReconstructionBlocked("metal_only_anisotropic_policy_received_anisotropic_nonmetal")
    metal_ids=tuple(a.label for a in source.atoms if source.element_of(a) in METALS) if anisotropic_metals else ()
    metal_elements=tuple(sorted({source.element_of(a) for a in source.atoms if a.label in metal_ids}))
    original_hkl = sha256_file(hkl_path)
    initial_only={"TREF","PATT","TEXP","FIND","PLOP","NTRY","PATS","GROP"}
    source_text="\n".join(physical for line in logical_lines(model_path)
                            if line.instruction not in initial_only for physical in line.raw_lines)+"\n"
    text, changes, _, _ = prepare_refinement_ins(source_text, RefinementSettings(cycles=cycles, plan_peaks=plan_peaks, method=method,isotropic=isotropic,fourier_map=fourier_map,
        damp_factor=damp_factor, damp_shift_esd_limit=damp_shift_esd_limit))
    locked = {a.label: a for a in source.atoms if a.label in locked_sites}
    if len(locked) != len(locked_sites):
        raise ReconstructionBlocked("anchor_label_missing_before_refinement")
    from crystalx_core.refinement.shelxl import _logical_from_text
    lines = [];selectors_inserted=False;source_ids={a.label for a in source.atoms}
    for logical in _logical_from_text(text.splitlines()):
        row = logical.text.split()
        if row and row[0] in source_ids and not selectors_inserted:
            # Explicit SFAC selectors precede the atoms; no blanket ANIS can
            # accidentally convert light atoms. Flat, unscoped models only.
            lines.extend("ANIS $"+element.upper() for element in metal_elements)
            selectors_inserted=True
        if row and row[0] in locked:
            row[2:5] = [f"{10+float(value):.8f}" for value in row[2:5]]
            lines.extend(shelx_continuation_rows(row))
        else:lines.extend(logical.raw_lines)
    ins, hkl = directory/"trial.ins", directory/"trial.hkl"
    ins.write_text("\n".join(lines)+"\n"); shutil.copyfile(hkl_path, hkl); hkl.chmod(0o444)
    ins_sha = sha256_file(ins)
    # The vendor's real banner documents default -b3000. Size the bound for an
    # isotropic model before ligand growth exceeds it; this changes no observations.
    parameter_bound=(4 if isotropic else 9)*len(source.atoms)+5*len(metal_ids)
    matrix_bound=max(3000,500*math.ceil((parameter_bound+100)/500))
    if method == "ls" and matrix_bound>12000:
        raise ReconstructionBlocked("full_matrix_parameter_resource_bound_exceeded")
    arguments=("trial",) if method == "cgls" or matrix_bound==3000 else ("trial",f"-b{matrix_bound}")
    plan = adapter.build_plan(args=arguments, cwd=directory, timeout_s=timeout_s,
        expected_outputs=("trial.res", "trial.lst"), declared_inputs=("trial.ins", "trial.hkl"),
        label="pretrained-reconstruction-shelxl", limits=limits)
    outcome, record = adapter.invoke(plan, log_dir=directory/"logs", cancel=cancel)
    from crystalx_core._serde import dumps
    (directory/"invocation.json").write_text(dumps(record))
    (directory/"instruction-changes.json").write_text(dumps({"changes":changes,
        "refinement_method":method,"isotropic":isotropic,"fourier_map":fourier_map,
        "plan_peaks":plan_peaks,"anisotropic_metals":anisotropic_metals,"anisotropic_metal_site_ids":metal_ids,
        "anisotropic_metal_selectors":["$"+e.upper() for e in metal_elements],"damp_factor_requested":damp_factor,
        "damp_shift_esd_limit_requested":damp_shift_esd_limit,
        "effective_damp_instructions":[line for line in ins.read_text().splitlines() if line.upper().startswith("DAMP ")],
        "damp_limit_is_shift_esd_not_angstrom":True,"damp_limit_applies":method=="ls",
        "fixed_coordinate_sites":locked_sites,"anchor_mode":"user_requested_metal_scaffold",
        "fixed_coordinate_print_tolerance_fractional":_FIXED_COORDINATE_PRINT_TOLERANCE,
        "fixed_coordinate_normalization":"restore_exact_input_after_verifying_raw_encoded_output",
        "phase_policy":"recomputed_from_each_partial_model_not_fixed_experimental_phases"}))
    if sha256_file(ins) != ins_sha or sha256_file(hkl) != original_hkl:
        raise ReconstructionBlocked("refinement_modified_declared_inputs")
    if cancel.cancelled:
        raise ReconstructionBlocked("cancelled")
    if not outcome.succeeded:
        raise ReconstructionBlocked("shelxl_process_failed:"+record.diagnostic)
    raw = directory/"trial.res"
    retained_outputs={}
    for name in ("trial.res","trial.lst","trial.cif","trial.fcf"):
        path=directory/name
        if name in ("trial.res","trial.lst") or path.exists() or path.is_symlink():
            expected=record.output_hashes.get(name)
            if not expected:raise ReconstructionBlocked("consumed_output_missing_from_invocation_evidence")
            retained_outputs[name]=pinned_bytes(path,expected,maximum=128*1024*1024 if name.endswith(".fcf") else _MAX_BYTES)
    listing = parse_lst_text(retained_outputs["trial.lst"].decode("utf-8",errors="replace"),source=directory/"trial.lst")
    if listing.has_fatal_error:
        raise ReconstructionBlocked("shelxl_listing_reports_fatal_error:" + "; ".join(listing.errors))
    # Decode only the exact hard-coordinate effects applied to this invocation. Raw bytes stay.
    output = []
    for logical in _logical_from_text(retained_outputs["trial.res"].decode("utf-8").splitlines()):
        row = logical.text.split()
        if row and row[0] in locked:
            if len(row) not in (7,12):
                raise ReconstructionBlocked("unexpected_anchor_parameterization")
            expected = locked[row[0]]
            values = [float(x) for x in row[2:5]]
            originals=(expected.x,expected.y,expected.z)
            if any(not math.isclose(value,10+original,rel_tol=0.,abs_tol=_FIXED_COORDINATE_PRINT_TOLERANCE)
                   for value,original in zip(values,originals)):
                raise ReconstructionBlocked("fixed_metal_coordinate_drift_or_encoding_changed")
            row[2:5] = [f"{value:.8f}" for value in originals]
            output.extend(shelx_continuation_rows(row))
        else:
            output.extend(logical.raw_lines)
    normalized = directory/"model.res"; normalized.write_text("\n".join(output)+"\n")
    returned = checked_res(normalized,allow_anisotropic=not isotropic or anisotropic_metals)
    if anisotropic_metals and any(a.is_anisotropic and returned.element_of(a) not in METALS for a in returned.atoms):
        raise ReconstructionBlocked("metal_only_anisotropic_policy_produced_anisotropic_nonmetal")
    if ([(a.label,returned.element_of(a)) for a in returned.atoms]
            != [(a.label,source.element_of(a)) for a in source.atoms]
            or setting_key(returned,original_hkl) != setting_key(source,original_hkl)):
        raise ReconstructionBlocked("refinement_changed_sites_elements_or_reflection_setting")
    r1 = returned.r_factors.r1_observed if returned.r_factors.r1_observed is not None else listing.r1_observed
    if r1 is None or not math.isfinite(r1) or not 0 <= r1 <= 1:
        raise ReconstructionBlocked("r1_missing_or_invalid_no_stability_assessment_possible")
    if (returned.r_factors.r1_observed is not None and listing.r1_observed is not None
            and abs(returned.r_factors.r1_observed-listing.r1_observed)>1e-5):
        raise ReconstructionBlocked("res_listing_r1_disagreement")
    for name,data in retained_outputs.items():
        if pinned_bytes(directory/name,record.output_hashes[name],maximum=128*1024*1024 if name.endswith(".fcf") else _MAX_BYTES)!=data:
            raise ReconstructionBlocked("output_changed_while_parsing")
    return FileRefinementResult(normalized,raw,directory/"trial.cif" if (directory/"trial.cif").is_file() else None,
        directory/"trial.fcf" if (directory/"trial.fcf").is_file() else None,float(r1),listing.wr2,
        listing.goof,listing.n_all,listing.parameters_refined,record,
        tuple([(name,record.output_hashes[name]) for name in retained_outputs]+[("model.res",sha256_file(normalized))]),method,fourier_map)


def finalize_cgls_exports(**kwargs):
    """Refuse an unapproved terminal model change or unsupported zero-shift ACTA.

    Local SHELXL rejects ACTA with both DAMP 0 0 and L.S. 0. A real final LS
    step must be budgeted and assessed as refinement, not disguised as export.
    """
    raise ReconstructionBlocked("cgls_native_cif_requires_recorded_nonzero_ls_refinement")
