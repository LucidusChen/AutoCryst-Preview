"""Opt-in heavy-stage recipe; upstream numerical parity remains a separate audit.

Only initial SHELXT atom rows are main sites. The context follows get_equiv_pos2's
3.2/0.2-Angstrom recipe and fixed [-2,2] translations. Gemmi supplies the orbit;
its ordering/roundoff must still be compared with cctbx after M0 installation.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from itertools import product
import math
import textwrap
from pathlib import Path

from crystalx_core.app.inference_protocols import pinned_upstream_helper
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS, atomic_number
from crystalx_core.domain.inference import ElementInferenceRequest
from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix
from crystalx_core.io.shelx_reader import logical_lines
from crystalx_core.io.shelx_res import read_res
from crystalx_core.workflow.reconstruction import ReconstructionBlocked, digest
from crystalx_core.workspace.hashing import sha256_file


@dataclass(frozen=True)
class HeavyPreparation:
    request: ElementInferenceRequest
    retained_site_ids: tuple[str, ...]
    trimmed_site_ids: tuple[str, ...]
    source_sha256: str
    provenance: dict


def prepare_upstream_heavy(model_path: Path, *, source_root: Path,
                           max_image_candidates: int = 5_000_000,
                           max_sites: int = 20_000) -> HeavyPreparation:
    import numpy as np
    from scipy.spatial import cKDTree
    from crystalx_core.app.reconstruction_runtime_inference import _complete_symmetry

    path = Path(model_path).absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024*1024:
        raise ReconstructionBlocked('upstream_heavy_requires_bounded_regular_initial_model')
    parsed = read_res(path)
    if len(parsed.atoms)>max_sites:
        raise ReconstructionBlocked('upstream_heavy_main_site_budget_exceeded')
    if not parsed.header.cell or not parsed.atoms or len({a.label for a in parsed.atoms}) != len(parsed.atoms):
        raise ReconstructionBlocked('upstream_heavy_requires_cell_and_unique_atom_labels')
    if any(a.part not in (None,0) or a.residue is not None or a.afix not in (None,0)
           or len(a.u_values) not in (1,2) or any(not math.isfinite(v) or not -5<v<5 for v in (a.x,a.y,a.z))
           for a in parsed.atoms):
        raise ReconstructionBlocked('upstream_heavy_initial_model_parameterization_unsupported')
    sfac = parsed.header.sfac
    unit = tuple(int(float(n)) for n in parsed.header.unit)
    if len(sfac) != len(unit) or not sfac or sum(n for e,n in zip(sfac,unit) if e!='H') <= 0:
        raise ReconstructionBlocked('upstream_heavy_requires_explicit_positive_sfac_unit')
    count = len(parsed.atoms)
    unit_total = sum(n for e,n in zip(sfac,unit) if e!='H')
    while count and count % unit_total and unit_total % count:
        count -= 1
    atoms = parsed.atoms[:count]
    if not atoms:raise ReconstructionBlocked('upstream_heavy_trim_left_no_atoms')
    # load_shelxt uses the alphabetic label prefix, not the SFAC index. Preserve
    # the upstream B/FE context corrections (including their case-sensitive SFAC).
    symbols = [a.element_prefix for a in parsed.atoms]
    raw_sfac = next((line.tokens[1:] for line in logical_lines(path) if line.instruction=='SFAC'),())
    if 'B' in raw_sfac and 'B' not in symbols:symbols[-1]='B'
    if 'FE' in raw_sfac and 'FE' not in symbols:
        for i,symbol in enumerate(symbols):
            if symbol in {'FE','CO','NI','CU','ZN'} and (symbols[i+1] if i+1<len(symbols) else None) not in {'FE','CO','NI','CU','ZN'}:
                symbols[i]='FE';break
    symbols=symbols[:count]
    try:label_z = [atomic_number(e.capitalize()) for e in symbols]
    except (ValueError,KeyError) as error:
        raise ReconstructionBlocked('upstream_heavy_requires_shelxt_element_prefix_labels') from error
    matrix = orthogonalisation_matrix(tuple(getattr(parsed.header.cell,k) for k in ('a','b','c','alpha','beta','gamma')))
    main_cart = [frac_to_cart(matrix,(a.x,a.y,a.z)) for a in atoms]
    if len(set(main_cart)) != len(main_cart):
        raise ReconstructionBlocked('upstream_heavy_duplicate_asu_positions_make_mask_ambiguous')
    operators,symmetry_source = _complete_symmetry(parsed)
    orbit=[]; orbit_z=[]
    for atom,z in zip(atoms,label_z):
        seen=set()
        for op in operators:
            xyz=(atom.x,atom.y,atom.z)
            frac=tuple((sum(op.rotation[i][j]*xyz[j] for j in range(3))+float(op.translation[i]))%1 for i in range(3))
            # A special-position orbit has one representative per exact image.
            if frac not in seen:seen.add(frac);orbit.append(frac);orbit_z.append(z)
    if len(orbit)*125 > max_image_candidates:
        raise ReconstructionBlocked('upstream_heavy_image_candidate_budget_exceeded')
    tree=cKDTree(np.asarray(main_cart,dtype=np.float64))
    near=[[] for _ in atoms]
    for translation in product((-2,-1,0,1,2),repeat=3):
        positions=[frac_to_cart(matrix,tuple(f[i]+translation[i] for i in range(3))) for f in orbit]
        for pos,z,neighbours in zip(positions,orbit_z,tree.query_ball_point(positions,3.2)):
            if not neighbours:continue
            distances=[(i,math.dist(pos,main_cart[i])) for i in neighbours]
            if min(d for _,d in distances)<=0.2:continue
            for i,d in distances:
                if d<3.2:near[i].append((pos,z))
    # np.where(cdist(main,images)<radius)[1] is ordered by ASU row then image.
    # Exact Cartesian de-duplication preserves its first occurrence and label.
    positions=list(main_cart); z=list(label_z); seen=set(positions)
    for neighbours in near:
        for pos,element in neighbours:
            if pos not in seen:
                if len(positions)>=max_sites:raise ReconstructionBlocked('upstream_heavy_graph_site_budget_exceeded')
                seen.add(pos);positions.append(pos);z.append(element)
    allocation=pinned_upstream_helper(source_root,'build_sorted_init_z_from_ratio')
    z[:count]=allocation(sfac,unit,count)
    candidates=tuple(sorted(set(z)))
    if len(candidates)<2 or any(n<=1 for n in candidates):
        raise ReconstructionBlocked('upstream_heavy_requires_two_nonhydrogen_descriptor_types')
    ids=tuple(a.label for a in atoms)
    context_ids=tuple(f'upstream-context-{i}' for i in range(len(z)-count))
    request=ElementInferenceRequest(ids+context_ids,tuple(z),tuple(positions),candidates,count)
    if sha256_file(path)!=parsed.sha256:raise ReconstructionBlocked('upstream_heavy_initial_model_changed')
    provenance={'preparation':'upstream_heavy_v1','main_site_source':'initial_SHELXT_atoms_in_file_order',
        'q_peaks_included':False,'context_radius_angstrom':3.2,'asu_image_exclusion_angstrom':0.2,
        'translation_components':[-2,-1,0,1,2],'deduplication':'exact_Cartesian_first_occurrence',
        'context_descriptors':'SHELXT_label_prefix_with_upstream_B_FE_corrections',
        'sfac_unit_source':'initial_SHELXT_RES','non_h_unit_total':unit_total,
        'trimmed_site_ids':[a.label for a in parsed.atoms[count:]],'retained_site_ids':ids,
        'initialization':'SFAC_UNIT_ratio_descending_Z_in_SHELXT_order',
        'candidate_source':'unique_all_graph_descriptors','forced_coverage':True,
        'preparation_implementation_sha256':sha256_file(Path(__file__)),
        'symmetry_source':symmetry_source,'symmetry_operator_count':len(operators),
        'orbit_site_count':len(orbit),'graph_site_count':len(z),
        'upstream_prediction_parity_verified':False,'upstream_preprocessing_parity_verified':False,
        'pending_reference_differences':['Gemmi vs cctbx orbit representation/order/roundoff',
            'nearest-32 deterministic graph vs torch_cluster implementation-defined cap']}
    return HeavyPreparation(request,ids,tuple(a.label for a in parsed.atoms[count:]),parsed.sha256,provenance)


def infer_upstream_heavy(assigner, model_path: Path):
    from crystalx_core.app.reconstruction_runtime_inference import RuntimeAssignment
    assigner._verify_assets()
    prepared=prepare_upstream_heavy(model_path,source_root=assigner.source_root,
                                    max_image_candidates=assigner.max_image_candidates,max_sites=assigner.max_sites)
    request=prepared.request
    if max(request.atomic_numbers)>=min(assigner.backend.shape.num_classes,assigner.backend.shape.max_z):
        raise ReconstructionBlocked('upstream_heavy_descriptors_outside_checkpoint')
    result,choices=assigner.backend.predict_elements_with_coverage(request,device=assigner.device,allow_cpu_fallback=False)
    if (result.input_sha256!=digest(request) or result.model_identity.weights_sha256!=assigner.expected_sha256
            or result.device_requested!=assigner.device or result.device_used!=assigner.device or result.fell_back
            or result.candidate_atomic_numbers!=request.candidate_atomic_numbers
            or tuple(p.site_id for p in result.predictions)!=prepared.retained_site_ids
            or len(choices)!=len(prepared.retained_site_ids)
            or any(z not in request.candidate_atomic_numbers for z in choices)):
        raise ReconstructionBlocked('upstream_heavy_prediction_identity_mismatch')
    assigner._verify_assets()
    if sha256_file(model_path)!=prepared.source_sha256:raise ReconstructionBlocked('upstream_heavy_initial_model_changed')
    symbols={z:e for e,z in ATOMIC_NUMBERS.items()}
    return RuntimeAssignment(request,result,prepared.source_sha256,prepared.retained_site_ids,(),
        {**prepared.provenance,'protocol_element_choices':dict(zip(prepared.retained_site_ids,(symbols[z] for z in choices))),
         'weights_sha256':assigner.expected_sha256,'actual_pretrained_weights_executed':True,
         'no_cpu_fallback':True,'scientific_validation_performed':False})


def write_heavy_stage_model(source: Path, destination: Path, assignment) -> dict:
    """Relabel selected sites and reproduce upstream's SFAC/UNIT/ANIS template.

    The upstream constant H UNIT=37 is recorded as a template, never a hydrogen
    prediction or an experimentally established chemical composition.
    """
    parsed=read_res(source); ids=assignment.site_ids
    if parsed.sha256!=assignment.model_sha256:raise ReconstructionBlocked('upstream_heavy_initial_model_changed')
    choices=assignment.provenance['protocol_element_choices']
    if set(choices)!=set(ids):raise ReconstructionBlocked('upstream_heavy_assignment_sites_differ')
    if not set(choices.values())<=set(parsed.header.sfac):raise ReconstructionBlocked('upstream_heavy_choice_not_in_declared_sfac')
    counts=Counter(choices[key] for key in ids)
    sfac=[*counts,'H']
    atoms={a.label:a for a in parsed.atoms}; output=[]; added_anis=False
    element_counts={};label_map={}
    for line in logical_lines(source):
        if line.instruction=='END':break
        if line.tokens and line.tokens[0].upper().startswith('Q') and line.tokens[0][1:].isdigit():continue
        if line.instruction=='TITL':output.append('TITL CrystalX initial heavy stage (unvalidated)')
        elif line.instruction=='REM':
            output.extend('REM '+chunk for chunk in textwrap.wrap(line.text[3:].strip(),width=74,
                break_long_words=True,break_on_hyphens=False))
        elif line.instruction=='SFAC':output.append('SFAC '+' '.join(sfac))
        elif line.instruction=='UNIT':output.append('UNIT '+' '.join(str(n) for n in (*counts.values(),37)))
        elif line.tokens and line.tokens[0] in atoms:
            key=line.tokens[0]
            if key not in choices:continue
            if not added_anis:output.append('ANIS');added_anis=True
            # SHELXT typed peaks have an eighth peak-height token; SHELXL atom
            # input uses the first seven tokens, preserving coordinates/SOF/Uiso.
            tokens=list(line.tokens[:7]);tokens[1]=str(sfac.index(choices[key])+1)
            element=choices[key];element_counts[element]=element_counts.get(element,0)+1
            # SHELX identifiers are case-insensitive; use its canonical printed
            # spelling before the strict input/output identity check.
            tokens[0]=f'{element.upper()}{element_counts[element]}';label_map[key]=tokens[0]
            output.append(' '.join(tokens))
        elif line.instruction not in {'ANIS','L.S.','CGLS'}:output.extend(line.raw_lines)
        if line.instruction=='HKLF':break
    if not added_anis:raise ReconstructionBlocked('upstream_heavy_no_retained_rows_written')
    destination.write_text('\n'.join(output)+'\nEND\n')
    return {'retained_site_ids':ids,'site_label_map':label_map,'trimmed_site_ids':assignment.provenance['trimmed_site_ids'],
        'anisotropic_requested':True,'least_squares_cycles':16,
        'unit_template':'upstream predicted ASU counts plus constant H=37; not validated composition',
        'hydrogen_atoms_predicted':False,'hydrogen_unit_template':37,
        'label_spelling':'uppercase SHELXL canonical spelling',
        'metadata_formatting':'short TITL; REM lines wrapped below 80 columns',
        'source_sha256':parsed.sha256,'output_sha256':sha256_file(destination)}
