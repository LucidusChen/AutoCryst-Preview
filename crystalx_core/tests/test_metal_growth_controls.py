"""Experimental metal-growth controls, opt-in ADPs and complete peak accounting."""
from pathlib import Path
import json
import pytest

from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.refinement.shelxl import RefinementSettings,prepare_refinement_ins,_logical_from_text
from crystalx_core.refinement.cycle import refine_file,checked_res,displacement_value,shelx_continuation_rows,inference_input
from crystalx_core.refinement.adaptive import AdaptivePolicy,select_growth_peaks,metal_ripple_rejections,uiso_consistency
from crystalx_core.workflow.reconstruction import read_snapshot,ReconstructionBlocked
from crystalx_core.io.shelx_res import read_res
from crystalx_core.tools.runner import CancelToken
from crystalx_core.domain.tools import ToolKind
from test_reconstruction_backend import backend,FixtureToolchain,run_backend


@pytest.fixture
def geometry():
    pytest.importorskip('gemmi');pytest.importorskip('scipy')


def model(tmp_path,*,anis=False,latt=-1):
    p=tmp_path/'metal.res'
    u='.001 .09 .09 0 0 0' if anis else '.03'
    p.write_text(f'TITL synthetic metal growth\nCELL .71073 10 11 12 90 103 90\nLATT {latt}\nSFAC C Zn\nUNIT 1 1\nFVAR 1\nZN1 2 .25 .3 .2 11 {u}\nC1 1 .45 .3 .2 11 .04\nHKLF 4\nEND\n')
    return p


def test_default_controls_and_expanded_plan_bound_are_explicit():
    p=CyclePolicy();assert (p.fourier_map,p.plan_peaks,p.anisotropic_metals,p.metal_ripple_exclusion_angstrom)==(2,20,False,0)
    for n in (0,200,400,999):
        assert CyclePolicy(plan_peaks=n).to_dict()['plan_peaks']==n
        assert RefinementSettings(plan_peaks=n).to_record()['plan_peaks']==n
    for n in (1000,-1,True):
        with pytest.raises(ValueError):CyclePolicy(plan_peaks=n)
        with pytest.raises(ValueError):RefinementSettings(plan_peaks=n)


@pytest.mark.parametrize('change',[{'fourier_map':1},{'fourier_map':True},{'fourier_map':5},
    {'anisotropic_metals':1},{'metal_ripple_exclusion_angstrom':-1},
    {'metal_ripple_exclusion_angstrom':float('nan')},{'metal_ripple_exclusion_angstrom':True}])
def test_invalid_controls_block_before_work(change):
    with pytest.raises(ValueError):CyclePolicy(**change)


@pytest.mark.parametrize('code',[2,3,4])
def test_map_choice_and_large_peak_list_are_written(tmp_path,code):
    text=model(tmp_path).read_text().replace('FVAR','FMAP 2\nPLAN 20\nFVAR')
    prepared,*_=prepare_refinement_ins(text,RefinementSettings(plan_peaks=999,fourier_map=code))
    assert prepared.count('FMAP ')==1 and f'FMAP {code}\n' in prepared
    assert 'PLAN 999\n' in prepared
    assert ('\nACTA\n' in prepared)==(code==2)


def anisotropic_chain(*,peaks=False):
    chain=FixtureToolchain(peaks=peaks);execute=chain.executor.execute
    def anisotropic(plan,env,**kwargs):
        raw=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            p=plan.cwd/'trial.res';rows=[]
            for line in _logical_from_text(p.read_text().splitlines()):
                tokens=line.text.split()
                if tokens and tokens[0].upper().startswith('ZN'):
                    rows.extend(shelx_continuation_rows(tokens[:6]+['.001','.09','.09','0','0','0']))
                else:rows.extend(line.raw_lines)
            p.write_text('\n'.join(rows)+'\n')
        return raw
    chain.executor.execute=anisotropic
    return chain


def test_selected_anis_and_fixed_coordinates_survive_multiline_native_output(tmp_path):
    p=model(tmp_path);hkl=tmp_path/'input.hkl';hkl.write_text('fixture\n');chain=anisotropic_chain()
    r=refine_file(model_path=p,hkl_path=hkl,directory=tmp_path/'trial',adapter=chain.adapter(ToolKind.SHELXL),
        cycles=2,timeout_s=10,cancel=CancelToken(),locked_sites=('ZN1',),anisotropic_metals=True,fourier_map=4,plan_peaks=400)
    parsed=checked_res(r.model_path,allow_anisotropic=True);zn,carbon=parsed.atoms
    assert zn.is_anisotropic and not carbon.is_anisotropic
    assert (zn.x,zn.y,zn.z)==(.25,.3,.2)
    text=(tmp_path/'trial/trial.ins').read_text()
    assert 'ANIS $ZN' in text and '\nANIS\n' not in text
    assert text.index('ANIS $ZN')<text.index('ZN1 2')
    assert all(len(line)<80 for line in text.splitlines())
    controls=json.loads((tmp_path/'trial/instruction-changes.json').read_text())
    assert controls['fourier_map']==4 and controls['plan_peaks']==400
    assert controls['anisotropic_metal_site_ids']==['ZN1']
    with pytest.raises(ReconstructionBlocked):checked_res(r.model_path)
    with pytest.raises(ReconstructionBlocked):read_snapshot(r.model_path,reflection_sha256='fixture')
    snapshot=read_snapshot(r.model_path,reflection_sha256='fixture',allow_anisotropic_metals=True)
    assert snapshot.sites[0].u_iso==pytest.approx(displacement_value(parsed,zn))
    assert snapshot.sites[0].u_iso!=zn.u_values[0]
    evidence=uiso_consistency(parsed,policy=AdaptivePolicy())['sites'][0]
    assert evidence['u_iso']==snapshot.sites[0].u_iso and evidence['classification']=='ordinary'


def peak_model(tmp_path,*,latt=-1):
    p=tmp_path/'peaks.res'
    p.write_text(f'TITL synthetic ripples\nCELL .71073 10 10 10 90 90 90\nLATT {latt}\nSFAC C Zn\nUNIT 1 1\nFVAR 1\nZN1 2 .05 .5 .5 11 .03\nC1 1 .39 .5 .5 11 .04\nHKLF 4\nEND\nQ1 1 .99 .5 .5 11 .05 100\nQ2 1 .25 .5 .5 11 .05 5\n')
    return read_res(p)


def test_periodic_metal_ripples_do_not_set_relative_ligand_threshold(tmp_path,geometry):
    m=peak_model(tmp_path);policy=AdaptivePolicy(growth_filter_enabled=True)
    baseline,_=select_growth_peaks(m,policy=policy,round_number=1,limit=12,minimum_height=.5,minimum_distance=.9)
    assert not baseline
    selected,evidence=select_growth_peaks(m,policy=policy,round_number=1,limit=12,minimum_height=.5,minimum_distance=.9,metal_ripple_exclusion_angstrom=1.2)
    assert selected==('Q2',)
    rows={r['peak_id']:r for r in evidence['decisions']}
    assert rows['Q1']['reason']=='metal_ripple_exclusion'
    assert rows['Q1']['metal_distance_angstrom']==pytest.approx(.6)
    assert rows['Q2']['strongest_residual_peak_height']==5
    assert evidence['relative_height_reference_scope']=='after_metal_ripple_exclusion'


def test_ripple_exclusion_uses_declared_symmetry_and_default_zero_is_off(tmp_path,geometry):
    p=model(tmp_path,latt=1);p.write_text(p.read_text()+'Q1 1 .76 .7 .8 11 .05 20\n')
    m=read_res(p)
    assert metal_ripple_rejections(m,policy=AdaptivePolicy(),radius=0)=={}
    assert 'Q1' in metal_ripple_rejections(m,policy=AdaptivePolicy(),radius=1.2)


def test_legacy_peak_preparation_records_exclusion_and_all_budget_rejections(tmp_path):
    m=peak_model(tmp_path);rows=[]
    selected=inference_input(m,tmp_path/'input.res',max_new_peaks=1,min_peak_height=.5,min_distance=.8,
        excluded_peak_ids=('Q1',),decision_records=rows)
    assert selected==('Q2',) and len(rows)==2
    assert rows[0]['reason']=='metal_ripple_exclusion' and rows[1]['decision']=='select'


def test_two_rounds_accept_anisotropic_metal_input_and_report_actual_growth(backend,monkeypatch,geometry):
    import test_reconstruction_backend as fixtures
    original=fixtures.read_snapshot
    monkeypatch.setattr(fixtures,'read_snapshot',lambda *a,**k:original(*a,**dict(k,allow_anisotropic_metals=True)))
    policy=CyclePolicy(iterations=2,plan_peaks=400,fourier_map=4,anisotropic_metals=True,metal_ripple_exclusion_angstrom=1.2)
    result,_,chain,_=run_backend(backend,chain=anisotropic_chain(peaks=True),policy=policy,metal=True)
    assert result['phase']=='blocked',result
    assert 'non_difference_map_native_cif_requires_recorded_fmap2_refinement' in result['stop_detail']
    assert result['candidate_available'] and not (Path(result['run_root'])/'outputs/ai_solved.cif').exists()
    assert len(result['history'])==2
    for row in result['history']:
        assert row['fourier_map']==4 and row['plan_peaks']==400 and row['anisotropic_metals']
        assert row['accepted_new_site_count']==row['accepted_new_sites_if_retained']
        assert row['listed_output_peak_count']>=0 and isinstance(row['peak_rejection_reason_counts'],dict)
    model_path=Path(result['ai_solved_res']);assert any(a.is_anisotropic for a in read_res(model_path).atoms)


def test_actual_pretrained_graph_can_read_opt_in_anisotropic_metal(tmp_path,geometry):
    pytest.importorskip('torch')
    from crystalx_core.app.reconstruction_runtime_inference import RuntimeElementAssigner
    from crystalx_core.domain.inference import InferenceDevice
    root=Path(__file__).resolve().parents[2];source=root/'crystalx_engine/inference_repo';weights=source/'weights/crystalx-heavy.pth'
    if not weights.is_file() or not (source/'crystalx_infer/models/torchmd_et.py').is_file():pytest.skip('optional official local source/weights absent')
    p=model(tmp_path,anis=True)
    assigner=RuntimeElementAssigner(source,weights,device=InferenceDevice.CPU,allow_anisotropic_metals=True)
    result=assigner.assign(p,candidate_elements=('C','Zn'),include_q_peaks=False)
    assert len(result.result.predictions)==2 and result.provenance['anisotropic_metals_allowed']
    assert result.result.fell_back is False and read_res(p).atoms[0].is_anisotropic
