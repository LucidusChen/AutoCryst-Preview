"""Initial-only heavy preparation on synthetic cells, never private answers."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from crystalx_core.app.upstream_heavy import prepare_upstream_heavy, write_heavy_stage_model
from crystalx_core.app.reconstruction_backend import ReconstructionBackendService, ReconstructionRequest
from crystalx_core.app.reconstruction_runtime_inference import RuntimeAssignment
from crystalx_core.app.reconstruction_demo import SyntheticInference
from crystalx_core.app.models import RunMode
from crystalx_core.app.solve_routes import InitialRouteConfig, InitialStrategy
from crystalx_core.io.shelx_res import read_res
from crystalx_core.private_acceptance.sealing import verify_seal
from crystalx_core.workflow.reconstruction import ReconstructionBlocked
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from test_reconstruction_backend import backend, FixtureToolchain

SOURCE=Path(__file__).resolve().parents[2]/'crystalx_engine/inference_repo'


@pytest.fixture
def source():
    pytest.importorskip('gemmi');pytest.importorskip('scipy');pytest.importorskip('numpy')
    if not (SOURCE/'crystalx_infer/common/heavy.py').is_file():pytest.skip('optional upstream heavy helper absent')
    return SOURCE


def model(tmp_path,*,latt=-1,cell='10 10 10 90 90 90',unit='1 1',rows=None):
    p=tmp_path/'initial.res'
    rows=rows or ['O1 2 .01 .02 .03 11 .03','C1 1 .21 .22 .23 11 .04']
    p.write_text(f'TITL synthetic heavy recipe\nCELL .71073 {cell}\nLATT {latt}\nSFAC C O\nUNIT {unit}\nL.S. 4\nPLAN 20\n'+
                 '\n'.join(rows)+'\nHKLF 4\nEND\nQ1 1 .4 .5 .6 11 .05 20\n')
    return p


def test_file_order_ratio_and_no_q_peaks(tmp_path,source):
    p=model(tmp_path)
    prepared=prepare_upstream_heavy(p,source_root=source)
    request=prepared.request
    assert request.site_ids[:2]==('O1','C1')
    assert request.atomic_numbers[:2]==(8,6)
    assert request.main_site_count==2 and 'Q1' not in request.site_ids
    assert prepared.provenance['forced_coverage'] and not prepared.provenance['upstream_preprocessing_parity_verified']


def test_trim_uses_divisibility_and_records_original_tail(tmp_path,source):
    p=model(tmp_path,unit='3 3',rows=[f'C{i+1} 1 .{i+1} .2 .3 11 .03' for i in range(5)])
    prepared=prepare_upstream_heavy(p,source_root=source)
    assert prepared.retained_site_ids==('C1','C2','C3')
    assert prepared.trimmed_site_ids==('C4','C5')
    assert len(read_res(p).atoms)==5  # original untouched


@pytest.mark.parametrize('latt,multiplicity',[(-1,1),(-2,2),(-4,4),(1,2)])
def test_centring_and_special_position_orbit(tmp_path,source,latt,multiplicity):
    p=model(tmp_path,latt=latt,rows=['O1 2 0 0 0 11 .03','C1 1 .1 .2 .3 11 .03'])
    prepared=prepare_upstream_heavy(p,source_root=source)
    expected=3 if latt==1 else 2*multiplicity
    assert prepared.provenance['orbit_site_count']==expected


def test_context_sphere_and_original_label_descriptors(tmp_path,source):
    p=model(tmp_path,cell='3 8 9 90 90 90',rows=['C1 1 .05 .2 .3 11 .03','O1 2 .15 .2 .3 11 .03'])
    prepared=prepare_upstream_heavy(p,source_root=source);request=prepared.request
    assert request.atomic_numbers[:2]==(8,6)  # ratio allocation sorted, not source labels
    assert len(request.site_ids)>2
    import math
    for xyz in request.positions_angstrom[2:]:
        d=min(math.dist(xyz,main) for main in request.positions_angstrom[:2])
        assert .2<d<3.2
    assert set(request.atomic_numbers[2:])=={6,8}


def test_duplicate_main_positions_block_instead_of_shifting_mask(tmp_path,source):
    p=model(tmp_path,rows=['C1 1 .1 .2 .3 11 .03','O1 2 .1 .2 .3 11 .03'])
    with pytest.raises(ReconstructionBlocked,match='mask_ambiguous'):
        prepare_upstream_heavy(p,source_root=source)


def test_stage_writer_strips_peak_height_records_upstream_unit_template_and_requests_anis(tmp_path,source):
    p=model(tmp_path,rows=['O1 2 .01 .02 .03 11 .03 12','C1 1 .21 .22 .23 11 .04 10'])
    prepared=prepare_upstream_heavy(p,source_root=source)
    assignment=SimpleNamespace(site_ids=prepared.retained_site_ids,model_sha256=prepared.source_sha256,
        provenance={'protocol_element_choices':{'O1':'C','C1':'O'},'trimmed_site_ids':[]})
    target=tmp_path/'assigned.res';record=write_heavy_stage_model(p,target,assignment)
    text=target.read_text()
    assert 'ANIS' in text and 'Q1' not in text and 'UNIT 1 1 37' in text
    assert 'UNIT 1 1\n' in p.read_text() and record['hydrogen_unit_template']==37
    assert [a.u_values for a in read_res(target).atoms]==[(.03,),(.04,)]
    assert [a.label for a in read_res(target).atoms]==['C1','O1']
    assert record['site_label_map']=={'O1':'C1','C1':'O1'}
    assert record['least_squares_cycles']==16


def test_recipe_is_opt_in_and_rejects_non_shelxt_or_pruned_routes(backend):
    _,ins,hkl=backend
    assert ReconstructionRequest(ins,hkl).heavy_preparation=='current_loop'
    with pytest.raises(ValueError,match='complete initial SHELXT'):
        ReconstructionRequest(ins,hkl,heavy_preparation='upstream_heavy_v1',metal_only=True)
    with pytest.raises(ValueError,match='complete initial SHELXT'):
        ReconstructionRequest(ins,hkl,heavy_preparation='upstream_heavy_v1',
            route=InitialRouteConfig(InitialStrategy.SMALL_MOLECULE))


def test_native_text_boundaries_use_canonical_labels_and_short_comments(tmp_path):
    p=model(tmp_path)
    p.write_text(p.read_text().replace('SFAC C O','REM '+ 'long synthetic title '*12+'\nSFAC C O Zn'))
    parsed=read_res(p)
    assignment=SimpleNamespace(site_ids=('O1','C1'),model_sha256=parsed.sha256,
        provenance={'protocol_element_choices':{'O1':'Zn','C1':'C'},'trimmed_site_ids':[]})
    target=tmp_path/'canonical.res';record=write_heavy_stage_model(p,target,assignment)
    assert [a.label for a in read_res(target).atoms]==['ZN1','C1']
    assert record['site_label_map']['O1']=='ZN1'
    assert all(len(line)<80 for line in target.read_text().splitlines())


def test_backend_records_one_stage_and_seals_final_phase(backend,source):
    repo,ins,hkl=backend;chain=FixtureToolchain();calls=[]
    class Assigner:
        def assign_upstream_heavy(self,path):
            calls.append(path)
            prepared=prepare_upstream_heavy(path,source_root=source)
            result=SyntheticInference().predict_elements(prepared.request)
            symbols={v:k for k,v in ATOMIC_NUMBERS.items()}
            choices={p.site_id:symbols[p.topk[0].atomic_number] for p in result.predictions[:len(prepared.retained_site_ids)]}
            return RuntimeAssignment(prepared.request,replace(result,predictions=result.predictions[:len(choices)]),
                prepared.source_sha256,prepared.retained_site_ids,(),
                {**prepared.provenance,'protocol_element_choices':choices,'synthetic':True})
    service=ReconstructionBackendService(repo,repo/'runs/heavy',toolchain_factory=lambda _:chain,
        assigner_factory=lambda _:Assigner())
    result=service.run(ReconstructionRequest(ins,hkl,heavy_preparation='upstream_heavy_v1',mode=RunMode.MOCK_DEMO))
    assert result['phase']=='completed',result
    assert len(calls)==1 and result['inference_calls']==1 and result['tool_calls']==2
    assert result['history']==[] and result['stages'][0]['shelxl_cycles']==16
    root=service.run_root(result['run_id'])
    assert 'ANIS' in (root/'heavy-stage/anisotropic-ls16/trial.ins').read_text()
    assert 'L.S. 16' in (root/'heavy-stage/anisotropic-ls16/trial.ins').read_text()
    seal=verify_seal(root,private_root=service.runs_root,expected_seal_sha256=result['seal_sha256'])
    assert seal.terminal_state==result['phase']==service.status(result['run_id'])['phase']


def test_anisotropic_output_is_supported_only_by_opt_in_refinement(tmp_path):
    from crystalx_core.refinement.cycle import checked_res, refine_file
    from crystalx_core.tools.runner import CancelToken
    from crystalx_core.domain.tools import ToolKind
    p=model(tmp_path);hkl=tmp_path/'input.hkl';hkl.write_text('generated fixture reflections\n')
    chain=FixtureToolchain();execute=chain.executor.execute
    def anisotropic(plan,env,**kwargs):
        raw=execute(plan,env,**kwargs)
        result=plan.cwd/'trial.res'
        result.write_text('\n'.join(' '.join(line.split()[:6])+ ' .03 .04 .05 .001 .002 .003'
            if line.split() and line.split()[0] in {'O1','C1'} else line
            for line in result.read_text().splitlines())+'\n')
        return raw
    chain.executor.execute=anisotropic
    r=refine_file(model_path=p,hkl_path=hkl,directory=tmp_path/'anisotropic',
        adapter=chain.adapter(ToolKind.SHELXL),cycles=16,timeout_s=10,cancel=CancelToken(),isotropic=False)
    assert all(a.is_anisotropic for a in checked_res(r.model_path,allow_anisotropic=True).atoms)
    with pytest.raises(ReconstructionBlocked,match='anisotropic'):
        checked_res(r.model_path)
