"""Review-driven budget, action memory, anchor policy and stability tests (generated inputs)."""
from pathlib import Path
import json
from dataclasses import replace
import pytest
from test_reconstruction_backend import backend, run_backend, FixtureToolchain
from test_reconstruction_backoff import RepeatedFlipAssigner, regressing_chain
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.domain.tools import ToolKind
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.app.reconstruction_demo import synthetic_seed_text


def shifted_pure_chain():
    chain = regressing_chain()
    execute = chain.executor.execute
    def shifted(plan, env, **kwargs):
        raw = execute(plan, env, **kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            path = plan.cwd/'trial.res'; rows=[]
            for row in path.read_text().splitlines():
                fields=row.split()
                if fields and fields[0]=='C1' and fields[1]=='3':
                    fields[2]=str(float(fields[2])+.001); row=' '.join(fields)
                rows.append(row)
            path.write_text('\n'.join(rows)+'\n')
        return raw
    chain.executor.execute=shifted
    return chain


def test_default_failure_budget_allows_four_level_ladder(backend):
    policy=CyclePolicy()
    assert policy.max_failures==2 and policy.backoff_failure_budget==4
    result,_,_,_=run_backend(backend,chain=regressing_chain(reject_pure=True),
                              assigner=RepeatedFlipAssigner(),policy=policy)
    assert result['stop_code']=='no_acceptable_step_in_current_action_set',result
    assert result['history'][-1]['backoff_level']==3
    assert result['history'][-1]['effective_failure_budget']==4


def test_acceptance_resets_base_budget_and_action_memory_survives_small_coordinate_changes(backend):
    result,_,_,_=run_backend(backend,chain=shifted_pure_chain(),assigner=RepeatedFlipAssigner(),
        policy=CyclePolicy(iterations=12,rejected_action_ttl_rounds=3))
    assert result['stop_code']=='fixed_rounds_completed',result
    assert result['distinct_rejections_total']>2  # Whole-run accumulation must not exhaust max_failures=2.
    assert result['base_rejection_count']==0
    assert result['skipped_rejected_actions']>0
    previous={}
    for row in result['history']:
        if row['decision']!='REJECT_R_REGRESSION':continue
        key=row['action_set_sha256']
        assert row['base_rejection_count']<=4
        if key in previous:assert row['round']-previous[key]>3
        previous[key]=row['round']
    assert result['repeated_rejected_actions']>0  # Expired actions may be retried, and are counted honestly.


@pytest.mark.parametrize('bad', [0,1,2,3,True])
def test_backoff_budget_cannot_be_shorter_than_ladder(bad):
    with pytest.raises(ValueError):CyclePolicy(backoff_failure_budget=bad)


def anchor_chain():
    chain=FixtureToolchain();execute=chain.executor.execute
    def generate(plan,env,**kwargs):
        raw=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXT:
            p=plan.cwd/(plan.argv[1]+'_a.res')
            p.write_text(p.read_text().replace('HKLF 4','Zn2 5 .45 .3 .2 11 .03\nZn3 5 .65 .3 .2 11 .03\nHKLF 4'))
        elif plan.tool_kind is ToolKind.SHELXL:
            p=plan.cwd/'trial.res';rows=[]
            for row in p.read_text().splitlines():
                fields=row.split()
                if fields and fields[0]=='Zn1':fields[-1]='.5';row=' '.join(fields)
                rows.append(row)
            p.write_text('\n'.join(rows)+'\n')
        return raw
    chain.executor.execute=generate
    return chain


def test_anchor_guard_is_opt_in_and_partial_rejection_unlocks_without_deleting(backend):
    assert CyclePolicy().anchor_uiso_guard is False
    chain=anchor_chain()
    result,_,_,_=run_backend(backend,chain=chain,metal=True,
        policy=CyclePolicy(iterations=2,anchor_uiso_guard=True,anchor_min_supported=2))
    assert result['phase']=='completed',result
    assert result['history'][0]['anchor_sites_to_unlock_if_retained']==['Zn1']
    trial=[p for p in chain.executor.calls if p.tool_kind is ToolKind.SHELXL][1]
    fields={row.split()[0]:row.split() for row in (trial.cwd/'trial.ins').read_text().splitlines() if row.startswith(('Zn1 ','Zn2 ','Zn3 '))}
    assert float(fields['Zn1'][2])<10 and float(fields['Zn2'][2])>10
    assert set(fields)=={'Zn1','Zn2','Zn3'}


def test_anchor_minimum_and_report_only_are_explicit(backend):
    result,_,_,_=run_backend(backend,chain=anchor_chain(),metal=True,
        policy=CyclePolicy(anchor_uiso_guard=True,anchor_min_supported=3))
    assert result['stop_code']=='anchor_displacement_unsupported',result
    reported,_,_,_=run_backend(backend,chain=anchor_chain(),metal=True,
        policy=CyclePolicy(iterations=1,anchor_uiso_guard=True,anchor_min_supported=3,anchor_uiso_policy='report_only'))
    assert reported['phase']=='completed'
    assert reported['history'][0]['anchor_sites_to_unlock_if_retained']==[]


def test_cgls_damp_and_default_preservation():
    source=synthetic_seed_text().replace('HKLF 4','DAMP 1 2\nCGLS 20\nHKLF 4')
    text,_,_,_=prepare_refinement_ins(source,RefinementSettings(method='cgls',cycles=3,damp_factor=.4,damp_shift_esd_limit=5))
    assert 'CGLS 3' in text and 'DAMP 0.4 5' in text
    assert 'DAMP 1 2' not in text and 'L.S.' not in text
    normal,_,_,_=prepare_refinement_ins(source,RefinementSettings())
    assert 'L.S. 4' in normal and 'DAMP 1 2' in normal  # None does not silently override supplied DAMP.


def test_cgls_early_schedule_and_net_displacement_diagnostics(backend):
    chain=FixtureToolchain();execute=chain.executor.execute
    def shift(plan,env,**kwargs):
        raw=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            p=plan.cwd/'trial.res';rows=[]
            for row in p.read_text().splitlines():
                fields=row.split()
                if fields and fields[0]=='C1':fields[2]=str(float(fields[2])+.1);row=' '.join(fields)
                rows.append(row)
            p.write_text('\n'.join(rows)+'\n')
        return raw
    chain.executor.execute=shift
    result,_,_,_=run_backend(backend,chain=chain,policy=CyclePolicy(iterations=2,early_cgls_rounds=1,damp_factor=.4))
    assert result['phase']=='completed',result
    trials=[p for p in chain.executor.calls if p.tool_kind is ToolKind.SHELXL]
    assert 'CGLS 4' in (trials[0].cwd/'trial.ins').read_text()
    assert 'L.S. 4' in (trials[1].cwd/'trial.ins').read_text()
    assert all(row['refinement_displacement_above_threshold_count']==1 for row in result['history'])
    assert result['history'][0]['refinement_max_displacement_angstrom']==pytest.approx(.75)


def test_failed_anchor_release_commit_restores_all_original_anchor_constraints(backend,monkeypatch):
    from crystalx_core.workspace.store import RunWorkspace
    write=RunWorkspace.write_json
    def fault(workspace,relative,value,**kwargs):
        if str(relative)=='cycles/round-001/commit.json':raise OSError('synthetic anchor-release persistence fault')
        return write(workspace,relative,value,**kwargs)
    monkeypatch.setattr(RunWorkspace,'write_json',fault)
    result,_,_,_=run_backend(backend,chain=anchor_chain(),metal=True,
        policy=CyclePolicy(iterations=2,anchor_uiso_guard=True,anchor_min_supported=2))
    assert result['phase']=='failed',result
    abort=json.loads((Path(result['run_root'])/'cycles/round-001/abort.json').read_text())
    assert set(abort['restored_fixed_site_ids'])=={'Zn1','Zn2','Zn3'}
    assert result['ai_solved_res'] is None


def test_exact_proposal_identity_includes_scale_weights_and_ignores_comments(tmp_path):
    from crystalx_core.workflow.refinement_cycle import RefinementCycleExecutor
    from crystalx_core.refinement.cycle import checked_res
    executor=object.__new__(RefinementCycleExecutor)
    executor.policy=CyclePolicy();executor.hkl_sha='a'*64
    path=tmp_path/'model.res';base=synthetic_seed_text()
    path.write_text(base);first=executor._model_key(checked_res(path))
    path.write_text(base.replace('FVAR 1','FVAR 1.1'))
    assert executor._model_key(checked_res(path))!=first
    path.write_text(base.replace('HKLF 4','WGHT .1 .2\nHKLF 4'))
    assert executor._model_key(checked_res(path))!=first
    path.write_text(base.replace('HKLF 4','REM a different display comment\nHKLF 4'))
    assert executor._model_key(checked_res(path))==first
