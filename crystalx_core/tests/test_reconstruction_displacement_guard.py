"""Synthetic whole-trial displacement guard and rollback evidence."""
from pathlib import Path
import json
import pytest
from test_reconstruction_backend import backend, run_backend, FixtureToolchain
from crystalx_core.domain.tools import ToolKind
from crystalx_core.workflow.refinement_cycle import CyclePolicy


def shifted_chain(only_calls=None):
    chain=FixtureToolchain();execute=chain.executor.execute;count=0
    def run(plan,env,**kwargs):
        nonlocal count
        result=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            count+=1
            if only_calls is None or count in only_calls:
                path=plan.cwd/'trial.res';rows=[]
                for line in path.read_text().splitlines():
                    fields=line.split()
                    if fields and fields[0]=='C1':
                        fields[2]=str(float(fields[2])+.1);line=' '.join(fields)
                    rows.append(line)
                path.write_text('\n'.join(rows)+'\n')
        return result
    chain.executor.execute=run
    return chain


@pytest.mark.parametrize('limit',[-1,0,float('nan'),float('inf'),True,'1'])
def test_shift_guard_rejects_invalid_policy(limit):
    with pytest.raises(ValueError,match='shift limit'):
        CyclePolicy(max_refinement_net_shift_angstrom=limit)


def test_default_keeps_motion_diagnostic_without_rejection(backend):
    policy=CyclePolicy(iterations=1)
    assert policy.max_refinement_net_shift_angstrom is None
    assert CyclePolicy.from_dict(policy.to_dict())==policy
    result,_,_,_=run_backend(backend,chain=shifted_chain(),policy=policy)
    assert result['phase']=='completed',result
    h=result['history'][0]
    assert h['refinement_max_displacement_angstrom']>.5
    assert h['max_refinement_net_shift_angstrom'] is None
    assert not h['refinement_displacement_limit_exceeded']


def test_shift_rejection_rolls_back_prior_model_and_charges_once(backend):
    result,_,_,_=run_backend(backend,chain=shifted_chain({2}),
        policy=CyclePolicy(iterations=3,max_refinement_net_shift_angstrom=.5))
    # The remaining action levels produce the identical refused proposal, so
    # proposal memory must block them rather than retrying a deterministic step.
    assert result['stop_code']=='no_acceptable_step_in_current_action_set',result
    assert result['history'][1]['decision']=='REJECT_REFINEMENT_DISPLACEMENT'
    assert result['history'][1]['refinement_displacement_limit_exceeded']
    assert result['distinct_rejections_total']==1
    root=Path(result['run_root'])
    abort=json.loads((root/'cycles/round-002/abort.json').read_text())
    assert abort['restored_model_sha256']==result['history'][0]['model_sha256']
    rejected=json.loads((root/'rejected-proposals.json').read_text())
    assert {v['reason'] for v in rejected.values()}=={'net_refinement_displacement'}
    assert len(result['history'])==2
    assert result['candidate_sha256']==result['history'][0]['model_sha256']


def test_pure_refinement_still_jumps_so_ladder_blocks_without_candidate(backend):
    result,_,_,_=run_backend(backend,chain=shifted_chain(),
        policy=CyclePolicy(max_refinement_net_shift_angstrom=.5))
    assert result['stop_code']=='no_acceptable_step_in_current_action_set',result
    assert result['ai_solved_res'] is None
    assert all(h['decision']=='REJECT_REFINEMENT_DISPLACEMENT' for h in result['history'])
    assert 1<=result['distinct_rejections_total']<=4
