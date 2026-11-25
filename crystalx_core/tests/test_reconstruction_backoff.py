"""Distinct proposal backoff, PLAN and unsupported-anchor safeguards on generated inputs."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from test_reconstruction_backend import backend, run_backend, FixtureAssigner, FixtureToolchain
from test_adaptive_reconstruction import model
from crystalx_core.domain.inference import ElementPrediction, ClassProbability
from crystalx_core.domain.tools import ToolKind
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.refinement.adaptive import AdaptivePolicy, propose_pruning


class RepeatedFlipAssigner(FixtureAssigner):
    def assign(self, *args, **kwargs):
        value = super().assign(*args, **kwargs)
        if self.calls > 1:
            rows = tuple(ElementPrediction(p.site_id, (ClassProbability(6, .8), ClassProbability(7, .2)))
                         if p.site_id == 'C1' else p for p in value.result.predictions)
            value = replace(value, result=replace(value.result, predictions=rows))
        return value


def regressing_chain(*, reject_pure=False):
    chain = FixtureToolchain(peaks=True)
    execute = chain.executor.execute
    def refine(plan, env, **kwargs):
        raw = execute(plan, env, **kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            from crystalx_core.io.shelx_res import read_res
            parsed = read_res(plan.cwd / 'trial.res')
            c1 = next((a for a in parsed.atoms if a.label == 'C1'), None)
            count = sum(p.tool_kind is ToolKind.SHELXL for p in chain.executor.calls)
            bad = count > 1 and (reject_pure or c1 and parsed.element_of(c1) == 'C')
            if bad:
                for suffix in ('res','lst'):
                    path = plan.cwd / ('trial.' + suffix)
                    path.write_text(path.read_text().replace('0.300000', '0.500000'))
        return raw
    chain.executor.execute = refine
    return chain


def test_rejected_actions_are_skipped_and_reduce_to_pure_refinement(backend):
    result, _, chain, _ = run_backend(backend, chain=regressing_chain(), assigner=RepeatedFlipAssigner(),
                                      policy=CyclePolicy())
    assert result['stop_code'] == 'movable_sites_r1_and_growth_stable', result
    rejected = [r for r in result['history'] if r['decision'] == 'REJECT_R_REGRESSION']
    assert len(rejected) == 2
    assert len({r['proposal_sha256'] for r in rejected}) == 2
    assert rejected[0]['backoff_level'] == 0 and rejected[1]['backoff_level'] == 2
    assert rejected[0]['accepted_new_sites_if_retained'] > 0
    assert rejected[1]['accepted_new_sites_if_retained'] == 0
    pure = [r for r in result['history'] if r['backoff_level'] == 3]
    assert pure and all(not r['inference_executed'] and not r['action_sets']['element_and_growth_changes'] for r in pure)
    memory = json.loads((Path(result['run_root'])/'rejected-proposals.json').read_text())
    assert len(memory) == 2
    skipped = [json.loads(p.read_text())['skipped_duplicates'] for p in Path(result['run_root']).glob('cycles/round-*/proposal-identity.json')]
    assert any(rows for rows in skipped)
    assert sum(p.tool_kind is ToolKind.SHELXL for p in chain.executor.calls) == len(result['history'])


def test_no_acceptable_action_set_has_own_stop_and_retains_prior_model(backend):
    result, _, chain, _ = run_backend(backend, chain=regressing_chain(reject_pure=True),
        assigner=RepeatedFlipAssigner(), policy=CyclePolicy())
    assert result['stop_code'] == 'no_acceptable_step_in_current_action_set', result
    assert result['phase'] == 'blocked' and result['candidate_available']
    memory = json.loads((Path(result['run_root'])/'rejected-proposals.json').read_text())
    assert len(memory) == 3  # Growth+flip, flip alone, and pure refinement; no repeated native trial.
    assert len(result['history']) == 4
    assert result['history'][-1]['backoff_level'] == 3
    assert result['history'][-1]['inference_executed'] is False
    assert result['history'][-1]['decision'] == 'REJECT_R_REGRESSION'


@pytest.mark.parametrize('peaks', [0,20,127,200])
def test_explicit_plan_peak_budget_reaches_refinement_ins(backend, peaks):
    result, _, chain, _ = run_backend(backend, policy=CyclePolicy(iterations=1, plan_peaks=peaks))
    assert result['phase'] == 'completed', result
    invocation = next(p for p in chain.executor.calls if p.tool_kind is ToolKind.SHELXL)
    assert f'PLAN {peaks}' in (invocation.cwd/'trial.ins').read_text()
    assert result['history'][0]['plan_peaks'] == peaks


@pytest.mark.parametrize('bad', [-1,1000,True,20.5])
def test_invalid_plan_budgets_fail_before_work(bad):
    with pytest.raises(ValueError):CyclePolicy(plan_peaks=bad)


@pytest.mark.parametrize('u', [.31,2.0])
def test_unsupported_fixed_anchor_stops_after_first_refinement(backend, u):
    chain = FixtureToolchain()
    execute = chain.executor.execute
    def oversized(plan, env, **kwargs):
        raw = execute(plan, env, **kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            path = plan.cwd/'trial.res'
            rows = []
            for row in path.read_text().splitlines():
                fields = row.split()
                if fields and fields[0] == 'Zn1': fields[-1] = str(u); row = ' '.join(fields)
                rows.append(row)
            path.write_text('\n'.join(rows)+'\n')
        return raw
    chain.executor.execute = oversized
    result, _, _, _ = run_backend(backend, chain=chain, metal=True, policy=CyclePolicy(anchor_uiso_guard=True))
    assert result['stop_code'] == 'anchor_displacement_unsupported', result
    assert result['phase'] == 'blocked' and result['ai_solved_res'] is None
    assert len(result['history']) == 1
    assert sum(p.tool_kind is ToolKind.SHELXL for p in chain.executor.calls) == 1


def test_joint_u_and_isolation_rule_preserves_connected_bridge(backend, tmp_path):
    pytest.importorskip('gemmi'); pytest.importorskip('scipy.spatial')
    res = model(tmp_path, [('A','C',(.1,.1,.1),.05),('B','C',(.17,.1,.1),.8),
        ('C','C',(.24,.1,.1),.05),('D','C',(.7,.7,.7),.05),('E','C',(.7,.7,.4),.8)])
    policy = AdaptivePolicy(pruning_enabled=True, prune_close=False, prune_confirmations=1, max_prune_fraction=1.)
    any_rule = propose_pruning(res, policy=policy)
    joint = propose_pruning(res, policy=replace(policy, prune_evidence_mode='high_u_and_isolated'))
    assert set(any_rule.site_ids) == {'B','D','E'}
    assert joint.site_ids == ('E',)
