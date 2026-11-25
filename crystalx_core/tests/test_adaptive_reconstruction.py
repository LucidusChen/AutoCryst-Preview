"""Generated geometry only: pure proposal, audit and transaction-state behavior."""

from dataclasses import replace
from copy import deepcopy

import pytest

from crystalx_core.domain.inference import ElementPrediction, ClassProbability
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.io.shelx_res import read_res
from crystalx_core.refinement.adaptive import (
    AdaptivePolicy, geometry_summary, propose_assignments, propose_pruning,
    select_growth_peaks, uiso_consistency,
)
from crystalx_core.workflow.reconstruction import ReconstructionBlocked


def model(tmp_path, atoms, *, cell=(20,20,20,90,90,90), peaks=(), symm=()):
    path=tmp_path/'generated.res'
    elements=list(dict.fromkeys(row[1] for row in atoms))
    lines=['TITL generated adaptive fixture','CELL 0.7 '+' '.join(map(str,cell)),'LATT -1']
    lines.extend('SYMM '+text for text in symm)
    lines+=['SFAC '+' '.join(elements),'UNIT '+' '.join('1' for _ in elements),'FVAR 0.5']
    for label,element,xyz,u in atoms:
        lines.append(f'{label} {elements.index(element)+1} {xyz[0]} {xyz[1]} {xyz[2]} 11 {u}')
    lines+=['HKLF 4','END']
    lines.extend(f'{label} 1 {xyz[0]} {xyz[1]} {xyz[2]} 11 .05 {height}' for label,xyz,height in peaks)
    path.write_text('\n'.join(lines)+'\n')
    return read_res(path)


def carbon_twenty(tmp_path, *, u1=.001,u2=.001):
    return model(tmp_path,[(f'C{i}', 'C', (.02+i*.03,.1,.1), u1 if i==1 else u2 if i==2 else .05) for i in range(1,21)])


def predictions(res, choices=None):
    choices=choices or {}
    result=[]
    for atom in res.atoms:
        chosen=choices.get(atom.label,res.element_of(atom));alternative='N' if chosen!='N' else 'C'
        result.append(ElementPrediction(atom.label,(ClassProbability(ATOMIC_NUMBERS[chosen],.999999),ClassProbability(ATOMIC_NUMBERS[alternative],.000001))))
    return tuple(result)


def test_defaults_are_opt_out_and_policy_roundtrip_is_strict():
    policy=AdaptivePolicy()
    assert not any((policy.hysteresis_enabled,policy.uiso_checks_enabled,policy.pruning_enabled,policy.growth_filter_enabled))
    assert AdaptivePolicy.from_dict(policy.to_dict())==policy
    with pytest.raises(ValueError):AdaptivePolicy.from_dict({'unknown_threshold':.4})
    with pytest.raises(ValueError):AdaptivePolicy(max_change_fraction=0)


def test_repeated_proposal_and_u_direction_are_both_required_with_literal_five_percent_cap(tmp_path):
    res=carbon_twenty(tmp_path);raw=predictions(res,{'C1':'O','C2':'N'})
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=True)
    first=propose_assignments(res,raw,policy=policy)
    assert first.overrides['C1']=='C' and first.overrides['C2']=='C'
    second=propose_assignments(res,raw,policy=policy,state=first.state)
    assert second.overrides['C1']=='O' and second.overrides['C2']=='C'
    assert second.evidence['changed_site_count']==second.evidence['change_limit']==1
    assert any(row['reason']=='per_round_change_cap' for row in second.evidence['decisions'])
    assert raw[0].topk[0].probability==.999999


def test_alternating_protocol_proposals_never_accumulate_confirmation(tmp_path):
    res=carbon_twenty(tmp_path)
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=True)
    state=None
    for element in ('O','N','O','N','O'):
        proposal=propose_assignments(res,predictions(res,{'C1':element}),policy=policy,state=state)
        assert proposal.overrides['C1']=='C'
        state=proposal.state


def test_uiso_wrong_direction_and_missing_references_abstain(tmp_path):
    res=carbon_twenty(tmp_path,u1=.5,u2=.05)
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=True)
    first=propose_assignments(res,predictions(res,{'C1':'I'}),policy=policy)
    second=propose_assignments(res,predictions(res,{'C1':'I'}),policy=policy,state=first.state)
    assert second.overrides['C1']=='C'
    row=next(row for row in second.evidence['decisions'] if row['site_id']=='C1')
    assert row['reason']=='uiso_does_not_support_proposed_direction'
    assert row['u_evidence']['classification']=='high'
    assert row['u_evidence']['reference_median'] is None


def test_repeated_confirmed_sites_freeze_and_u_anomaly_can_explicitly_thaw(tmp_path):
    res=carbon_twenty(tmp_path,u1=.05,u2=.05)
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=True)
    state=None
    for _ in range(3):state=propose_assignments(res,predictions(res),policy=policy,state=state).state
    assert state['assignment']['C1']['frozen']
    held=propose_assignments(res,predictions(res,{'C1':'O'}),policy=policy,state=state)
    assert held.overrides['C1']=='C'
    assert held.evidence['decisions'][0]['reason']=='frozen_after_repeated_confirmation'
    altered=replace(res,atoms=(replace(res.atoms[0],u_values=(.001,)),*res.atoms[1:]))
    thawed=propose_assignments(altered,predictions(altered,{'C1':'O'}),policy=policy,state=state)
    assert not thawed.state['assignment']['C1']['frozen']


def test_coverage_choices_are_separate_from_unmodified_raw_topk(tmp_path):
    res=carbon_twenty(tmp_path);raw=predictions(res)
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=True)
    first=propose_assignments(res,raw,policy=policy,proposed_elements={'C1':'Zn'})
    second=propose_assignments(res,raw,policy=policy,state=first.state,proposed_elements={'C1':'Zn'})
    assert second.overrides['C1']=='Zn'
    assert raw[0].topk[0].atomic_number==6
    assert second.evidence['decisions'][0]['proposal_source']=='explicit_protocol_choice'


def test_nonfinite_raw_prediction_is_blocked_before_a_proposal_can_be_applied(tmp_path):
    res=carbon_twenty(tmp_path);raw=list(predictions(res))
    raw[0]=ElementPrediction('C1',(ClassProbability(8,float('nan')),ClassProbability(6,0.)))
    with pytest.raises(ReconstructionBlocked,match='adaptive_prediction_probability_invalid'):
        propose_assignments(res,raw,policy=AdaptivePolicy(hysteresis_enabled=True))


def test_uiso_low_can_confirm_a_heavier_model_alternative_when_top1_stays_current(tmp_path):
    res=carbon_twenty(tmp_path);raw=predictions(res)
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=True)
    first=propose_assignments(res,raw,policy=policy)
    second=propose_assignments(res,raw,policy=policy,state=first.state)
    assert first.overrides['C1']=='C' and second.overrides['C1']=='N'
    row=next(row for row in second.evidence['decisions'] if row['site_id']=='C1')
    assert row['protocol_proposed_element']=='C' and row['proposed_element']=='N'
    assert row['proposal_source']=='uiso_supported_model_alternative'
    assert row['proposal_streak']==2 and row['u_evidence']['classification']=='low'
    assert raw[0].topk[0].atomic_number==6 and raw[0].topk[0].probability==.999999
    disabled=replace(policy,uiso_directional_proposals=False)
    held=propose_assignments(res,raw,policy=disabled,state=first.state)
    assert held.overrides['C1']=='C'


def test_directional_model_alternatives_remain_dormant_without_uiso_checks(tmp_path):
    res=carbon_twenty(tmp_path);raw=predictions(res)
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=False)
    state=None
    for _ in range(3):
        proposal=propose_assignments(res,raw,policy=policy,state=state)
        assert proposal.overrides['C1']=='C'
        state=proposal.state


def test_post_refinement_u_report_does_not_advance_or_mutate_state(tmp_path):
    res=carbon_twenty(tmp_path,u1=.001,u2=.5)
    policy=AdaptivePolicy(hysteresis_enabled=True,uiso_checks_enabled=True)
    state=None
    for _ in range(3):state=propose_assignments(res,predictions(res),policy=policy,state=state).state
    before=deepcopy(state);report=uiso_consistency(res,policy=policy,state=state)
    assert state==before and report['state_advanced'] is False
    sites={row['site_id']:row for row in report['sites']}
    assert sites['C1']['suggested_atomic_number_direction']=='higher'
    assert sites['C2']['suggested_atomic_number_direction']=='lower_or_prune'
    assert sites['C1']['reference_median']==pytest.approx(.05)
    assert sites['C1']['u_over_reference_median']==pytest.approx(.02)


def test_pruning_is_reversible_and_high_u_false_metals_are_not_automatically_protected(tmp_path):
    pytest.importorskip('scipy')
    atoms=[('Zn1','Zn',(.1,.1,.1),.9)]+[(f'C{i}','C',(.15+i*.02,.1,.1),.05) for i in range(1,20)]
    res=model(tmp_path,atoms);before=res.source.read_bytes()
    policy=AdaptivePolicy(pruning_enabled=True,prune_isolated=False,prune_close=False)
    first=propose_pruning(res,policy=policy)
    second=propose_pruning(res,policy=policy,state=first.state)
    assert first.site_ids==() and second.site_ids==('Zn1',)
    assert 'Zn1' in {row['site_id'] for row in second.evidence['decisions'] if row['decision']=='delete_proposal'}
    # Rejecting the trial means discarding its returned state; old counters and
    # immutable source bytes remain suitable for an exact controller rollback.
    assert first.state['pruning']['Zn1']['streak']==1
    assert res.source.read_bytes()==before
    protected=propose_pruning(res,policy=policy,state=first.state,protected_site_ids=('Zn1',))
    assert protected.site_ids==()


def test_isolation_exempts_metals_halogens_and_possible_water_oxygen(tmp_path):
    pytest.importorskip('scipy')
    atoms=[('C1','C',(.1,.1,.1),.05),('N1','N',(.4,.4,.4),.05),('O1','O',(.7,.1,.5),.05),
           ('I1','I',(.2,.7,.5),.05),('Zn1','Zn',(.8,.8,.8),.05)]
    res=model(tmp_path,atoms)
    policy=AdaptivePolicy(pruning_enabled=True,prune_high_u=False,prune_close=False,prune_confirmations=1,max_prune_fraction=1.)
    proposal=propose_pruning(res,policy=policy)
    assert set(proposal.site_ids)=={'C1','N1'}
    assert proposal.evidence['geometry']['isolated_site_count']==5
    assert proposal.evidence['geometry']['isolated_nonexempt_site_count']==2


def test_close_chain_keeps_nonadjacent_endpoints_and_periodic_images_are_seen(tmp_path):
    pytest.importorskip('scipy')
    res=model(tmp_path,[('C1','C',(.1,.1,.1),.05),('C2','C',(.14,.1,.1),.05),('C3','C',(.18,.1,.1),.05)])
    policy=AdaptivePolicy(pruning_enabled=True,prune_high_u=False,prune_isolated=False,prune_confirmations=1,max_prune_fraction=1.)
    proposal=propose_pruning(res,policy=policy)
    assert proposal.site_ids==('C2',)
    periodic=model(tmp_path,[('C1','C',(.005,.1,.1),.05),('C2','C',(.965,.1,.1),.05)])
    summary=geometry_summary(periodic,policy=policy)
    assert summary['close_pair_count']==1
    assert summary['close_pairs'][0]['distance_angstrom']==pytest.approx(.8)


def test_close_survivor_policy_retains_low_u_as_a_possible_under_typed_heavy_site(tmp_path):
    pytest.importorskip('scipy')
    res=model(tmp_path,[('C1','C',(.1,.1,.1),-.001),('C2','C',(.14,.1,.1),.05)])
    policy=AdaptivePolicy(pruning_enabled=True,prune_high_u=False,prune_isolated=False,
                          prune_confirmations=1,max_prune_fraction=1.)
    proposed=propose_pruning(res,policy=policy)
    assert proposed.site_ids==('C2',)
    legacy=propose_pruning(res,policy=replace(policy,close_survivor_policy='reference_consistency'))
    assert legacy.site_ids==('C1',)
    locked=propose_pruning(res,policy=policy,protected_site_ids=('C2',))
    assert locked.site_ids==('C1',)


def test_symmetry_neighbors_prevent_false_isolation(tmp_path):
    pytest.importorskip('scipy')
    # The inversion mate of C2 is 1.4 A from C1; their raw Cartesian separation is large.
    res=model(tmp_path,[('C1','C',(.1,.1,.1),.05),('C2','C',(.83,.9,.9),.05)],symm=('-X,-Y,-Z',))
    summary=geometry_summary(res,policy=AdaptivePolicy())
    assert summary['bonded_neighbor_counts']=={'C1':1,'C2':1}
    assert summary['symmetry_operation_count']==2


def test_declared_symmetry_must_preserve_the_supplied_metric(tmp_path):
    pytest.importorskip('scipy')
    res=model(tmp_path,[('C1','C',(.1,.1,.1),.05)],cell=(10,11,12,90,90,90),symm=('Y,X,Z',))
    with pytest.raises(ReconstructionBlocked,match='adaptive_symmetry_incompatible_with_cell_metric'):
        geometry_summary(res,policy=AdaptivePolicy())


def test_geometry_resource_bounds_block_without_truncating(tmp_path):
    pytest.importorskip('scipy')
    res=carbon_twenty(tmp_path)
    with pytest.raises(ReconstructionBlocked,match='adaptive_geometry_image_budget_exceeded'):
        geometry_summary(res,policy=AdaptivePolicy(max_image_candidates=10))


def test_disabled_pruning_needs_no_optional_geometry_provider(tmp_path,monkeypatch):
    import crystalx_core.refinement.adaptive as module
    res=carbon_twenty(tmp_path)
    def forbidden(*args,**kwargs):pytest.fail('disabled pruning reached a geometry provider')
    monkeypatch.setattr(module,'geometry_summary',forbidden)
    proposal=propose_pruning(res,policy=AdaptivePolicy())
    assert proposal.site_ids==() and proposal.evidence['geometry'] is None


def test_tiny_cell_is_blocked_before_allocating_periodic_translation_product(tmp_path):
    pytest.importorskip('scipy')
    res=model(tmp_path,[('C1','C',(.1,.1,.1),.05)],cell=(1e-9,1e-9,1e-9,90,90,90))
    with pytest.raises(ReconstructionBlocked,match='adaptive_geometry_image_budget_exceeded'):
        geometry_summary(res,policy=AdaptivePolicy())


def test_growth_requires_relative_height_and_bond_contact_and_has_recorded_early_allowance(tmp_path):
    pytest.importorskip('scipy')
    res=model(tmp_path,[('Zn1','Zn',(.1,.1,.1),.05)],peaks=[
        ('Q1',(.2,.1,.1),10.),('Q2',(.1,.2,.1),1.),('Q3',(.6,.6,.6),9.)])
    policy=AdaptivePolicy(growth_filter_enabled=True)
    selected,evidence=select_growth_peaks(res,policy=policy,round_number=1,limit=12,minimum_height=.5,minimum_distance=.8)
    assert selected==('Q1',) and evidence['effective_peak_limit']==24
    decisions={row['peak_id']:row for row in evidence['decisions']}
    assert decisions['Q2']['reason']=='relative_or_absolute_height'
    assert decisions['Q3']['reason']=='no_bond_distance_contact_to_existing_scaffold'
    later=select_growth_peaks(res,policy=policy,round_number=4,limit=12,minimum_height=.5,minimum_distance=.8)
    assert later[1]['effective_peak_limit']==12
    disabled=select_growth_peaks(res,policy=policy,round_number=1,limit=0,minimum_height=.5,minimum_distance=.8)
    assert disabled[0]==() and disabled[1]['effective_peak_limit']==0


def test_symmetry_duplicate_growth_peaks_are_not_selected_twice(tmp_path):
    pytest.importorskip('scipy')
    res=model(tmp_path,[('Zn1','Zn',(.1,.1,.1),.05)],symm=('-X,-Y,-Z',),peaks=[
        ('Q1',(.2,.1,.1),10.),('Q2',(.8,.9,.9),9.)])
    selected,evidence=select_growth_peaks(res,policy=AdaptivePolicy(growth_filter_enabled=True),round_number=1,
                                        limit=12,minimum_height=.5,minimum_distance=.8)
    assert selected==('Q1',)
    assert evidence['decisions'][1]['reason']=='too_close_to_selected_peak'
