"""Public termination and read-only frontend contract, using synthetic inputs only."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from test_reconstruction_backend import backend, run_backend, FixtureToolchain
from crystalx_core.app.reconstruction_backend import FRONTEND_CONTRACT
from crystalx_core.app.reconstruction_stops import STOP_CODES, STOP_CATEGORIES, stop_record
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.domain.tools import ToolKind


def test_registry_is_complete_described_and_immutable():
    assert {info.category for info in STOP_CODES.values()} == set(STOP_CATEGORIES)
    for code, info in STOP_CODES.items():
        assert len(info.description) > 20
        result = stop_record(code)
        assert result['phase'] == info.phase and result['stop_category'] == info.category
    with pytest.raises(TypeError):
        STOP_CODES['invented'] = object()
    guarded = stop_record('adaptive_geometry_image_budget_exceeded')
    assert guarded['stop_code'] == 'reconstruction_blocked'
    assert guarded['stop_category'] == 'blocked'  # A geometry prerequisite is not run-budget completion.
    assert guarded['stop_detail'] == 'adaptive_geometry_image_budget_exceeded'
    assert stop_record('shelxl_process_failed:nonzero')['stop_code'] == 'refinement_failed'


@pytest.mark.parametrize('policy,code', [
    (CyclePolicy(), 'movable_sites_r1_and_growth_stable'),
    (CyclePolicy(stopping_policy='legacy_important'), 'important_coordinates_and_r1_stable'),
    (CyclePolicy(iterations=1), 'fixed_rounds_completed'),
    (CyclePolicy(max_rounds=1), 'round_safety_budget_exhausted'),
])
def test_status_summary_and_persisted_stop_share_registry_phase(backend, policy, code):
    result, service, _, _ = run_backend(backend, policy=policy)
    state = service.status(result['run_id'])
    assert result['stop_code'] == state['stop_code'] == code
    assert result['phase'] == state['phase'] == STOP_CODES[code].phase
    assert result['frontend_contract'] == FRONTEND_CONTRACT == '2026-11-20'
    assert result['stop_description'] == state['stop_description'] == STOP_CODES[code].description
    root = service.run_root(result['run_id'])
    assert root == Path(result['run_root'])
    for bad in ('../x', str(root), 'r-20261113T000000Z-xxxxxxxx'):
        with pytest.raises(ValueError):service.run_root(bad)
    # Enrich a historical read without modifying the stored bytes or known contract marker.
    summary_path = root / 'summary.json'
    historical = dict(result, frontend_contract='2026-01-01')
    summary_path.write_text(json.dumps(historical))
    before = summary_path.read_bytes()
    assert service.summary(result['run_id'])['frontend_contract'] == '2026-01-01'
    assert summary_path.read_bytes() == before


def test_preview_and_initial_guard_have_documented_terminal_codes(backend):
    result, service, _, _ = run_backend(backend, chain=FixtureToolchain(missing=(ToolKind.SHELXT,)))
    assert result['stop_code'] == 'preview_blocked'
    assert service.status(result['run_id'])['phase'] == STOP_CODES['preview_blocked'].phase
    _, ins, hkl = backend
    from crystalx_core.app.reconstruction_backend import ReconstructionRequest
    from crystalx_core.app.models import RunMode
    request = ReconstructionRequest(ins,hkl,mode=RunMode.MOCK_DEMO)
    preview = service.preview(request)
    assert preview['stop_description'] and preview['stop_category'] == 'blocked'


def test_history_reports_nonpositive_inference_input_sites_without_changing_u(backend, monkeypatch):
    import test_reconstruction_backend as fixtures
    original_read = fixtures.read_snapshot
    monkeypatch.setattr(fixtures, 'read_snapshot',
        lambda *args, **kwargs: original_read(*args, **dict(kwargs, allow_nonpositive_u=True)))
    chain = FixtureToolchain()
    execute = chain.executor.execute
    def negative_seed(plan, env, **kwargs):
        raw = execute(plan, env, **kwargs)
        if plan.tool_kind is ToolKind.SHELXT:
            seed = plan.cwd / (plan.argv[1] + '_a.res')
            seed.write_text(seed.read_text().replace('11.00000 0.03000', '11.00000 -0.01000', 1))
        return raw
    chain.executor.execute = negative_seed
    result, service, _, _ = run_backend(backend, chain=chain, policy=CyclePolicy(iterations=1))
    assert result['phase'] == 'completed', result
    assert result['history'][0]['nonpositive_u_site_ids'] == ['Zn1']
    assert service.status(result['run_id'])['history'][0]['nonpositive_u_site_ids'] == ['Zn1']
