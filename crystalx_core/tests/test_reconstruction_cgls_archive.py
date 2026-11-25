"""Reproduce the zero-exit native CGLS/ACTA failure and verify safe archive output."""
from pathlib import Path
import pytest
from test_reconstruction_backend import backend, run_backend, FixtureToolchain
from crystalx_core.app.reconstruction_demo import synthetic_seed_text
from crystalx_core.domain.tools import ToolKind
from crystalx_core.io.shelxl_lst import parse_lst_text
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.tools.runner import RawResult
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.private_acceptance.sealing import verify_seal


def native_boundary(*, force_incompatibility=False, move_archive=False):
    chain = FixtureToolchain()
    execute = chain.executor.execute
    def run(plan, env, **kwargs):
        outcome = execute(plan, env, **kwargs)
        if plan.tool_kind is not ToolKind.SHELXL:
            return outcome
        text = (plan.cwd / 'trial.ins').read_text()
        cgls = 'CGLS ' in text
        if 'DAMP 0 0\n' in text and '\nACTA\n' in text:
            message = '** Second DAMP parameter too small for ACTA **\n'
            (plan.cwd / 'trial.res').write_bytes(b'')
            (plan.cwd / 'trial.lst').write_text(message)
            return RawResult(0, message, '')
        if force_incompatibility or (cgls and '\nACTA\n' in text):
            message = '** CGLS incompatible with ACTA **\n'
            (plan.cwd / 'trial.res').write_bytes(b'')
            (plan.cwd / 'trial.lst').write_text(message)
            return RawResult(0, message, '')
        if cgls:
            (plan.cwd / 'trial.cif').unlink()
        elif move_archive and plan.cwd.name == 'archive-shelxl':
            model = plan.cwd / 'trial.res'
            rows = []
            for line in model.read_text().splitlines():
                fields = line.split()
                if fields and fields[0] == 'C1':
                    fields[2] = str(float(fields[2]) + .01)
                    line = ' '.join(fields)
                rows.append(line)
            model.write_text('\n'.join(rows) + '\n')
        return outcome
    chain.executor.execute = run
    return chain


def test_cgls_never_writes_acta_even_if_present_in_source():
    text, changes, _, _ = prepare_refinement_ins(
        synthetic_seed_text().replace('HKLF 4', 'ACTA\nHKLF 4'), RefinementSettings(method='cgls'))
    assert 'CGLS 4' in text and '\nACTA\n' not in text
    assert 'LIST 4' in text and 'FMAP 2' in text
    assert any('deferred ACTA/CIF' in change for change in changes)
    assert parse_lst_text('** CGLS incompatible with ACTA **').has_fatal_error


def test_retained_cgls_seals_res_fcf_but_blocks_unapproved_cif_refinement(backend):
    chain = native_boundary()
    result, service, _, _ = run_backend(backend, chain=chain,
        policy=CyclePolicy(iterations=1, refinement_method='cgls'))
    assert result['phase'] == 'blocked', result
    assert 'cgls_native_cif_requires_recorded_nonzero_ls_refinement' in result['stop_detail']
    assert len(result['history']) == 1 and result['tool_calls'] == 2
    calls = [p for p in chain.executor.calls if p.tool_kind is ToolKind.SHELXL]
    assert len(calls) == 1
    assert '\nACTA\n' not in (calls[0].cwd/'trial.ins').read_text()
    seal = verify_seal(Path(result['run_root']), private_root=service.runs_root,
                       expected_seal_sha256=result['seal_sha256'])
    assert seal.terminal_state == result['phase'] == service.status(result['run_id'])['phase'] == 'blocked'
    assert seal.file_for_role('model') and seal.file_for_role('fcf')
    assert not (Path(result['run_root'])/'outputs/ai_solved.cif').exists()
    assert (Path(result['run_root'])/'archive-blocked.json').is_file()
    assert not (Path(result['run_root'])/'archive-shelxl').exists()


def test_native_incompatibility_is_not_hidden_by_empty_res_parse(backend):
    result, _, _, _ = run_backend(backend, chain=native_boundary(force_incompatibility=True),
                                  policy=CyclePolicy(iterations=1, refinement_method='cgls'))
    assert result['stop_code'] == 'refinement_failed', result
    assert 'CGLS incompatible with ACTA' in result['stop_detail']
    assert result['ai_solved_res'] is None


@pytest.mark.parametrize('notice', ['** Second DAMP parameter too small for ACTA **', '** L.S. 0 incompatible with ACTA **'])
def test_native_zero_damp_acta_failure_is_recognized(notice):
    parsed = parse_lst_text(notice)
    assert parsed.has_fatal_error and not parsed.notices
