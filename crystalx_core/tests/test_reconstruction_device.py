"""Run-level device decisions: startup auto selection, durable pin, no replay."""
import json
from types import SimpleNamespace

import pytest

from crystalx_core.app import reconstruction_device as devices
from crystalx_core.app import reconstruction_backend as backend_module
from crystalx_core.app.reconstruction_backend import ReconstructionBackendService, ReconstructionRequest
from crystalx_core.app.models import RunMode
from crystalx_core.domain.inference import InferenceDevice
from test_reconstruction_backend import backend, FixtureToolchain, FixtureAssigner


@pytest.mark.parametrize('host,arch,available,expected', [
    ('Darwin', 'arm64', True, 'mps'),
    ('Darwin', 'arm64', False, 'cpu'),
    ('Darwin', 'x86_64', True, 'cpu'),
    ('Linux', 'aarch64', True, 'cpu'),
])
def test_auto_uses_available_mps_only_on_apple_silicon(monkeypatch, host, arch, available, expected):
    monkeypatch.setattr(devices.platform, 'system', lambda: host)
    monkeypatch.setattr(devices.platform, 'machine', lambda: arch)
    calls = []
    def select(requested):
        calls.append(requested)
        return SimpleNamespace(used=InferenceDevice.MPS if available else InferenceDevice.CPU,
                               reason='' if available else 'MPS unavailable at startup')
    monkeypatch.setattr(devices, '_native_selection', select)
    decision = devices.resolve_inference_device('auto')
    assert decision['device_used'] == expected and not decision['operator_cpu_fallback_allowed']
    assert calls == (['auto'] if host == 'Darwin' and arch == 'arm64' else [])


def test_explicit_cpu_and_synthetic_auto_do_not_import_accelerator(monkeypatch):
    def forbidden(_):raise AssertionError('accelerator import/probe is forbidden')
    monkeypatch.setattr(devices, '_native_selection', forbidden)
    assert devices.resolve_inference_device('cpu')['device_used'] == 'cpu'
    assert devices.resolve_inference_device('auto', synthetic=True)['device_used'] == 'cpu'
    assert devices.resolve_inference_device('mps', synthetic=True)['device_blocker']


def test_unavailable_explicit_mps_blocks_but_auto_can_start_on_cpu(monkeypatch):
    monkeypatch.setattr(devices.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(devices.platform, 'machine', lambda: 'arm64')
    def missing(_):raise RuntimeError('strict MPS unavailable')
    monkeypatch.setattr(devices, '_native_selection', missing)
    explicit = devices.resolve_inference_device('mps')
    assert explicit['device_used'] is None and explicit['device_blocker']
    assert devices.resolve_inference_device('auto')['device_used'] == 'cpu'


def test_one_resolution_is_pinned_before_worker_and_survives_restart(backend, monkeypatch):
    repo, ins, hkl = backend
    resolutions = []
    def select(requested, **kwargs):
        resolutions.append(requested)
        return devices.resolve_inference_device(requested, **kwargs)
    monkeypatch.setattr(backend_module, 'resolve_inference_device', select)
    seen = []
    def factory(request):
        seen.append(request.device)
        return FixtureAssigner()
    service = ReconstructionBackendService(repo, repo/'runs/devices',
        toolchain_factory=lambda _: FixtureToolchain(), assigner_factory=factory)
    request = ReconstructionRequest(ins,hkl,mode=RunMode.MOCK_DEMO)
    result = service.run(request)
    assert result['phase'] == 'completed', result
    assert resolutions == ['auto'] and seen == ['cpu']
    root = service.run_root(result['run_id'])
    persisted = json.loads((root/'request.json').read_text())
    assert persisted['device'] == 'cpu'
    assert ReconstructionRequest.from_record(persisted).device == 'cpu'
    restarted = ReconstructionBackendService(repo, repo/'runs/devices')
    for record in (result, restarted.status(result['run_id']), restarted.summary(result['run_id']),
                   json.loads((root/'inference-device.json').read_text()),
                   json.loads((root/'preview.json').read_text())):
        assert record['device_requested'] == 'auto' and record['device_used'] == 'cpu'
        assert record['operator_cpu_fallback_allowed'] is False
    assert resolutions == ['auto']


def test_device_error_has_durable_block_and_no_tool_or_assigner(backend):
    repo, ins, hkl = backend
    def forbidden(_):raise AssertionError('no numerical provider should be created')
    service = ReconstructionBackendService(repo, repo/'runs/blocked-device',
        toolchain_factory=forbidden, assigner_factory=forbidden)
    result = service.run(ReconstructionRequest(ins,hkl,device='mps',mode=RunMode.MOCK_DEMO))
    assert result['phase'] == 'blocked' and result['device_used'] is None
    state = service.status(result['run_id'])
    assert state['tool_calls'] == state['inference_calls'] == 0
    assert state['device_blocker'] == result['device_blocker']


def test_selected_mps_failure_is_not_retried_on_cpu(backend, monkeypatch):
    repo, ins, hkl = backend
    pinned = dict(device_requested='auto',device_used='mps',device_selection_reason='test probe',
                  device_blocker=None,operator_cpu_fallback_allowed=False)
    monkeypatch.setattr(backend_module,'resolve_inference_device',lambda *a,**k:pinned)
    seen = []
    def fail(request):
        seen.append(request.device)
        raise RuntimeError('unsupported MPS operation')
    service = ReconstructionBackendService(repo,repo/'runs/failure',
        toolchain_factory=lambda _:FixtureToolchain(),assigner_factory=fail)
    result = service.run(ReconstructionRequest(ins,hkl,mode=RunMode.MOCK_DEMO))
    assert result['phase'] == 'failed' and seen == ['mps']
    assert result['device_used'] == 'mps'  # Never rewritten as CPU after failure.
