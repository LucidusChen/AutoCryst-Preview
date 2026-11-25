"""Independent controller/file review over generated inputs and scripted tools only."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from crystalx_core.app.demo import write_demo_case
from crystalx_core.app.mock_toolchain import _mock_identity
from crystalx_core.app.models import RunMode
from crystalx_core.app.reconstruction_backend import ReconstructionBackendService, ReconstructionRequest
from crystalx_core.app.reconstruction_demo import synthetic_seed_text
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.inference import (
    ClassProbability, ElementInferenceRequest, ElementInferenceResult, ElementPrediction,
    InferenceDevice, ModelIdentity,
)
from crystalx_core.domain.tools import (
    BinaryArch, HostPlatform, ToolAvailabilityReport, ToolDiagnostic,
    ToolDiagnosticKind, ToolKind, ToolResolution,
)
from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix
from crystalx_core.io.shelx_res import read_res
from crystalx_core.refinement.cycle import refine_file
from crystalx_core.tools.adapter import ExternalToolAdapter
from crystalx_core.tools.runner import CancelToken, ProcessRunner, RawResult
from crystalx_core.workflow.reconstruction import ReconstructionBlocked, digest
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.workspace.hashing import sha256_file


class ReviewExecutor:
    def __init__(self, *, residuals=(0.2,), initial_model=None, callback=None, rejected_shift=False):
        self.calls = []
        self.residuals = residuals
        self.initial_model = initial_model or synthetic_seed_text()
        self.callback = callback
        self.refine_calls = 0
        self.raw_outputs = {}
        self.rejected_shift = rejected_shift

    def execute(self, plan, env):
        self.calls.append(plan)
        if plan.tool_kind is ToolKind.SHELXT:
            stem = plan.argv[1]
            (plan.cwd / f"{stem}_a.res").write_text(self.initial_model)
            (plan.cwd / f"{stem}_a.hkl").write_bytes((plan.cwd / f"{stem}.hkl").read_bytes())
        elif plan.tool_kind is ToolKind.SHELXL:
            self.refine_calls += 1
            r = self.residuals[min(self.refine_calls - 1, len(self.residuals) - 1)]
            text = (plan.cwd / "trial.ins").read_text()
            if self.rejected_shift and self.refine_calls == 2:
                lines = text.splitlines()
                for n, line in enumerate(lines):
                    row = line.split()
                    if row and row[0] == "C1":
                        row[2] = f"{float(row[2])+0.1:.8f}"
                        lines[n] = " ".join(row)
                text = "\n".join(lines) + "\n"
            summary = (f"R1 = {r:.6f} for 800 Fo > 4sig(Fo) and {r:.6f} for all 1000 data\n"
                       f"wR2 = {2*r:.6f}, GooF = S = 1.000000, Restrained GooF = 1.000000 for all data\n")
            raw = text + "".join("REM " + line + "\n" for line in summary.splitlines())
            raw += "REM 8 parameters refined using 0 restraints\n"
            (plan.cwd / "trial.res").write_text(raw)
            self.raw_outputs[plan.cwd / "trial.res"] = raw.encode()
            (plan.cwd / "trial.lst").write_text(summary + "8 / 8 parameters\nLeast-squares cycle 4\n")
            (plan.cwd / "trial.cif").write_text("data_generated_synthetic_model\n")
            (plan.cwd / "trial.fcf").write_text("data_generated_synthetic_reflections\n")
        else:
            pytest.fail("review fixture attempted an unconfigured tool")
        if self.callback:
            self.callback(plan)
        return RawResult(exit_code=0, stdout="scripted synthetic tool output", stderr="")


class ReviewToolchain(Toolchain):
    def __init__(self, executor, *, blocked=False, mode=RunMode.MOCK_DEMO):
        self.executor, self.blocked, self.mode = executor, blocked, mode

    def availability(self):
        return ToolAvailabilityReport(HostPlatform("synthetic", BinaryArch.UNKNOWN), {
            kind.value: ToolResolution(kind, None if self.blocked else _mock_identity(kind),
                ToolDiagnostic(ToolDiagnosticKind.QUARANTINED if self.blocked else ToolDiagnosticKind.OK,
                               "synthetic static blocker" if self.blocked else "generated review fixture"))
            for kind in (ToolKind.SHELXT, ToolKind.SHELXL)
        })

    def _runner(self):
        return ProcessRunner(self.executor, base_env={})


class ReviewAssigner:
    def __init__(self, callback=None):
        self.calls = []
        self.callback = callback
        self.inputs = []

    def assign(self, path, *, candidate_elements, include_q_peaks=True):
        if self.callback:
            self.callback(path)
        model = read_res(path)
        self.calls.append(path)
        self.inputs.append(model)
        sites = tuple(a.label for a in model.atoms) + tuple(p.label for p in model.q_peaks)
        zs = tuple(ATOMIC_NUMBERS[model.element_of(a)] for a in model.atoms) + (6,) * len(model.q_peaks)
        cell = model.header.cell
        matrix = orthogonalisation_matrix((cell.a,cell.b,cell.c,cell.alpha,cell.beta,cell.gamma))
        positions = tuple(frac_to_cart(matrix,(a.x,a.y,a.z)) for a in (*model.atoms,*model.q_peaks))
        candidates = tuple(sorted(ATOMIC_NUMBERS[e] for e in candidate_elements if e != "H"))
        request = ElementInferenceRequest(sites,zs,positions,candidates,len(sites))
        predictions = tuple(ElementPrediction(site,(ClassProbability(z if z in candidates else candidates[0],.8),
            ClassProbability(next(other for other in candidates if other != z),.2))) for site,z in zip(sites,zs))
        result = ElementInferenceResult(predictions,ModelIdentity(digest("review-synthetic-weights"),(),8,98,1,8),
            InferenceDevice.CPU,InferenceDevice.CPU,False,"","synthetic",0,"synthetic-review-only",(),0.,
            digest(request),candidates,"no-torch-test-provider")
        payload = {"request":asdict(request),"result":asdict(result),"model_sha256":sha256_file(path)}
        return SimpleNamespace(request=request,result=result,model_sha256=payload["model_sha256"],
                               to_record=lambda:payload)


@pytest.fixture
def review_backend(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git","init","-q",str(repo)],check=True,capture_output=True)
    (repo / ".gitignore").write_text("/runs/\n")
    ins,hkl = write_demo_case(tmp_path / "generated-inputs")
    def make(*,executor=None,assigner=None,assigner_factory=None,mode=RunMode.MOCK_DEMO,blocked=False):
        executor = executor or ReviewExecutor()
        assigner = assigner or ReviewAssigner()
        chain = ReviewToolchain(executor,blocked=blocked,mode=mode)
        service = ReconstructionBackendService(repo,repo/"runs/review",toolchain_factory=lambda _:chain,
            assigner_factory=assigner_factory or (lambda _:assigner))
        return service,executor,assigner
    return make,ins,hkl


def _request(ins,hkl,**kwargs):
    return ReconstructionRequest(ins,hkl,mode=RunMode.MOCK_DEMO,**kwargs)


def test_fixed_iterations_do_not_stop_early_on_stability(review_backend):
    make,ins,hkl = review_backend
    service,executor,assigner = make()
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(iterations=4,max_new_peaks_per_round=0)))
    assert result["phase"] == "completed",result
    assert result["stop_code"] == "fixed_rounds_completed"
    assert executor.refine_calls == 4 and len(assigner.calls) == 4
    assert not result["scientifically_validated"]


def test_automatic_two_stable_transitions_require_three_successful_rounds(review_backend):
    make,ins,hkl = review_backend
    service,executor,assigner = make()
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(max_new_peaks_per_round=0)))
    assert result["phase"] == "completed",result
    assert result["stop_code"] == "movable_sites_r1_and_growth_stable"
    assert executor.refine_calls == 3 and len(assigner.calls) == 3
    assert result["stable_transitions"] == 2
    assert not result["automatic_stop_is_scientific_convergence"]


def test_scientific_gate_waivers_never_persist_validated_candidate_status(review_backend):
    make,ins,hkl = review_backend
    service,_,_ = make()
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(iterations=1,max_new_peaks_per_round=0)))
    candidates = list((Path(result["run_root"])/"candidates").glob("*/candidate.json"))
    assert candidates,result
    assert all(json.loads(path.read_text())["status"] != "validated" for path in candidates)
    workflow = json.loads((Path(result["run_root"])/"workflow-result.json").read_text())
    assert workflow["stop_reason"]["kind"] != "converged"


def test_raw_refinement_outputs_survive_normalization_and_export(review_backend):
    make,ins,hkl = review_backend
    service,executor,_ = make()
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(iterations=1,max_new_peaks_per_round=0)))
    assert result["phase"] == "completed",result
    assert executor.raw_outputs
    assert all(path.read_bytes() == raw for path,raw in executor.raw_outputs.items())


def test_rejected_trial_does_not_become_next_round_model(review_backend):
    make,ins,hkl = review_backend
    executor = ReviewExecutor(residuals=(.2,.5,.2,.2),rejected_shift=True)
    service,_,assigner = make(executor=executor)
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(rejection_backoff=False,iterations=4,max_new_peaks_per_round=0)))
    assert result["phase"] == "completed",result
    assert result["history"][1]["decision"] == "REJECT_R_REGRESSION"
    before = next(a for a in assigner.inputs[0].atoms if a.label == "C1")
    after = next(a for a in assigner.inputs[2].atoms if a.label == "C1")
    assert (after.x,after.y,after.z) == (before.x,before.y,before.z)


def test_rejection_breaks_consecutive_stability_without_refunding_attempts(review_backend):
    make,ins,hkl = review_backend
    executor = ReviewExecutor(residuals=(.2,.2,.5,.2,.2))
    service,_,assigner = make(executor=executor)
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(rejection_backoff=False,max_new_peaks_per_round=0)))
    assert result["phase"] == "completed",result
    assert result["stop_code"] == "movable_sites_r1_and_growth_stable"
    assert executor.refine_calls == 5 and len(assigner.calls) == 5
    assert result["history"][2]["decision"] == "REJECT_R_REGRESSION"
    assert result["history"][3]["stable_transition_count"] == 1
    assert result["history"][4]["stable_transition_count"] == 2


def test_assigner_cannot_repin_an_input_it_changed(review_backend):
    make,ins,hkl = review_backend
    def mutate(path):
        text = path.read_text()
        lines = text.splitlines()
        for n,line in enumerate(lines):
            row = line.split()
            if row and row[0] == "CELL":
                row[2] = str(float(row[2])+1.0)
                lines[n] = " ".join(row)
        path.write_text("\n".join(lines)+"\n")
    service,executor,_ = make(assigner=ReviewAssigner(mutate))
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(iterations=1,max_new_peaks_per_round=0)))
    assert result["phase"] != "completed",result
    assert executor.refine_calls == 0
    assert result["ai_solved_res"] is None


def test_accepted_source_hash_change_blocks_before_refinement(review_backend):
    make,ins,hkl = review_backend
    def mutate(path):
        accepted = path.parents[2]/"seed.res"
        with accepted.open("a") as handle:
            handle.write("REM external mutation of accepted synthetic source\n")
    service,executor,_ = make(assigner=ReviewAssigner(mutate))
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(iterations=1,max_new_peaks_per_round=0)))
    assert result["phase"] != "completed",result
    assert executor.refine_calls == 0


def test_metal_seed_assignment_obeys_cancellation_after_constructor(review_backend):
    make,ins,hkl = review_backend
    cancel = CancelToken()
    initial = "\n".join(line for line in synthetic_seed_text().splitlines() if not line.startswith("Zn1 "))+"\n"
    assigner = ReviewAssigner()
    def factory(_):
        cancel.cancel("synthetic cancel during assigner construction")
        return assigner
    service,executor,_ = make(executor=ReviewExecutor(initial_model=initial),assigner=assigner,
                              assigner_factory=factory)
    result = service.run(_request(ins,hkl,metal_only=True,cycle=CyclePolicy(iterations=1)),cancel=cancel)
    assert result["phase"] == "cancelled",result
    assert not assigner.calls and executor.refine_calls == 0


def test_cancelled_metal_seed_assignment_keeps_charged_inference_count(review_backend):
    make,ins,hkl=review_backend
    cancel=CancelToken()
    initial="\n".join(line for line in synthetic_seed_text().splitlines() if not line.startswith("Zn1 "))+"\n"
    assigner=ReviewAssigner(callback=lambda _:cancel.cancel("cancel after entering inference"))
    service,executor,_=make(executor=ReviewExecutor(initial_model=initial),assigner=assigner)
    result=service.run(_request(ins,hkl,metal_only=True,cycle=CyclePolicy(iterations=1)),cancel=cancel)
    assert result["phase"]=="cancelled" and result["inference_calls"]==1
    assert len(assigner.calls)==1 and executor.refine_calls==0
    assert service.status(result["run_id"])["inference_calls"]==1


def test_blocked_real_tools_do_not_launch_infer_or_open_reference(review_backend,tmp_path,monkeypatch):
    make,ins,hkl = review_backend
    marker = tmp_path/"synthetic-held-out-marker.res"
    marker.write_text("never a solve input\n")
    original_open = Path.open
    def guard(path,*args,**kwargs):
        if path == marker:
            pytest.fail("blocked solve opened a held-out marker")
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    def forbidden(_):pytest.fail("blocked solve constructed inference")
    service,executor,_ = make(mode=RunMode.REAL,blocked=True,assigner_factory=forbidden)
    result = service.run(ReconstructionRequest(ins,hkl,cycle=CyclePolicy(iterations=1)))
    assert result["phase"] == "blocked",result
    assert not executor.calls


@pytest.mark.parametrize("name", ["trial.lst","trial.cif","trial.fcf"])
def test_late_output_alias_is_rejected_before_read_or_export(tmp_path,monkeypatch,name):
    source = tmp_path/"generated-source.res"
    source.write_text(synthetic_seed_text())
    _,hkl = write_demo_case(tmp_path/"generated-inputs")
    marker = tmp_path/"synthetic-held-out-marker.txt"
    marker.write_text("never read through a process artifact\n")
    executor = ReviewExecutor()
    chain = ReviewToolchain(executor)
    wrapped = chain.adapter(ToolKind.SHELXL)
    original_invoke = wrapped.invoke
    def late_alias(*args,**kwargs):
        result = original_invoke(*args,**kwargs)
        path = args[0].cwd/name
        path.unlink()
        path.symlink_to(marker)
        return result
    monkeypatch.setattr(wrapped,"invoke",late_alias)
    original_open = Path.open
    def guard(path,*args,**kwargs):
        if path.exists() and path.resolve() == marker:
            pytest.fail("late process alias exposed the held-out marker")
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ReconstructionBlocked):
        refine_file(model_path=source,hkl_path=hkl,directory=tmp_path/"trial",adapter=wrapped,
                    cycles=2,timeout_s=5,cancel=CancelToken())


@pytest.mark.parametrize("failure", ["commit_journal", "outer_status"])
def test_failed_commit_persistence_does_not_export_fresh_candidate(review_backend,monkeypatch,failure):
    from crystalx_core.workspace.store import RunWorkspace
    make,ins,hkl = review_backend
    original_write = RunWorkspace.write_json
    injected = []
    def fail_write(self,relative,value,*args,**kwargs):
        target = str(relative)
        fail = ((failure == "commit_journal" and target == "cycles/round-001/commit.json")
                or (failure == "outer_status" and target == "state.json" and value.get("phase") == "running"
                    and value.get("round") == 1))
        if fail and not injected:
            injected.append(target)
            raise OSError("synthetic persistence fault")
        return original_write(self,relative,value,*args,**kwargs)
    monkeypatch.setattr(RunWorkspace,"write_json",fail_write)
    service,executor,assigner = make()
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(iterations=1,max_new_peaks_per_round=0)))
    assert injected
    assert result["phase"] == "failed",result
    assert result["ai_solved_res"] is None
    assert not (Path(result["run_root"])/"outputs/ai_solved.res").exists()
    assert result["tool_calls"] == 2 and result["inference_calls"] == 1
    assert executor.refine_calls == 1 and len(assigner.calls) == 1
    assert result["history"][0]["decision"] == "REJECTED_BY_CONTROLLER"
    assert result["stop_code"] != "fixed_rounds_completed"


def test_failed_successor_commit_restores_previous_model_and_stability(review_backend,monkeypatch):
    from crystalx_core.workspace.store import RunWorkspace
    make,ins,hkl = review_backend
    original_write = RunWorkspace.write_json
    def fail_write(self,relative,value,*args,**kwargs):
        if str(relative) == "cycles/round-003/commit.json":
            raise OSError("synthetic third-commit fault")
        return original_write(self,relative,value,*args,**kwargs)
    monkeypatch.setattr(RunWorkspace,"write_json",fail_write)
    service,executor,_ = make()
    result = service.run(_request(ins,hkl,cycle=CyclePolicy(max_new_peaks_per_round=0)))
    assert result["phase"] == "failed",result
    assert result["stop_code"] != "movable_sites_r1_and_growth_stable"
    assert result["stable_transitions"] == 1
    assert result["tool_calls"] == 4 and result["inference_calls"] == 3
    assert [entry["decision"] for entry in result["history"]] == [
        "RETAIN_UNVALIDATED","RETAIN_UNVALIDATED","REJECTED_BY_CONTROLLER"]
    root=Path(result["run_root"])
    previous=root/result["history"][1]["model_path"]
    exported=Path(result["ai_solved_res"])
    assert exported.read_bytes() == previous.read_bytes()
    assert "round-003" not in str(previous)
    abort=json.loads((root/"cycles/round-003/abort.json").read_text())
    assert abort["restored_model_path"] == str(previous)
    assert abort["restored_model_sha256"] == sha256_file(previous)
    assert abort["stable_transitions"] == 1 and abort["stop_code"] == ""
