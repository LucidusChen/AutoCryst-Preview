"""Generated end-to-end portfolio/cycle fixtures; none is a scientific program."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import pytest

from crystalx_core.app.demo import DEMO_CELL, write_demo_case
from crystalx_core.app.models import RunMode
from crystalx_core.app.mock_toolchain import _mock_identity
from crystalx_core.app.reconstruction_backend import ReconstructionBackendService, ReconstructionRequest
from crystalx_core.app.reconstruction_demo import SyntheticInference, SyntheticPreparer, synthetic_seed_text
from crystalx_core.app.reconstruction_runtime_inference import RuntimeAssignment
from crystalx_core.app.solve_routes import InitialRouteConfig, InitialStrategy, PhasingDataset, ProteinPhasingConfig
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolAvailabilityReport, ToolDiagnostic, ToolDiagnosticKind, ToolKind, ToolResolution
from crystalx_core.io.shelx_res import read_res
from crystalx_core.private_acceptance.sealing import verify_seal
from crystalx_core.refinement.cycle import assign_to_model, cell_values, checked_res, periodic_distance, promote_initial_peaks, setting_key
from crystalx_core.domain.inference import ClassProbability, ElementPrediction
from crystalx_core.tools.runner import ProcessRunner, RawResult
from crystalx_core.workflow.reconstruction import DensityObservation, DensityPeak, digest, read_snapshot
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_core.workspace.hashing import sha256_file


class FixtureExecutor:
    supports_control=True
    def __init__(self,*,peaks=False):self.calls=[];self.peaks=peaks
    def execute(self,plan,env,*,cancel=None,stdout_path=None,stderr_path=None):
        self.calls.append(plan)
        if cancel and cancel.cancelled:return RawResult(None,"","",cancelled=True)
        root=plan.cwd;stem=plan.argv[1]
        if plan.tool_kind is ToolKind.SHELXC:
            (root/(stem+".hkl")).write_bytes((root/"original.hkl").read_bytes())
            (root/(stem+"_fa.ins")).write_text(synthetic_seed_text())
            (root/(stem+"_fa.hkl")).write_text("generated FA/alpha placeholder\n")
        elif plan.tool_kind is ToolKind.SHELXE:
            import gemmi
            structure=gemmi.Structure();structure.cell=gemmi.UnitCell(*DEMO_CELL)
            structure.spacegroup_hm="P 1 21 1"
            model=gemmi.Model("1");chain=gemmi.Chain("A");residue=gemmi.Residue()
            residue.name="ALA";residue.seqid=gemmi.SeqId(1," ")
            for name,element,xyz in [("CA","C",(2.,3.,2.)),("N","N",(3.,3.,2.))]:
                atom=gemmi.Atom();atom.name=name;atom.element=gemmi.Element(element);atom.pos=gemmi.Position(*xyz)
                residue.add_atom(atom)
            chain.add_residue(residue);model.add_chain(chain);structure.add_model(model)
            prefix=stem+("_i" if "-i" in plan.argv else "")
            (root/(prefix+".pdb")).write_text("REMARK CC = 30.0\n"+structure.make_pdb_string())
            (root/(prefix+".phs")).write_text("generated phases placeholder\n")
        elif plan.tool_kind in (ToolKind.SHELXT,ToolKind.SHELXS,ToolKind.SHELXD):
            suffix="_a.res" if plan.tool_kind is ToolKind.SHELXT else ".res"
            (root/(stem+suffix)).write_text(synthetic_seed_text())
            if plan.tool_kind is ToolKind.SHELXT:(root/(stem+"_a.hkl")).write_bytes((root/(stem+".hkl")).read_bytes())
        elif plan.tool_kind is ToolKind.SHELXL:
            text=(root/"trial.ins").read_text()
            atoms=[]
            for row in text.splitlines():
                fields=row.split()
                if len(fields)==7:
                    try:int(fields[1]);float(fields[2])
                    except ValueError:continue
                    atoms.append(fields)
            summary="R1 = 0.300000 for 800 Fo > 4sig(Fo) and 0.300000 for all 1000 data\n"
            summary+="wR2 = 0.600000, GooF = S = 1.030000, Restrained GooF = 1.030000 for all data\n"
            rem="".join("REM "+line+"\n" for line in summary.splitlines())
            rem+=f"REM {4*len(atoms)} parameters refined using 0 restraints\n"
            q="Q1 1 0.45 0.5 0.35 11 0.05 2.0\n" if self.peaks and not any(a[0].startswith("X") for a in atoms) else ""
            (root/"trial.res").write_text(text+rem+q)
            (root/"trial.lst").write_text(summary+f"{4*len(atoms)} / {4*len(atoms)} parameters\n")
            (root/"trial.cif").write_text("data_synthetic\n_audit_creation_method 'software fixture'\n")
            (root/"trial.fcf").write_text("data_synthetic\n")
        else:raise AssertionError(plan.tool_kind)
        return RawResult(0,"synthetic fixture only\n","")


class FixtureToolchain(Toolchain):
    mode=RunMode.MOCK_DEMO
    def __init__(self,*,peaks=False,missing=()):self.executor=FixtureExecutor(peaks=peaks);self.missing=set(missing)
    def availability(self):
        values={kind.value:ToolResolution(kind,None if kind in self.missing else _mock_identity(kind),
            ToolDiagnostic(ToolDiagnosticKind.NOT_FOUND if kind in self.missing else ToolDiagnosticKind.OK,"generated fixture")) for kind in ToolKind}
        return ToolAvailabilityReport(HostPlatform("demo",BinaryArch.UNKNOWN),values)
    def _runner(self):return ProcessRunner(self.executor,base_env={})


class FixtureAssigner:
    def __init__(self):self.calls=0;self.inference=SyntheticInference()
    def assign(self,path,*,candidate_elements,include_q_peaks=True):
        self.calls+=1;parsed=read_res(path)
        model=read_snapshot(path,reflection_sha256=digest("generated-reflections"),coordinate_bound=5.)
        peaks=tuple(DensityPeak(p.label,(p.x,p.y,p.z),p.height) for p in parsed.q_peaks) if include_q_peaks else ()
        observation=DensityObservation(model.sha256,model.basis_key,model.sha256,"SYNTHETIC_PEAK_LIST",peaks)
        prepared=SyntheticPreparer().prepare(model,observation)
        result=self.inference.predict_elements(prepared.request)
        return RuntimeAssignment(prepared.request,result,sha256_file(path),prepared.request.site_ids,(),
                                 {"actual_pretrained_weights_executed":False,"synthetic":True})


@pytest.fixture
def backend(tmp_path):
    repo=tmp_path.resolve()/"repo";repo.mkdir()
    subprocess.run(["git","init","-q",str(repo)],check=True)
    (repo/".gitignore").write_text("/runs/\n")
    ins,hkl=write_demo_case(repo/"runs/inputs")
    return repo,ins,hkl


def run_backend(backend,*,route=None,policy=None,metal=False,peaks=False,assigner=None,chain=None):
    repo,ins,hkl=backend
    chain=chain or FixtureToolchain(peaks=peaks);assigner=assigner or FixtureAssigner()
    service=ReconstructionBackendService(repo,repo/"runs/backend",toolchain_factory=lambda request:chain,
                                          assigner_factory=lambda request:assigner)
    request=ReconstructionRequest(ins,hkl,route=route or InitialRouteConfig(),cycle=policy or CyclePolicy(),
                                  metal_only=metal,mode=RunMode.MOCK_DEMO)
    result=service.run(request)
    return result,service,chain,assigner


@pytest.mark.parametrize("route",[
    InitialRouteConfig(),InitialRouteConfig(InitialStrategy.SMALL_MOLECULE),
    InitialRouteConfig(InitialStrategy.SMALL_MOLECULE,small_solver="shelxd",find_sites=2,plop=(10,2)),
])
def test_initial_routes_enter_same_pretrained_interface_and_stop_after_two_stable_transitions(backend,route):
    result,service,chain,assigner=run_backend(backend,route=route)
    assert result["phase"]=="completed",result
    assert result["stop_code"]=="movable_sites_r1_and_growth_stable"
    assert result["stable_transitions"]==2 and len(result["history"])==3
    assert assigner.calls==3 and result["tool_calls"]==4
    assert not result["scientifically_validated"] and not result["expert_reference_used"]
    seal=verify_seal(Path(result["run_root"]),private_root=service.runs_root,expected_seal_sha256=result["seal_sha256"])
    assert seal.file_for_role("model").sha256==sha256_file(Path(result["ai_solved_res"]))
    assert service.status(result["run_id"])["phase"]=="completed"
    candidates=json.loads((Path(result["run_root"])/"candidates/index.json").read_text())
    assert not any(item["status"]=="validated" for item in candidates["candidates"])


def test_fixed_outer_rounds_do_not_stop_early_on_stability_and_are_not_shelxl_cycles(backend):
    result,_,chain,assigner=run_backend(backend,policy=CyclePolicy(iterations=5,shelxl_cycles=2))
    assert result["phase"]=="completed",result
    assert result["stop_code"]=="fixed_rounds_completed" and assigner.calls==5
    assert len([p for p in chain.executor.calls if p.tool_kind is ToolKind.SHELXL])==5
    assert all("L.S. 2" in (p.cwd/"trial.ins").read_text() for p in chain.executor.calls if p.tool_kind is ToolKind.SHELXL)


def test_protein_cde_trace_converts_then_enters_same_assignment_refinement_cycle(backend):
    pytest.importorskip("gemmi")
    repo,_,_=backend
    sad=repo/"runs/inputs/synthetic-sad.hkl"
    sad.write_text("\n".join(f"{h:4d}{k:4d}{l:4d}{100.:8.2f}{2.:8.2f}" for h,k,l in
        [(1,0,1),(-1,0,-1),(2,0,1),(-2,0,-1)])+"\n   0   0   0    0.00    0.00\n")
    protein=ProteinPhasingConfig(True,"SAD","P 1 21 1",1,"Zn",.5,
        (PhasingDataset("SAD",sad,friedel_pairs_retained=True),),preferred_hand="original")
    route=InitialRouteConfig(InitialStrategy.PROTEIN_CDE,protein=protein)
    result,_,chain,assigner=run_backend(backend,route=route,policy=CyclePolicy(iterations=1))
    assert result["phase"]=="completed",result
    assert result["initial"]["evidence"]["substructure_is_not_full_seed"]
    assert result["initial"]["seed_path"].endswith("trace-converted.res")
    assert [call.tool_kind for call in chain.executor.calls][:4]==[
        ToolKind.SHELXC,ToolKind.SHELXD,ToolKind.SHELXE,ToolKind.SHELXE]
    assert assigner.calls==1 and result["tool_calls"]==5


def test_metal_only_retains_and_fixes_metal_then_grows_observed_peak_sites(backend):
    result,_,chain,_=run_backend(backend,metal=True,peaks=True)
    assert result["phase"]=="completed",result
    root=Path(result["run_root"])
    seed=checked_res(root/"metal-only-seed.res")
    assert [a.label for a in seed.atoms]==["Zn1"]
    first=next(p for p in chain.executor.calls if p.tool_kind is ToolKind.SHELXL)
    assert "10.25000000" in (first.cwd/"trial.ins").read_text()
    assert "10.25000000" in (first.cwd/"trial.res").read_text()
    final=checked_res(Path(result["ai_solved_res"]))
    assert final.atoms[0].x==.25 and any(a.label.startswith("X") for a in final.atoms)


@pytest.mark.parametrize("rounding,completed",[(0.000001,True),(0.0001,False)])
def test_fixed_anchor_print_rounding_is_bounded_and_does_not_accumulate(backend,rounding,completed):
    chain=FixtureToolchain();execute=chain.executor.execute
    def rounded(plan,env,**kwargs):
        result=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            path=plan.cwd/"trial.res";rows=[]
            for line in path.read_text().splitlines():
                fields=line.split()
                if fields and fields[0]=="Zn1":
                    fields[2]=f"{float(fields[2])+rounding:.6f}";line=" ".join(fields)
                rows.append(line)
            path.write_text("\n".join(rows)+"\n")
        return result
    chain.executor.execute=rounded
    # This checks raw fixed-coordinate encoding, not numerical convergence of a
    # fixed-only scaffold (which is deliberately no longer stop-eligible).
    result,_,_,_=run_backend(backend,metal=True,chain=chain,policy=CyclePolicy(iterations=3))
    if completed:
        assert result["phase"]=="completed",result
        assert checked_res(Path(result["ai_solved_res"])).atoms[0].x==.25
    else:
        assert result["stop_code"]=="reconstruction_blocked"
        assert result["stop_detail"]=="fixed_metal_coordinate_drift_or_encoding_changed"
        assert result["ai_solved_res"] is None


def test_blocked_real_prerequisite_never_reads_expert_or_infers(backend,monkeypatch):
    repo,ins,hkl=backend
    chain=FixtureToolchain(missing=(ToolKind.SHELXT,));chain.mode=RunMode.REAL
    def forbidden(_request):pytest.fail("blocked tools reached inference construction")
    service=ReconstructionBackendService(repo,repo/"runs/blocked",toolchain_factory=lambda request:chain,assigner_factory=forbidden)
    result=service.run(ReconstructionRequest(ins,hkl))
    assert result["phase"]=="blocked" and result["ai_solved_res"] is None
    assert not chain.executor.calls


def test_request_has_no_expert_or_training_input_and_roundtrips(backend):
    _,ins,hkl=backend;request=ReconstructionRequest(ins,hkl)
    assert ReconstructionRequest.from_record(request.to_record()).to_record()==request.to_record()
    for key in ("expert_path","reference_res","training","answer_path"):
        with pytest.raises(ValueError):ReconstructionRequest.from_record({**request.to_record(),key:"held-out"})
    restored=ReconstructionRequest.from_record({**request.to_record(),"checkpoint":"local-weights.pth"})
    assert isinstance(restored.checkpoint,Path)


def test_initial_untyped_maxima_are_placeholders_before_actual_assignment(tmp_path):
    seed=tmp_path/"seed.res";seed.write_text(synthetic_seed_text().replace("Zn1 5 ","Q1 5 ").replace("C1 1 ","Q2 1 "))
    output=tmp_path/"promoted.res";report=promote_initial_peaks(seed,output)
    model=checked_res(output)
    assert len(model.atoms)==2 and len(report["promoted_peaks"])==2
    assert not report["placeholder_is_an_element_assignment"]


def test_real_shelxt_eighth_density_column_is_preserved_as_metadata_not_an_adp(tmp_path):
    source=tmp_path/"automatic.res"
    lines=[]
    for line in synthetic_seed_text().splitlines():
        if line.split() and line.split()[0] in {"Zn1","C1"}:line+=" 25.25"
        lines.append(line)
    source.write_text("\n".join(lines)+"\n")
    before=source.read_bytes();destination=tmp_path/"normalized.res"
    report=promote_initial_peaks(source,destination,source_program="shelxt")
    assert len(checked_res(destination).atoms)==2
    assert report["shelxt_typed_peak_heights"]["Zn1"]["height"]==25.25
    assert source.read_bytes()==before


def test_skew_cell_nearest_image_is_metric_based():
    cell=(10,10,10,90,90,20)
    distance=periodic_distance(cell,(0,0,0),(.49,.49,0))
    assert distance<2


def test_nonhydrogen_head_cannot_relabel_an_existing_hydrogen(tmp_path):
    source=tmp_path/"seed.res"
    source.write_text(synthetic_seed_text().replace("HKLF 4","H1 2 0.1 0.2 0.3 11 0.03\nHKLF 4"))
    model=checked_res(source)
    predictions=tuple(ElementPrediction(a.label,(ClassProbability(6,.8),ClassProbability(7,.2))) for a in model.atoms)
    output=tmp_path/"assigned.res"
    assign_to_model(model,predictions,output,candidate_elements=model.header.sfac)
    returned=checked_res(output)
    assert returned.element_of(next(a for a in returned.atoms if a.label=="H1"))=="H"


def test_broad_element_hypotheses_do_not_add_unused_sfac_species(tmp_path):
    from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
    source=tmp_path/'seed.res';source.write_text(synthetic_seed_text());model=checked_res(source)
    predictions=tuple(ElementPrediction(a.label,(ClassProbability(6,.8),ClassProbability(7,.2))) for a in model.atoms)
    elements=tuple(e for e,z in ATOMIC_NUMBERS.items() if 1<z<98)
    out=tmp_path/'assigned.res';report=assign_to_model(model,predictions,out,candidate_elements=elements)
    assert checked_res(out).header.sfac==model.header.sfac
    assert not report['new_sfac_types']


def test_reflection_basis_compares_semantics_and_keeps_twin_parameter_count(tmp_path):
    first=tmp_path/"first.res";second=tmp_path/"second.res"
    first.write_text(synthetic_seed_text().replace("HKLF 4","BASF 0.5\nHKLF 4"))
    second.write_text(first.read_text().replace("BASF 0.5","BASF 0.45").replace("CELL 0.71073","CELL 0.710730"))
    assert setting_key(checked_res(first),"a"*64)==setting_key(checked_res(second),"a"*64)
    second.write_text(first.read_text().replace("HKLF 4","OMIT 1 2 3\nHKLF 4"))
    assert setting_key(checked_res(first),"a"*64)!=setting_key(checked_res(second),"a"*64)


def test_continued_initial_search_controls_and_inline_comments_are_not_atoms(backend):
    repo,_,_=backend;source=repo/"runs/control-seed.res"
    source.write_text(synthetic_seed_text().replace("SFAC C H N O Zn","SFAC C H N O Zn ! inline note =")
        .replace("HKLF 4","PLOP 1 2 =\n 3 4 5 6\nHKLF 4"))
    parsed=checked_res(source)
    assert len(parsed.atoms)==2 and parsed.header.sfac==("C","H","N","O","Zn")
    from crystalx_core.refinement.cycle import refine_file
    from crystalx_core.tools.runner import CancelToken
    chain=FixtureToolchain()
    result=refine_file(model_path=source,hkl_path=backend[2],directory=repo/"runs/refine-control",
        adapter=chain.adapter(ToolKind.SHELXL),cycles=2,timeout_s=5,cancel=CancelToken())
    ins=(result.model_path.parent/"trial.ins").read_text()
    assert "PLOP" not in ins and " 3 4 5 6" not in ins


def test_multithreaded_tools_get_cpu_budget_for_all_logical_workers(backend,monkeypatch):
    import os
    monkeypatch.setattr(os,"cpu_count",lambda:4)
    result,_,chain,_=run_backend(backend,policy=CyclePolicy(iterations=1))
    assert result["phase"]=="completed",result
    for plan in chain.executor.calls:
        assert plan.limits.cpu_seconds>=4*plan.timeout_s


def test_generated_initial_reuse_pins_source_and_does_not_repeat_solver(backend):
    first,service,chain,_=run_backend(backend,policy=CyclePolicy(iterations=1))
    source=Path(first["run_root"]);initial_sha=sha256_file(source/"initial-result.json")
    _,ins,hkl=backend
    request=ReconstructionRequest(ins,hkl,mode=RunMode.MOCK_DEMO,cycle=CyclePolicy(iterations=1),label="child")
    before=sha256_file(source/"initial-result.json");count=len(chain.executor.calls)
    child=service.run_from_generated_initial(request,source_run_root=source,expected_initial_result_sha256=initial_sha)
    assert child["phase"]=="completed",child
    assert len(chain.executor.calls)==count+1 and chain.executor.calls[-1].tool_kind is ToolKind.SHELXL
    assert child["initial_reuse"]["new_initial_tool_invocations"]==0 and child["tool_calls"]==2
    assert sha256_file(source/"initial-result.json")==before
    with pytest.raises(Exception):
        service.run_from_generated_initial(request,source_run_root=source,expected_initial_result_sha256="0"*64)


def test_large_isotropic_model_raises_documented_matrix_capacity_before_vendor_limit(backend):
    from crystalx_core.refinement.cycle import refine_file
    from crystalx_core.tools.runner import CancelToken
    repo,_,hkl=backend
    model=repo/"runs/large-generated.res"
    header="\n".join(line for line in synthetic_seed_text().splitlines() if not line.startswith(("Zn1 ","C1 ","HKLF ","END")))
    atoms="\n".join(f"C{i:03d} 1 0.1 0.2 0.3 11 0.05" for i in range(760))
    model.write_text(header+"\n"+atoms+"\nHKLF 4\nEND\n")
    chain=FixtureToolchain()
    refine_file(model_path=model,hkl_path=hkl,directory=repo/"runs/large-refinement",
                adapter=chain.adapter(ToolKind.SHELXL),cycles=2,timeout_s=5,cancel=CancelToken())
    assert "-b3500" in chain.executor.calls[0].argv


def test_async_frontend_submit_status_and_cancel_share_durable_root(backend):
    repo,ins,hkl=backend;chain=FixtureToolchain();assigner=FixtureAssigner()
    service=ReconstructionBackendService(repo,repo/"runs/jobs",toolchain_factory=lambda _:chain,assigner_factory=lambda _:assigner)
    handle=service.submit(ReconstructionRequest(ins,hkl,mode=RunMode.MOCK_DEMO,cycle=CyclePolicy(iterations=1)))
    deadline=time.monotonic()+10
    while service.status(handle.run_id)["phase"] in ("queued","running") and time.monotonic()<deadline:time.sleep(.01)
    assert service.status(handle.run_id)["phase"]=="completed"
    assert service.summary(handle.run_id)["ai_solved_res"]
    assert not service.cancel(handle.run_id)


def test_terminal_state_is_never_published_before_its_summary(backend,monkeypatch):
    from crystalx_core.workspace.store import RunWorkspace
    write=RunWorkspace.write_json;observed=[]
    def inspect(workspace,relative,value,**kwargs):
        if relative=="state.json" and value["phase"] not in ("queued","running"):
            summary=json.loads((workspace.layout.root/"summary.json").read_text())
            assert summary["phase"]==value["phase"]
            assert summary["ai_solved_res"]
            observed.append(value["phase"])
        return write(workspace,relative,value,**kwargs)
    monkeypatch.setattr(RunWorkspace,"write_json",inspect)
    result,_,_,_=run_backend(backend,policy=CyclePolicy(iterations=1))
    assert result["phase"]=="completed" and observed==["completed"]


def test_fixed_only_metal_scaffold_cannot_claim_numerical_stability(backend):
    result,_,_,_=run_backend(backend,metal=True,policy=CyclePolicy(max_rounds=5))
    assert result['stop_code']=='round_safety_budget_exhausted'
    assert len(result['history'])==5 and result['stable_transitions']==0
    assert all(not row['monitored_movable_site_ids'] for row in result['history'])


def test_metal_stop_monitors_moving_ligand_and_requires_no_new_sites(backend):
    chain=FixtureToolchain(peaks=True);execute=chain.executor.execute
    def moving(plan,env,**kwargs):
        result=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXL:
            path=plan.cwd/'trial.res';rows=[]
            for line in path.read_text().splitlines():
                fields=line.split()
                if fields and fields[0]=='X001':fields[2]=str(float(fields[2])+.02);line=' '.join(fields)
                rows.append(line)
            path.write_text('\n'.join(rows)+'\n')
        return result
    chain.executor.execute=moving
    result,_,_,_=run_backend(backend,metal=True,chain=chain,policy=CyclePolicy(max_rounds=6))
    assert result['stop_code']=='round_safety_budget_exhausted'
    assert result['history'][-1]['monitored_movable_site_ids']==['X001']
    assert result['history'][-1]['important_coordinate_max_change_angstrom']>.02
    assert result['stable_transitions']==0


def test_growth_breaks_two_transition_stop_until_two_quiet_rounds(backend):
    result,_,_,_=run_backend(backend,metal=True,peaks=True)
    assert result['stop_code']=='movable_sites_r1_and_growth_stable'
    assert len(result['history'])==5
    assert result['history'][1]['accepted_new_sites_if_retained']==1
    assert not result['history'][2]['stable_transition']


def test_pruning_commit_failure_restores_prior_model_and_policy_state(backend,monkeypatch):
    pytest.importorskip("gemmi")
    pytest.importorskip("scipy.spatial")
    from crystalx_core.refinement.adaptive import AdaptivePolicy
    from crystalx_core.workspace.store import RunWorkspace
    chain=FixtureToolchain();execute=chain.executor.execute
    def generated(plan,env,**kwargs):
        result=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXT:
            p=plan.cwd/(plan.argv[1]+'_a.res')
            text=synthetic_seed_text();text='\n'.join(row for row in text.splitlines() if not row.startswith(('Zn1 ','C1 ')))
            atoms='\n'.join(f'C{i:03d} 1 {i*.01:.3f} .2 .3 11 {0.8 if i==0 else .05}' for i in range(20))
            p.write_text(text.replace('HKLF 4',atoms+'\nHKLF 4')+'\n')
        return result
    chain.executor.execute=generated
    write=RunWorkspace.write_json
    def fault(workspace,relative,value,**kwargs):
        if str(relative)=='cycles/round-002/commit.json':raise OSError('synthetic pruning commit fault')
        return write(workspace,relative,value,**kwargs)
    monkeypatch.setattr(RunWorkspace,'write_json',fault)
    adaptive=AdaptivePolicy(pruning_enabled=True,prune_close=False,prune_isolated=False)
    result,_,_,_=run_backend(backend,chain=chain,policy=CyclePolicy(iterations=3,adaptive=adaptive))
    assert result['phase']=='failed'
    final=checked_res(Path(result['ai_solved_res']))
    assert len(final.atoms)==20 and any(a.label=='C000' for a in final.atoms)
    root=Path(result['run_root'])
    assert json.loads((root/'cycles/round-002/deleted-sites.json').read_text())['removed_sites'][0]['site_id']=='C000'
    assert result['history'][-1]['decision']=='REJECTED_BY_CONTROLLER'


def test_original_ins_is_default_candidate_source_even_when_seed_omits_type(backend):
    chain=FixtureToolchain();execute=chain.executor.execute
    def omit(plan,env,**kwargs):
        result=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXT:
            p=plan.cwd/(plan.argv[1]+'_a.res');text=p.read_text()
            text=text.replace('SFAC C H N O Zn','SFAC C H O Zn').replace('Zn1 5 ','Zn1 4 ')
            text=text.replace('UNIT 18 20 4 2 1','UNIT 18 20 2 1')
            p.write_text(text)
        return result
    chain.executor.execute=omit
    result,_,_,_=run_backend(backend,chain=chain,policy=CyclePolicy(iterations=1))
    source=json.loads((Path(result['run_root'])/'candidate-element-source.json').read_text())
    assert source['effective_source']=='original_ins' and 'N' in source['elements']
    assert 'N' not in source['initial_seed_sfac']


def test_adaptive_request_roundtrip_records_protocol_and_all_defaults(backend):
    from crystalx_core.refinement.adaptive import AdaptivePolicy
    _,ins,hkl=backend
    request=ReconstructionRequest(ins,hkl,cycle=CyclePolicy(inference_protocol='initial_labels',adaptive=AdaptivePolicy(hysteresis_enabled=True)))
    record=request.to_record();restored=ReconstructionRequest.from_record(record)
    assert restored.to_record()==record
    assert record['candidate_element_source']=='original_ins'
    assert record['cycle']['adaptive']['confirmations']==2
    assert record['cycle']['adaptive']['max_change_fraction']==.05
    assert record['cycle']['stopping_policy']=='movable_and_growth'


def test_initial_metal_reset_reserves_deleted_seed_ids_against_q_reuse(backend):
    chain=FixtureToolchain(peaks=True);execute=chain.executor.execute
    def rename(plan,env,**kwargs):
        result=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXT:
            p=plan.cwd/(plan.argv[1]+'_a.res');p.write_text(p.read_text().replace('C1 1 ','X001 1 '))
        return result
    chain.executor.execute=rename
    result,_,_,_=run_backend(backend,chain=chain,metal=True,policy=CyclePolicy(iterations=3))
    final=checked_res(Path(result['ai_solved_res']))
    assert 'X001' not in {a.label for a in final.atoms}
    assert 'X002' in {a.label for a in final.atoms}


def test_staged_original_modified_by_assigner_factory_is_not_re_pinned_as_truth(backend):
    repo,ins,hkl=backend;chain=FixtureToolchain()
    def factory(request):
        for p in (repo/'runs/guard').glob('*/inputs/'+ins.name):
            p.chmod(0o644);p.write_text(p.read_text().replace('UNIT ','REM changed\nUNIT '))
        return FixtureAssigner()
    service=ReconstructionBackendService(repo,repo/'runs/guard',toolchain_factory=lambda _:chain,assigner_factory=factory)
    result=service.run(ReconstructionRequest(ins,hkl,mode=RunMode.MOCK_DEMO,cycle=CyclePolicy(iterations=1)))
    assert result['stop_code']=='reconstruction_blocked'
    assert result['stop_detail']=='original_instruction_staged_hash_differs_from_upload'
    assert result['ai_solved_res'] is None
    assert not any(plan.tool_kind is ToolKind.SHELXL for plan in chain.executor.calls)


def test_upstream_coverage_choice_reaches_initial_metal_warmup_without_rewriting_scores(backend):
    chain=FixtureToolchain();execute=chain.executor.execute
    def untyped(plan,env,**kwargs):
        result=execute(plan,env,**kwargs)
        if plan.tool_kind is ToolKind.SHELXT:
            p=plan.cwd/(plan.argv[1]+'_a.res');p.write_text(p.read_text().replace('Zn1 5 ','Zn1 1 '))
        return result
    chain.executor.execute=untyped
    class CoverageAssigner(FixtureAssigner):
        def assign(self,path,*,candidate_elements,include_q_peaks=True,**kwargs):
            result=super().assign(path,candidate_elements=candidate_elements,include_q_peaks=include_q_peaks)
            predictions=tuple(ElementPrediction(key,(ClassProbability(6,.9),ClassProbability(7,.1))) for key in result.site_ids)
            return replace(result,result=replace(result.result,predictions=predictions),provenance={
                **result.provenance,'input_protocol':kwargs.get('input_protocol','current_labels'),
                'protocol_element_choices':{key:'Zn' if key=='Zn1' else 'C' for key in result.site_ids}})
    result,_,_,_=run_backend(backend,chain=chain,assigner=CoverageAssigner(),metal=True,
        policy=CyclePolicy(iterations=1,inference_protocol='upstream_ins',max_new_peaks_per_round=0))
    assert result['phase']=='completed',result
    root=Path(result['run_root']);warm=json.loads((root/'metal-seed-inference.json').read_text())
    assert warm['result']['predictions'][0]['topk'][0]['atomic_number']==6
    assert checked_res(root/'metal-only-seed.res').element_of(checked_res(root/'metal-only-seed.res').atoms[0])=='Zn'
