"""Private local client for the multi-route backend and its independent postprocessors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crystalx_core._serde import to_jsonable
from crystalx_core.app.reconstruction_backend import ReconstructionBackendService, ReconstructionRequest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository",type=Path,default=Path.cwd())
    parser.add_argument("--runs-root",type=Path,default=None)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("preview","run"):
        command=sub.add_parser(name);command.add_argument("--request",type=Path,required=True)
    for name in ("checkcif","compare"):
        command=sub.add_parser(name);command.add_argument("--run-root",type=Path,required=True)
        command.add_argument("--seal-sha256",required=True)
        command.add_argument("--output-dir",type=Path,required=True)
    checker=sub.choices["checkcif"]
    checker.add_argument("--execute",action="store_true")
    checker.add_argument("--allow-builtin-checkdef",action="store_true")
    comparison=sub.choices["compare"]
    comparison.add_argument("--expert",type=Path,required=True)
    comparison.add_argument("--expert-root",type=Path,required=True)
    comparison.add_argument("--distance-tolerance",type=float,default=0.5)
    comparison.add_argument("--metal",action="append",default=[])
    args=parser.parse_args(argv)
    repository=args.repository.resolve(strict=True)
    runs=args.runs_root or repository/"runs/reconstruction"
    service=ReconstructionBackendService(repository,runs)
    if args.command in ("preview","run"):
        request=ReconstructionRequest.from_record(json.loads(args.request.read_text()))
        result=service.preview(request) if args.command=="preview" else service.run(request)
    elif args.command=="checkcif":
        from crystalx_core.app.checkcif import run_local_checkcif
        from crystalx_core.app.toolchain import RealToolchain
        result=run_local_checkcif(args.run_root,private_root=service.runs_root,
            expected_seal_sha256=args.seal_sha256,output_dir=args.output_dir,
            toolchain=RealToolchain(toolbox_dir=repository/"toolbox",allow_translation=False),
            execute=args.execute,allow_builtin_checkdef=args.allow_builtin_checkdef).to_record()
    else:
        from crystalx_core.app.model_comparison import ComparisonPolicy,compare_sealed_res
        policy=ComparisonPolicy(distance_tolerance_angstrom=args.distance_tolerance,metal_elements=tuple(args.metal))
        result=compare_sealed_res(args.run_root,private_root=service.runs_root,
            expected_seal_sha256=args.seal_sha256,expert_path=args.expert,expert_root=args.expert_root,
            output_dir=args.output_dir,policy=policy).to_record()
    print(json.dumps(to_jsonable(result),indent=2,allow_nan=False))
    blocked=bool(result.get("blockers")) or result.get("phase") in {"failed","blocked","cancelled"}
    return 3 if blocked else 0


if __name__=="__main__":
    raise SystemExit(main())
