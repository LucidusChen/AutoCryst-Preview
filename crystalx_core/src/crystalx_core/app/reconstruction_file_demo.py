"""Run the iterative software demonstration with real controlled child processes and files."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import uuid

from crystalx_core.app.demo import write_demo_case
from crystalx_core.app.reconstruction import ReconstructionProviders, ReconstructionService
from crystalx_core.app.reconstruction_demo import (
    SYNTHETIC_AUDIT, SyntheticInference, SyntheticPreparer, synthetic_seed_text,
)
from crystalx_core.app.reconstruction_files import FileEvidenceStore, QPeakDensity, ShelxlFileRefinement
from crystalx_core.private_acceptance.harness import private_directory
from crystalx_core.workflow.reconstruction import Assessment, ReconstructionPolicy, digest


def synthetic_file_assessment(model, listing, evidence):
    """Scripted software checks only. No chemical/scientific evaluator is implemented here."""
    bad = any(s.site_id.startswith("X") and s.element == "O" for s in model.sites)
    checks = tuple((name, "FAIL" if bad and name == "geometry" else "PASS") for name in (
        "geometry", "coordination", "occupancy_adp", "chirality_symmetry", "restraints",
        "heavy_site_drift", "residual_distribution"))
    return Assessment(model.basis_key, listing.r1_observed, listing.wr2, listing.goof,
        (listing.r1_observed * 10, listing.r1_observed * 3), listing.n_all,
        listing.parameters_refined, checks, evidence.source_sha256,
        least_squares_converged=True, model_sha256=model.sha256)


class FileDemoInference(SyntheticInference):
    def predict_elements(self, request, **options):
        # Reuse the demonstration's scripted probabilities with a physical Q-peak identity.
        mapped = replace(request, site_ids=tuple("peak_one" if key == "Q1" else key for key in request.site_ids))
        output = super().predict_elements(mapped, **options)
        return replace(output, input_sha256=digest(request), predictions=tuple(
            replace(prediction, site_id=key) for prediction, key in zip(output.predictions, request.site_ids)))


def file_demo_providers(*, evaluated: bool = True, runner=None):
    evidence = FileEvidenceStore()
    return ReconstructionProviders(QPeakDensity(evidence), FileDemoInference(), SyntheticPreparer(),
        SYNTHETIC_AUDIT, ShelxlFileRefinement(evidence,
            synthetic_assessor=synthetic_file_assessment if evaluated else None, runner=runner))


def run_file_demo(repository: Path, output_root: Path) -> dict:
    root = private_directory(repository, output_root)
    inputs = root / ("synthetic-file-inputs-" + uuid.uuid4().hex)
    inputs.mkdir(mode=0o700)
    ins, hkl = write_demo_case(inputs)
    seed = inputs / "automatic-seed.res"
    seed.write_text(synthetic_seed_text())
    service = ReconstructionService(repository, root)
    result = service.run(ins_path=ins, hkl_path=hkl, initial_model=seed,
        providers=file_demo_providers(), policy=ReconstructionPolicy(), synthetic=True)
    return {"synthetic": True, "run_root": result["run_root"],
            "decisions": [item["decision"] for item in result["decisions"]],
            "stop_code": result["stop_code"], "status": result["status"],
            "controlled_subprocesses": result["usage"].tool_calls,
            "real_crystallographic_invocations": 0, "scientifically_reviewed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_file_demo(args.repository,
        args.output_root or args.repository / "runs/iterative_file_demo"), indent=2))


if __name__ == "__main__":
    main()
