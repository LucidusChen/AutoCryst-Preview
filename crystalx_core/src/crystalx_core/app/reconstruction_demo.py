"""Replayable synthetic multi-round reconstruction; no measured data or trained weights.

Run from the checkout with ``python -m crystalx_core.app.reconstruction_demo``. The fixed
providers deliberately exercise one good element change, one tempting but invalid addition,
an alternative accepted addition, honest no-progress, and restart into a fresh child run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from crystalx_core.app.demo import DEMO_CELL, demo_atoms, write_demo_case
from crystalx_core.app.reconstruction import ReconstructionProviders, ReconstructionService
from crystalx_core.domain.chemistry import ATOMIC_NUMBERS
from crystalx_core.domain.inference import (
    ClassProbability, ElementInferenceRequest, ElementInferenceResult, ElementPrediction,
    InferenceDevice, ModelIdentity,
)
from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix
from crystalx_core.workflow.reconstruction import (
    Assessment, DensityObservation, DensityPeak, InferenceAudit, ModelSnapshot,
    PreparedInference, ReconstructionPolicy, RefinementOutcome, RefinementTrial, digest,
)


SYNTHETIC_AUDIT = InferenceAudit("synthetic", digest("synthetic-weights-v1"),
    (("synthetic-inference-code", digest("synthetic-inference-code-v1")),),
    digest("synthetic-cartesian-preprocessing-v1"))


def synthetic_seed_text() -> str:
    atoms = demo_atoms()
    metal, ligand = atoms[0], atoms[2]
    row = lambda label, index, xyz: f"{label} {index} " + " ".join(f"{v:.8f}" for v in xyz) + " 11.00000 0.03000"
    return ("TITL CRYSTALX-SYNTHETIC-DEMO iterative seed\nCELL 0.71073 "
            + " ".join(str(v) for v in DEMO_CELL)
            + "\nLATT -1\nSYMM -X,Y+1/2,-Z\nSFAC C H N O Zn\nUNIT 10 28 4 12 2\nFVAR 1\n"
            + row("Zn1", 5, metal.frac) + "\n" + row("C1", 1, ligand.frac)
            + "\nHKLF 4\nEND\n")


class SyntheticDensity:
    def __init__(self, *, restart: bool = False):
        self.calls = 0
        self.restart = restart

    def observe(self, model):
        self.calls += 1
        if self.restart and self.calls == 1:
            peaks = (DensityPeak("peak_restart", (0.65, 0.55, 0.45), 1.5),)
        elif not self.restart and self.calls in (2, 3):
            peaks = (DensityPeak("peak_one", (0.45, 0.5, 0.35), 2.0),)
        else:
            peaks = ()
        return DensityObservation(model.sha256, model.basis_key,
                                  digest(("synthetic-partial-density", model.sha256, peaks)),
                                  "SYNTHETIC_PEAK_LIST", peaks)


class SyntheticPreparer:
    def prepare(self, model: ModelSnapshot, observation: DensityObservation):
        cell = next(line.split()[2:8] for line in model.text.splitlines() if line.startswith("CELL "))
        matrix = orthogonalisation_matrix(tuple(float(v) for v in cell))
        ids = tuple(s.site_id for s in model.sites) + tuple(p.peak_id for p in observation.peaks)
        zs = tuple(ATOMIC_NUMBERS[s.element] for s in model.sites) + (6,) * len(observation.peaks)
        positions = tuple(frac_to_cart(matrix, s.fractional) for s in model.sites)
        positions += tuple(frac_to_cart(matrix, p.fractional) for p in observation.peaks)
        request = ElementInferenceRequest(ids, zs, positions, (1, 6, 7, 8, 30), len(ids))
        return PreparedInference(request, model.sha256, SYNTHETIC_AUDIT.preprocessing_policy_sha256)


class SyntheticInference:
    name = "synthetic-controller-test-provider"

    def __init__(self):
        self.calls = 0

    def predict_elements(self, request, **_):
        self.calls += 1
        predictions = []
        for site_id, z in zip(request.site_ids, request.atomic_numbers):
            if site_id == "C1" and z == 6:
                alternatives = ((7, .8), (6, .2))
            elif site_id == "peak_one":
                alternatives = ((8, .8), (7, .2))
            elif site_id == "peak_restart":
                alternatives = ((6, .8), (7, .2))
            else:
                alternatives = ((z, .8), (7 if z != 7 else 6, .2))
            predictions.append(ElementPrediction(site_id, tuple(ClassProbability(*p) for p in alternatives)))
        return ElementInferenceResult(tuple(predictions),
            ModelIdentity(SYNTHETIC_AUDIT.weights_sha256, SYNTHETIC_AUDIT.source_files, 8, 119, 1, 8),
            InferenceDevice.CPU, InferenceDevice.CPU, False, "", "synthetic, no numerical graph", 0,
            "synthetic-scripted-probabilities", ("Not trained inference or experimental evidence.",),
            0., digest(request), request.candidate_atomic_numbers, "synthetic-no-torch")


class SyntheticRefinement:
    def __init__(self):
        self.calls = 0
        self.trials = []

    def refine(self, request: RefinementTrial):
        self.calls += 1
        self.trials.append(request)
        model = request.proposed_model
        nitrogen = sum(s.element == "N" for s in model.sites)
        bad = any(s.site_id.startswith("X") and s.element == "O" for s in model.sites)
        r1 = .45 - .04 * nitrogen - .03 * (len(model.sites) - 2)
        if bad:
            r1 = .05  # A lower R must still lose to a failing geometry check.
        checks = tuple((name, "FAIL" if bad and name == "geometry" else "PASS") for name in (
            "geometry", "coordination", "occupancy_adp", "chirality_symmetry", "restraints",
            "heavy_site_drift", "residual_distribution"))
        assessment = Assessment(model.basis_key, r1, r1 * 2, 1 + r1 / 10,
            (r1 * 10, r1 * 3), 1000, 4 * len(model.sites), checks,
            digest(("synthetic-assessment", model.sha256, checks)), least_squares_converged=True,
            model_sha256=model.sha256)
        # The provider returns the separately retained lock-free model, never its managed INS.
        return RefinementOutcome(model.text, assessment, "synthetic")


def synthetic_providers(*, restart=False) -> ReconstructionProviders:
    return ReconstructionProviders(SyntheticDensity(restart=restart), SyntheticInference(),
                                   SyntheticPreparer(), SYNTHETIC_AUDIT, SyntheticRefinement())


def run_demo(repository: Path, output_root: Path) -> dict:
    from crystalx_core.private_acceptance.harness import private_directory
    root = private_directory(repository, output_root)
    # Inputs are immutable and reproducible. A second demo invocation uses another directory.
    import uuid
    inputs = root / ("synthetic-inputs-" + uuid.uuid4().hex)
    inputs.mkdir(mode=0o700)
    ins, hkl = write_demo_case(inputs)
    seed = inputs / "automatic-seed.res"
    seed.write_text(synthetic_seed_text())
    policy = ReconstructionPolicy(max_rounds=8, max_inference_calls=8)
    service = ReconstructionService(repository, root)
    first = service.run(ins_path=ins, hkl_path=hkl, initial_model=seed,
                        providers=synthetic_providers(), policy=policy, synthetic=True)
    relative, pin = first["checkpoints"][-1]
    second = service.run(ins_path=ins, hkl_path=hkl, initial_model=None,
                         providers=synthetic_providers(restart=True), policy=policy, synthetic=True,
                         checkpoint=(Path(first["run_root"]) / relative, pin))
    return {"synthetic": True, "first_run": first["run_root"], "restart_run": second["run_root"],
            "first_decisions": [d["decision"] for d in first["decisions"]],
            "restart_decisions": [d["decision"] for d in second["decisions"]],
            "stop_codes": [first["stop_code"], second["stop_code"]],
            "real_invocations": 0, "scientifically_reviewed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = args.output_root or args.repository / "runs/iterative_demo"
    print(json.dumps(run_demo(args.repository, root), indent=2))


if __name__ == "__main__":
    main()
