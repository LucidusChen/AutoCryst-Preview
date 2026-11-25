# Experimental CrystalX Apple Silicon Backend

Native PyTorch CPU/MPS inference plumbing, **not MetalFX**, a native GUI, a
new crystallographic solver, or an accepted structure-solution workflow.

## What runs

- The unchanged, locally supplied CrystalX inference model definitions are loaded
  into isolated namespaces after all four audited source hashes are checked.
  No training model is merged, upstream file edited, or global import monkeypatch used.
- The two message-passing patterns use native torch sum aggregation, without
  `torch_geometric`, `torch_scatter`, `torch_cluster`, RDKit or cctbx dependencies.
- A bounded, chunked radius graph uses a named nearest-distance/index-tie policy.
  It records neighbor truncation. **This policy is not equivalent to upstream's
  implementation-defined truncated graph.** CPU/MPS agreement on this new graph
  is not agreement with the upstream CPU/CUDA baseline.
- Local float32 tensor state_dict checkpoints are loaded with `weights_only=True`
  and strict parameter matching. Hidden width, classes, layers and RBF count are
  read from tensors, not hard-coded CLI class counts.
- A two-pass heavy-element prediction retains a runner-up, model/source digests,
  actual device, explicit fallback and uncalibrated-probability warnings.

Only the audited inference defaults are supported: 8 attention heads, 5 Angstrom
cutoff, 32 neighbors, exponential-normal RBF and SiLU, with no vector whitening.
Tensor shapes cannot establish these training settings. Alternative architectures
need a separately reviewed configuration contract, not a silent best guess.

## Install in the existing development environment

From the repository root, after the core development bootstrap:

```sh
.venv/bin/python -m pip install --no-build-isolation --no-deps -e crystalx_engine_apple_silicon
sh scripts/crystalx.sh check
sh scripts/crystalx.sh apple doctor
```

`--no-deps` is suitable only when the local core and compatible PyTorch are already
installed. A normal package installation declares `crystalx-core` and `torch`; it
never bundles or downloads the upstream source or model weights itself.

## Synthetic hardware smoke test

```sh
sh scripts/crystalx.sh apple self-test \
  --source-root /path/to/licensed-upstream/inference_repo --device cpu
env -u PYTORCH_ENABLE_MPS_FALLBACK sh scripts/crystalx.sh apple self-test \
  --source-root /path/to/licensed-upstream/inference_repo --device mps --no-cpu-fallback
```

This creates temporary **random synthetic weights**, compares native CPU with the
selected backend, reports per-site argmax agreement and probability differences,
and removes the temporary checkpoint. It does not use trained CrystalX weights,
solve a structure, or establish scientific accuracy. MPS may be unavailable inside
a sandbox even when normal host execution can access it. Strict MPS requests fail
honestly instead of being reported as GPU success. Do not enable
`PYTORCH_ENABLE_MPS_FALLBACK=1` when attesting pure MPS execution.

## Local checkpoint interface

```sh
sh scripts/crystalx.sh apple predict \
  --source-root /path/to/licensed-upstream/inference_repo \
  --checkpoint /private/path/to/crystalx-heavy.pth \
  --input /private/path/to/sites.json \
  --output /private/path/to/new-predictions.json --device auto
```

Optionally pin a separately recorded digest using `--weights-sha256`. Output files
are created exclusively, never overwritten. Example input (hand-authored, not a
real case or ground-truth structure):

```json
{
  "site_ids": ["peak-1", "peak-2", "context-1"],
  "atomic_numbers": [6, 8, 7],
  "positions_angstrom": [[0, 0, 0], [1.4, 0, 0], [0, 1.5, 0]],
  "candidate_atomic_numbers": [6, 7, 8],
  "main_site_count": 2
}
```

Positions are already Cartesian Angstrom coordinates. Sites are one graph;
symmetry context is explicit and retained during both passes, while only the
first `main_site_count` sites are returned. Float32-coincident positions fail
instead of silently generating NaNs or deleting sites. The caller owns unit-cell
conversion, symmetry expansion, peak ordering and input descriptors. Candidate
probabilities are conditional softmax scores, not calibrated confidences. No
forced SFAC coverage or element replacement is performed. At least two candidate
elements are required so that every prediction has a meaningful runner-up.

## Acceptance boundary and next steps

The backend is **experimental and not wired into automatic scaffold locks or
scientific candidate promotion**. The core contract contains no torch import.
The upstream CPU/CUDA reference is a separately supplied local input.
`crystalx_engine/` is retained in this private snapshot at the owner's request; no license file was found. See NOTICE. It is not approved for public redistribution.
The caller must provide the authorized inference source directory explicitly; the
example paths in this document are placeholders, explicit caller-supplied paths. No upstream source
or weight download happens automatically. Missing or mismatched source is a
reported blocker. No upstream redistribution license is supplied by this snapshot.

Before production use: obtain an authorised local trained checkpoint, establish
an upstream CPU reference with its actual graph, measure graph/prediction drift
on fixed non-secret and authorised local cases (especially capped dense metal
environments), and obtain scientific acceptance of tolerances. The preprocessing
and measurement boundaries below are implemented; they do not establish model
accuracy or automatically connect predictions to the application service.
No missing step is replaced by a mock, expected element label or answer key.

Run focused tests from the repository root:

```sh
sh scripts/crystalx.sh test apple
```

The upstream source and weights are separate user-supplied inputs. This package
does not grant or change their redistribution rights. Do not bundle source,
weights, proprietary tools, raw cases or run results into a public distribution.

## Explicit preprocessing boundary

`preprocessing.prepare_initial_solution` consumes an already safely parsed core
`ResFile`. It reads no files and imports neither torch nor a geometry provider.
Supply `CompleteSymmetry` with exact fractional operators and a provenance source;
the supplied set must equal the full declared LATT/SYMM group, including inversion
and centring. Missing computation never falls back to P1.

Main sites preserve parsed atom order, then all Q peaks in parsed order. Atom
descriptors follow SFAC; every Q peak needs an explicit caller hypothesis descriptor.
Nothing adjusts SFAC, trims peaks or forces element coverage. Site IDs, source
indices, occupancy/displacement data and peak heights are retained as provenance.
These are not calibrated model features or evidence of assigned elements.

The required context radius is an explicit policy. Periodic context is complete
within that radius using reciprocal-cell bounds, with original sites first and
context masked from output. Model receptive-field coverage and upstream
preprocessing parity remain unverified. Resource caps block instead of truncating.
Malformed/skipped input rows, unsupported fractional coordinate encodings, decimal
SYMM, nonzero PART and coincident distinct sites also block for explicit review.

Optional CLI preparation uses a JSON symmetry config such as this *only when the
initial solution explicitly declares P1*:

```json
{"source": "reviewed initial-solution P1 operators", "operators": ["x,y,z"]}
```

```sh
sh scripts/crystalx.sh apple prepare --input /private/path/initial-solution.res \
  --symmetry-config /private/path/symmetry.json --context-radius 5 \
  --candidates 6 7 8 26 --output /private/path/new-preparation.json
```

Choose the radius deliberately: the example 5 A is a single-edge cutoff, not a
promise of multilayer receptive-field completeness. If peaks exist supply one
`--q-descriptors` integer for each peak in parsed order. The new JSON contains a
neutral `request`, `main_mask` and separate preparation provenance. It is labelled
`EXPERIMENTAL_PREPARED` and does not perform inference. Existing output files are
never overwritten.

Claude's opt-in application boundary can pass `prepared.request` to
`AppleSiliconBackend.predict_elements`, or construct the pure core request from
the preparation JSON's `request` member. Keep preparation provenance beside the
result and map predictions by site ID. Do not pass the entire preparation envelope
as the existing Cartesian prediction CLI input. The current Web/solver path is
unchanged; no automatic anchor locks or scientific gate promotion are authorised.

## Actual upstream graph measurement framework

```sh
sh scripts/crystalx.sh apple parity \
  --source-root /path/to/licensed-upstream/inference_repo \
  --checkpoint /private/path/authorised-heavy.pth \
  --weights-sha256 YOUR_INDEPENDENT_SHA256 \
  --provenance 'Owner-supplied local checkpoint and origin' \
  --output /private/path/new-parity-measurement.json
```

No download occurs. Missing checkpoint/digest or actual upstream dependencies
produces `BLOCKED` (CLI exit 3). The default actual upstream CPU worker requires
PyG, torch-scatter, torch-cluster and ASE. An optional `--reference-python` must
already have those and the adapter/core installed. No reference predictions or
native graph relabelled as upstream are accepted. Workers are isolated processes;
six unchanged upstream source bodies (including the original two-pass helper) are
hash-pinned before execution. Source, weight, input, device and run identity are
checked. Caller provenance is an attestation, not publisher authentication.

Fixed hand-authored sparse and dense 41-site Fe cases record full conditional
probabilities, top element, runner-up, per-site changes, both-pass edge changes
and neighbor truncation. They exercise the machinery, not representative scientific
coverage. `MEASURED` means measurements completed, never a tolerance pass. All
scientific/calibration/upstream-parity approval flags remain false and the
scientific tolerance stays unset pending owner review. Synthetic fixture weights
can exercise plumbing but never count as a trained comparison.
