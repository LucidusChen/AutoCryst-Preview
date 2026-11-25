# crystalx_core

Platform-neutral workflow core for the AutoCryst / CrystalX GUI project.

This package owns the crystallographic **workflow**, the **provenance** model, the **external tool
adapters**, and the **candidate lineage**. It deliberately does **not** own the neural network.

## Hard constraints

* **No PyTorch.** `crystalx_core` never imports `torch`, `torch_geometric`, `torch_scatter` or
  `torch_cluster`. Inference backends are reached through a protocol so that CUDA and Apple Silicon
  implementations can be swapped without the core knowing.
* **No CUDA or Apple-specific code.**
* **No third-party runtime dependencies.** The standard library only, so the core installs and tests
  anywhere. A test asserts this rather than trusting it.
* **No proprietary binaries.** SHELX and PLATON are *user-supplied, locally installed* programs. This
  package resolves and invokes them; it never downloads, compiles, bundles, or redistributes them.
* **No solver is privileged.** SHELXT is one backend among several. The selector, the branch model
  and the workflow state machine contain no reference to any particular program — there is a test
  for that too.

## What is implemented

| Area | Module |
| --- | --- |
| Error taxonomy | `crystalx_core.errors` |
| Typed domain model | `crystalx_core.domain.*` |
| Radiation policy and dataset-merge guard | `crystalx_core.domain.radiation` |
| SHELX readers (`.ins`/`.res` header, `.hkl` scan, full `.res`) | `crystalx_core.io.shelx_reader`, `crystalx_core.io.shelx_res` |
| SHELX `.ins` writer, with lock directives | `crystalx_core.io.shelx_writer` |
| Immutable run workspace, hashing, event log | `crystalx_core.workspace.*` |
| Executable resolver, binary-format probe, process runner | `crystalx_core.tools.*` |
| Generic tool adapter with full invocation provenance | `crystalx_core.tools.adapter` |
| Solver portfolio, strategy selector, branch state machine | `crystalx_core.solvers.*` |
| SHELXT backend (code-complete; awaiting successful live positive control) | `crystalx_core.solvers.shelxt` |
| Heavy-atom routes: SHELXC/D, SHELXE, SHELXS-PATT, ANODE (plan-only) | `crystalx_core.solvers.heavy_atom` |
| Peeling-onion workflow state machine | `crystalx_core.workflow.onion` |
| Candidate eligibility gates and ranking | `crystalx_core.workflow.gates`, `.ranking` |
| CLI | `crystalx_core.cli.main` |

### Backend readiness

A backend declares how far it has been taken:

* **`PRODUCTION`** — command shape and output parser both implemented and tested. SHELXT is here.
* **`PLAN_ONLY`** — the command can be built and displayed, but the output parser is not written and
  the command shape has not been verified against a real installation. The branch machine **refuses
  to execute** these unless a caller explicitly opts in, so nothing can produce an uninterpretable
  result. Every heavy-atom route is here today.

## Not yet implemented

cctbx integration (space-group-aware completeness, the `f''` table), geometry and charge-balance
*calculators* (the domain types and the gates that consume them exist), the rigid-fragment library
for NTf₂⁻/OTf⁻/ClO₄⁻ and friends, CrystalX inference wiring, and any GUI.

Where a computation needs information the core does not yet have, it reports `None` with a stated
reason or `INSUFFICIENT_INFORMATION` — never a guess. A gate whose calculator is missing returns
`NOT_EVALUATED`, which **blocks** ranking rather than passing.

## Radiation assumption

The project's working assumption is **Cu Kα unless the reflection file says otherwise**, and the
assumption is recorded per run rather than hidden. A wavelength present in the `CELL` record always
wins.

Datasets from different X-ray sources are **refused for merging**: the wavelength changes the
dispersion corrections, the absorption and the accessible resolution, so a mixed Cu/Mo merge would
look fine and mean nothing.

```bash
crystalx merge-check --dataset a.ins a.hkl --dataset b.ins b.hkl
# exit 0 = compatible, exit 1 = refused with the reason
```

## What the reader refuses to guess

Two things a reflection file does not state outright, and how they are handled:

* **HKLF dialect.** A sixth column is not enough to call a file twinned. HKLF 5 marks the
  non-final members of an overlap group with a *negative* component; a positive sixth column is a
  batch number, and more than six numeric fields is the absorption / psi-scan layout with
  direction cosines. Each inference is reported as a warning naming the guess. An `HKLF` record in
  the instruction file always wins and suppresses the warning.
* **Unmerged anomalous data.** A merged file often retains a handful of accidental
  `(h,k,l)`/`(-h,-k,-l)` coincidences. Those are reported as `FRIEDEL_PAIRS_SPARSE`, which does
  **not** satisfy the prerequisite for SHELXC/D/E or ANODE — the count and the coverage are both
  stated so the crystallographer can see why the route stayed closed.

Both rules exist because real files from a PLATON distribution were misread by the earlier,
simpler versions.

## Run workspace layout

This is what the code actually writes.

```
runs/
└── r-20260818T064500Z-1a2b3c4d_ag-peptide/     <run_id>_<slug>; sorts chronologically
    ├── run.json                                versions, identities, metadata
    ├── result.json                             ranking, stop reason, review checkpoints
    ├── events.ndjson                           append-only, flushed per line
    ├── inputs/                                 verbatim copies, mode 0444
    │   └── manifest.json                       sha256 of every input
    ├── stages/
    │   ├── s0-diagnose/
    │   ├── s1-initial_solution/
    │   │   ├── stage.json
    │   │   ├── selection_trace.json            every rule, verdict and nomination
    │   │   └── branches/
    │   │       └── b00-shelxt/
    │   │           ├── branch.json
    │   │           ├── command_00.json         argv written before the program starts
    │   │           ├── result_00.json
    │   │           ├── provenance_00.json      digests in and out, executable identity
    │   │           ├── shelxt.stdout.log
    │   │           └── outputs/
    │   ├── s2-anchor/
    │   │   └── lock_directives.json            mode, reason, generated SHELXL lines
    │   ├── s3-first_shell/
    │   ├── s4-shell_expansion-01/              repeats, zero-padded shell index
    │   └── s5-finalize/
    └── candidates/
        ├── index.json                          every candidate, kept, with lineage
        └── c000/ c001/ ...
```

Rules: inputs are copied and made read-only; nothing is ever overwritten; a retry is a *new*
iteration; every artefact is content-hashed; a crashed run still leaves a readable record.

## Install and test

From the repository root, bootstrap the supported local development environment:

```bash
sh crystalx_core/scripts/bootstrap_dev.sh
source .venv/bin/activate
cd crystalx_core
pytest
```

On macOS, the bootstrap script clears a Finder-derived hidden flag from `.venv` before and after
the editable install. Python 3.12 otherwise skips the editable-install `.pth` file and the
`crystalx` command cannot import the package. The script ends by running both an import check and
`crystalx --version`.

### Test boundaries

The default run touches **no** crystallographic software:

```bash
pytest                       # 450 passed, 12 skipped, 8 xfailed in the 2026-09-23 owner environment
```

The default suite executes no SHELX, PLATON, cctbx, or model weights. The 8 `xfail` cases are
documented, deliberately unimplemented IDE handoff tasks; they are not counted as passing tests.

Two opt-in groups exist:

```bash
# Read the user's own PLATON distribution in place. Nothing is copied into this repo, because
# PLATON's licence forbids redistribution. 9 tests; these found two real reader defects.
CRYSTALX_PLATON_TEST_DIR=/path/to/platon_instruction/TEST pytest -q

# Invoke real, locally installed binaries. Requires the flag AND the tool to actually resolve.
CRYSTALX_REAL_TOOLS=1 CRYSTALX_TOOLBOX=/path/to/toolbox pytest -m real_tools -q

# The full SHELXT vertical slice against a real case you are happy to solve.
CRYSTALX_REAL_TOOLS=1 CRYSTALX_TOOLBOX=/path/to/toolbox \
CRYSTALX_REAL_CASE_INS=/path/case.ins CRYSTALX_REAL_CASE_HKL=/path/case.hkl \
pytest -m real_tools -q -s
```

The real positive-control test passes only when SHELXT actually produces a parseable solution. A
tool-launch, Rosetta, architecture, or parser failure is a **test failure**, not an acceptable
substitute for a scientific integration result.

A test asserts that the `real_tools` marker cannot leak into any other file.

## CLI

```bash
crystalx doctor --toolbox <dir>        # host arch, Rosetta status, per-tool diagnosis
crystalx diagnose --hkl F --ins F      # cell, wavelength, resolution, Friedel pairs, HKLF
crystalx plan --hkl F --ins F --metal Ag [--force-solver K] [--exclude-solver K]
crystalx merge-check --dataset A.ins A.hkl --dataset B.ins B.hkl
crystalx policy                        # the candidate ordering, stated
```

`plan` prints every selection rule and every nomination with its justification. The crystallographer
is never asked which solver to use.
