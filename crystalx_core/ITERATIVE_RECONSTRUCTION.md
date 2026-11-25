# Multi-route pretrained reconstruction

The 2026-11-11 real-run compatibility and separate diagnostics checkpoint is recorded in
`ai_records/acceptance/20261111_codex_real_pipeline_and_diagnostics.md`. Experimental candidates
exist privately and remain scientifically unvalidated. Protein C/D/E requires suitable explicit
metadata. Older synthetic-only restrictions below describe their historical entry points.

The 2026-11-10 owner workflow is implemented by `app.reconstruction_backend` and
`workflow.refinement_cycle`, using the existing OnionWorkflow. It accepts original atom-free
INS/HKL, an initial route and a numerical cycle policy. It does not accept expert/reference
models or train a checkpoint. Actual traditional programs must pass tool availability and
input/output identity checks before execution. The new entry does not inherit the legacy
synthetic-only dispatch prohibition described later in this guide.

Supported initial routes are SHELXT; explicit protein experimental phasing via SHELXC/D/E;
and SHELXS direct/Patterson or small-molecule SHELXD with explicit search parameters.
Protein requires suitable experimental datasets, group, solvent fraction and substructure
parameters. Its trace PDB is converted only within the supported same-cell/setting scope;
a heavy-atom substructure RES is not substituted for a protein trace. Residue-aware refinement,
general reindexing and traces above 999 sites remain unsupported by this first converter.

Use `ReconstructionBackendService.preview/submit/status/summary/cancel` for a frontend.
The CLI consumes the same serialized `ReconstructionRequest`:

```sh
sh scripts/crystalx.sh python -m crystalx_core.app.reconstruction_cli \
  --runs-root runs/reconstruction preview --request runs/requests/experiment.json
sh scripts/crystalx.sh python -m crystalx_core.app.reconstruction_cli \
  --runs-root runs/reconstruction run --request runs/requests/experiment.json
```

All measured-data requests and artifacts stay in ignored `runs/`. Preview does not launch
programs. The actual published heavy-element checkpoint is loaded with `weights_only=True`
and pinned source/checkpoint hashes. Numerical/preprocessing parity and scientific calibration
remain unverified; a real pretrained prediction is not a scientific acceptance claim. This
entry uses the non-hydrogen head, not hydrogen prediction or model training.

An independent `metal_only` option removes nonmetal seed sites in a derivative copy, retains
selected metal elements and fixes those coordinates in actual SHELXL input. The returned
fixed-coordinate encodings are verified and decoded into a separate model; raw RES/LST survive.
Partial-model phases are recalculated each round. This is not a claim to fix all experimental
phases. New ligand sites use selected observed residual Q coordinates and pretrained element
alternatives; the Q list is partial, not a full map or a complete framework generator.

When `iterations=None`, two successive compatible important-coordinate/R1 changes below
`coordinate_tolerance_angstrom=0.02` and `r1_tolerance=0.001` stop automatically. That normally
requires three successful rounds. Important IDs are explicit or default to high-Z initial sites,
with non-H fallback. Explicit iterations disable that early numerical stop. Safety round/time/
failure/peak caps, cancellation, reflection/setting identity and malformed-file stops remain.
`iterations` is distinct from `shelxl_cycles` (default four internal cycles per outer round).
These are configurable development thresholds, not scientific convergence criteria.

Scientific gates are explicitly deferred under the owner's 2026-11-10 instruction. Waived gates
retain candidate status `refined`, never `validated`, and cannot establish scientific convergence.
Bad trials can roll back without losing the previous model; output seals pin the retained
`ai_solved.res` and available CIF/FCF. Separate `app.checkcif` local PLATON diagnostics and
`app.model_comparison` expert comparison verify that external seal pin before reading inputs.
Warnings do not suppress the raw candidate. The expert answer is never a solve feature.

See `../ai_records/advisory/20261110_codex_claude_reconstruction_frontend_handoff.md` for the
complete Claude contract and `../ai_records/acceptance/20261110_codex_multiroute_backend.md`
for actual software, pretrained and real-data preflight evidence.

# Earlier guarded synthetic demonstrations

Run from the checkout root:

```sh
sh scripts/crystalx.sh python -m crystalx_core.app.reconstruction_demo
sh scripts/crystalx.sh python -m crystalx_core.app.reconstruction_file_demo
```

This uses generated synthetic inputs/providers only. No SHELX/PLATON process or trained
checkpoint runs. The output names two fresh private directories under ignored `runs/`: the
first run and a restarted child run. Expected first decisions are `ACCEPT_PROVISIONAL`,
`REJECT`, `ACCEPT_PROVISIONAL`, followed by `no_progress_not_convergence`. The rejected
trial has a deliberately lower synthetic R value and a failing geometry check. The restarted
run loads a pinned accepted checkpoint, preserves the counters and accepted constraints, and
adds another site without overwriting parent history. These results test control logic, not
crystallographic accuracy.

The second command exercises real controlled Python child processes through ExternalToolAdapter
and ProcessRunner. It writes and parses SHELXL-shaped files but performs no least-squares
calculation. Four child invocations give the same accept/reject/accept sequence. Its explicit
synthetic evaluator supplies scripted software checks. Without that evaluator, all scientific
checks and convergence remain NOT_EVALUATED; the service blocks and retains an initial model
without exporting an accepted-model artifact. Neither command starts a proprietary program.

## Existing controller, application entry and retained evidence

`RunService.reconstruct(...)` delegates to `app.reconstruction.ReconstructionService`, which
registers `workflow.reconstruction.ReconstructionExecutor` with the existing `OnionWorkflow`.
There is one iteration driver. The ordinary one-pass solve path remains available as a distinct
diagnostic; it does not become iterative merely because its SHELXL job has several cycles.
Web/desktop UI integration is a later task in Claude's separate lane.

Each trial writes its model proposal, actual generated `trial.ins`, unchanged `trial.hkl`,
separate `lock-free-model.res`, assessment and decision. The file-backed provider also retains
the original `trial.res`/`trial.lst`, process logs, invocation and normalization/evidence records.
The controller never overwrites those raw outputs. An element flip changes the SFAC index
while keeping the site label stable. A new site's coordinates come exclusively from an explicitly
identified density-provider peak. Inference supplies element alternatives/probabilities, not
experimental density or coordinates. At most one site is changed per proposal. Alternative
elements and supporting observation identities survive in the record.

The current runnable providers are deliberately synthetic. Their R/residual/check values are
scripted software probes. They cannot be mixed with measured data: the application requires
byte-identical built-in synthetic INS/HKL and a deterministic synthetic seed, and rechecks the
staged copies. Actual provider implementations and real dispatch remain blocked.

Decisions require the same reflection/setting identity and data count, compatible R1/wR2/GooF,
residual-region evidence, controlled parameter growth, heavy-site drift and completed checks
for geometry, coordination, occupancy/ADP, chirality/symmetry and restraints. Lower R alone
cannot buy acceptance. Missing checks require review. The demonstration's numerical limits
are software policies, not approved tolerances for the private benchmark.

The executor returns a pending model. Only an OnionWorkflow commit acknowledgment after
its own budgets, gates and review checks promotes it. Abort restores the accepted model and
active lock set; issued/rejected lock history and candidates remain visible. The next stage's
pre-execution cancellation cannot roll back an earlier committed stage. Decision files contain
the proposal verdict; separate controller commit records acknowledge actual acceptance.

## Reversible constraints

`refinement.locks.apply_locks` generates constraints from a retained lock-free baseline and
the active ledger. Soft locks emit a narrow, explicit DFIX subset into the actual input.
Hard coordinate locks require externally approved ledger IDs and use reviewed SHELXL encoding.
No hard lock is automatically approved by the reconstruction executor. Exploratory/free
directives introduce no new constraint and never silently free occupancy/disorder links.

Unsupported selectors, included files, coupled/ambiguous coordinate contexts and unmanaged
instruction injections block. Rebuilding with no locks restores the exact baseline. A provider
must explicitly retain/return its lock-free model; feeding back managed input as a new baseline
is refused. `normalize_managed_output` reproduces the baseline/application/approved lock IDs,
removes only its exact managed block and decodes only its newly approved fixed XYZ fields.
Free refined values and pre-existing constraints survive. Missing/moved/reformatted blocks,
changed scientific settings (including WGHT), altered identity or changed parameter freedom
block. Actual SHELXL may reformat outputs: that dialect is not yet verified for production.

The synthetic scaffold is explicitly a provisional hypothesis (`validated=False`); only this
guarded demonstration permits exploration without a validated anchor. The normal OnionWorkflow
default still requires a validated anchor before expansion. No scientific anchor gate was waived
for a real run.

## Bounds, cancellation and restart

The controller enforces stage reservations before work and checks returned usage after it.
The reconstruction executor separately caps cumulative rounds, proposals, attempted atom
additions, tool-call reservations, inference calls, failures, stagnant rounds and active elapsed
time. Rejected attempts consume budget. SHELXL cycles per trial are a separate bound. No new
lineage or no new atom is interpreted as scientific convergence. Explicit convergence must
come from a qualified executor with an eligible accepted candidate; this synthetic demo stops
for review and does not assert scientific convergence.

Python providers must cooperate with checkpoints/cancellation; this does not preempt arbitrary
blocking Python code. The file-backed fixture forwards actual remaining timeout/cancellation
to ProcessRunner and records CPU/file/open-file limits, interpreter and literal-program pins.
It uses isolated Python without site startup or loader overrides. File snapshots reject
symlinks/hardlinks before hashing; bounded reads and post-evaluator rechecks protect the files.
The shared macOS runner still has no reliable memory
ceiling, which remains a real-run blocker.

An in-flight checkpoint records a charged call before provider entry. Restart requires its
externally retained SHA-256, unchanged input/policy/inference-audit pins, valid nonnegative usage,
and consistent model text/site fields. It creates a new child workspace and never rewrites the
parent. Resource limits are not refunded on restart. The pin is an integrity boundary, not a
same-user filesystem sandbox or authorization to replay old checkpoints indefinitely.

## Real inference and data roles

`app.reconstruction_inference.create_apple_reconstruction_bridge` is an optional lazy binding
to the existing Apple backend and preprocessing. It requires explicit checkpoint, full source,
preprocessing implementation/policy and actual reference-comparison report pins. It recomputes
saved measured comparison consistency and rejects synthetic/BLOCKED reports before loading a
backend. An explicit parsed-model callback supplies only a known retained model, complete
symmetry and Q computational descriptors. Stable model/peak IDs and prepared-request identity
are checked again at prediction. No dependency or weight is downloaded. File pins and caller
provenance do not authenticate a publisher or approve numerical/scientific tolerances.

No real trained inference was performed in this milestone. Missing inference blocks AI-required
reconstruction; the explicit `SHELX_ONLY_DIAGNOSTIC` capability does not pretend to provide AI
proposals. Q-peak lists remain partial observations; requiring a full map blocks without an
audited full-map provider. Global peak extrema alone are not density feedback.

Private contracts distinguish `MEASURED_REFLECTIONS`, `INSTRUMENT_INSTRUCTIONS`,
`AUTOMATIC_INITIAL_MODEL`, and `EXPERT_FINAL_ANSWER`. `INITIAL_MODEL_RECOVERY` requires a
pinned explicitly declared automatic seed/HKL pair. Missing seed blocks that track without
substitution; strict blind `DE_NOVO` still rejects atom-bearing seeds. The expert-model component
regression remains separate. A future newly generated SHELXT seed needs a verified handoff
with reflection lineage. `app.reconstruction_handoff` now verifies explicit source-run,
stage/branch/outcome/invocation records; original input pins; new automatic RES and matching
HKL output pins; and the exact instruction transformation. It is read-only and accepts a
succeeded initial stage even when later full-workflow stages fail. Original-HKL fallback,
including equal CELL plus an unverified orientation REM, remains blocked. An independently
retained receipt digest is required on use; the receipt grants no scientific/launch authority.

`ReconstructionService.run_from_initial_handoff` stages those explicit outputs into a fresh
synthetic run, retains the receipt, and charges the selected initial invocation's call/duration
before refinement. Original measured and paired reflections remain separately pinned. The
same byte-identical demo-input/seed gates apply. These counters cover the selected synthetic
invocation, not all other jobs in the source run. Real whole-chain budget/authority, source
output dialect and scientific provider validation remain prerequisites. Real dispatch refuses
before reading a handoff or calling providers.

`FileEvidenceStore` and `QPeakDensity` use an exact normalized-model/basis lookup. They recheck
raw RES/LST, HKL, normalized model and invocation pins and rederive Q peaks from the raw output.
Rejected trials do not become a later accepted model's density source. The synthetic label
cannot be promoted to real evidence. Lists are partial; they contain no inferred regional map
or scientific convergence evidence. Persistent file-evidence restoration into a restarted
file-backed run remains separate work; the in-memory demonstration still covers pinned restart.

The expert final model belongs only to the independent sealed-output grader. Full periodic,
origin/setting/symmetry-aware heavy-site and ordered-framework topology assessment needs an
audited provider and owner-approved tolerances. Solvent, disorder and hydrogen need separate
reports. No such scientific acceptance is claimed here. The repeatedly used difficult case is
a development/regression benchmark; later independent cases require frozen policies.
