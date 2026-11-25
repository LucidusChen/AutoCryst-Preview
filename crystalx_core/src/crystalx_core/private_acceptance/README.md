# Private local acceptance

This milestone provides offline plans and synthetic software verification. It does not launch
proprietary tools. All measurements, complete manifests, descriptions, answers, private plans,
logs and scores stay under ignored local paths. The whole `realdata_examples/` archive is
excluded, including `case.json`, `INDEX.json` and README files. Ignore rules do not erase history.

From the checkout root:

```sh
sh scripts/crystalx.sh python -m crystalx_core.private_acceptance template
sh scripts/crystalx.sh python -m crystalx_core.private_acceptance privacy-audit
sh scripts/crystalx.sh python -m crystalx_core.private_acceptance plan \
  --solver-spec runs/private_acceptance/local-solver.json \
  --case-root /absolute/path/to/the/local/case \
  --toolbox /absolute/path/to/the/local/tools
```

The template is fictional and contains no answer. Save a locally completed copy under ignored
`runs/`; preserve the original case identifier and filenames. `plan` saves a new private directory
with `plan.json` and its SHA-256 pin. It returns exit 3 for BLOCKED. The command prints the saved
location and aggregate status, rather than input excerpts or scientific diagnostics. Inspect the
private plan locally for exact paths, priors, argv, limits and blockers. Never send it to cloud CI.
`--toolbox` supplies a search directory, not an assumed executable. Omit it to use normal
resolver locations; unresolved argv remain explicitly null. `--checkdef` names an existing
validation resource but does not certify that a real process can load it.

Solver and grader specifications are distinct strict contracts. The solver accepts only
`schema_version`, `case_id`, `track`, `inputs`, `priors` and `preparation`; nested fields are also
allowlisted. A descriptive archive manifest is rejected. Inputs contain only relative INS/RES and
HKL paths contained by the explicit case directory. Answer-bearing fields, model atoms in blind
inputs, file includes, traversal and symlink escapes are rejected. No scientific files are
discovered by a glob. HKL is already integrated intensity data; detector image integration is
outside this workflow.

The tracks are:

- `DE_NOVO`: original INS/HKL and independent owner-confirmed chemistry. Expert facts and
  answer-derived handedness/counts are excluded. Independent prior labels are assumptions, not
  successful AI identification.
- `COMPONENT_REGRESSION`: an explicitly supplied expert model/HKL pair. The component pipeline
  uses the same `RunService`, bounded SHELXL, CIF+FCF-gated PLATON and viewer stages. It skips
  SHELXT and has a distinct retained-evidence verifier. It cannot establish initial-solution
  success. Real component dispatch remains disabled in this offline milestone.
- `UNSOLVED`: no ground-truth answer. A related structure can only be a declared advisory
  reference in the independent grader; it is not truth or a silent solver input.
- `INITIAL_MODEL_RECOVERY`: original measured reflections plus an explicitly declared automatic
  initial-model seed. The seed pins its model hash and compatible HKL hash with owner provenance;
  it is distinct from the held-out expert answer. Missing seed does not search for another RES
  or substitute an answer. This route refuses the one-shot solve API and has its own blocked
  iterative plan. See the core iterative-reconstruction guide for the runnable synthetic loop.

`--metal`/`--element` style priors do not rewrite SFAC/UNIT. If a prior is absent from the
effective SFAC, planning records a blocker. `preserve_input` makes no instruction edit. An
`explicit_reviewed_composition` requires a complete element/count list, owner review and official
semantics references; those strings are declarations, not authenticated approvals. This milestone
does not invent uncertain counts or approve an instruction-preparation policy.

Preflight inspects the actual selected toolchain, including its preferred PLATON. Missing tools,
unsupported/script formats, invalid Mach-O structure, incompatible platform/native architecture,
quarantine, unresolved dependencies, loader overrides or missing/unbound validation resources
fail closed. Local dependency inspection is bounded and recursive. Static evidence is separate
from startup and scientific evidence; cryptographic signature/Gatekeeper approval is not claimed.
The standalone native inventory's open findings are not fixed by this stricter acceptance gate.

Default plans allow at most one invocation each: SHELXT 900 seconds, SHELXL 600 seconds with
four isotropic cycles, then PLATON 600 seconds only with both CIF and FCF. CPU is capped at 3600
seconds per process; file/open-file/niceness bounds come from the shared runner. CPU limits are
not aggregate process-tree quotas. The shared macOS runner has no reliable memory ceiling;
plans record this as a blocker rather than claim a memory limit. Working directories are fresh,
and inputs/executables/resources must be rechecked before dispatch. Exact future execution
requires both `CRYSTALX_REAL_TOOLS=1` and owner authority for the pinned case/track/plan. There
is no `run` command in this milestone, and setting the environment variable cannot enable one.

For a completed future run, `seal_run` requires an explicit output role/path map, case identifier
and terminal state. It hashes all selected outputs, writes exclusively, and rejects links, path
escapes and changes. It never discovers an answer. `grade_sealed_run` requires the externally
retained seal digest and matching case/track before reading any answer. The CLI can verify this:

```sh
sh scripts/crystalx.sh python -m crystalx_core.private_acceptance grade \
  --run runs/private_acceptance/sealed-local-run \
  --seal-sha256 THE_EXTERNALLY_RETAINED_DIGEST \
  --grader-spec runs/private_acceptance/local-grader.json \
  --case-root /absolute/path/to/the/local/case
```

No audited matching provider ships here, so scientific grading defaults to `NOT_EVALUATED`
without reading the answer. An optional provider must have externally audited periodic, origin,
symmetry, setting and equivalent-site matching capabilities, exact pinned identity and approved
owner thresholds. There is no naive Cartesian RMSD fallback. Even valid matching evidence requires
scientific review; final R metrics are not equated across different reflection/model settings.

This is an application input and evidence boundary, not an operating-system filesystem sandbox.
Same-user binaries/Python code can access other files. Filesystem isolation and an atomic
check-to-exec guarantee are not supplied. Ordinary tests contain generated synthetic data and
never run proprietary tools. Status layers remain separate: `SOFTWARE_TESTED`,
`REAL_PROCESS_VERIFIED`, and `SCIENTIFICALLY_REVIEWED`.
