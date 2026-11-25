# AutoCryst local helper (S0)

This Python helper exposes the existing reconstruction service to a future native frontend.
No Swift, shaders, packaging, installation, network listener or scientific engine is included.
Use an existing owner-selected Python environment and checkout imports; no dependency download
is required for synthetic tests. The package metadata is a development placeholder, not a release.

```
PYTHONPATH=crystalx_core/src:crystalx_web/src:crystalx_bridge/src \
  .venv/bin/python -m crystalx_bridge.helper \
  --repository <repo> --runs-root <repo>/runs/native-helper --synthetic-demo
```

Real mode omits `--synthetic-demo` and uses installed adapters/official pinned weights. A helper
started in demo mode accepts only exact built-in demo bytes, with the supported synthetic
providers explicitly injected. It never loads trained weights or starts a scientific executable.

## Transport

Local UTF-8 JSON lines on stdin/stdout; other Python/native-worker prints go to stderr. One
request per line, at most 1 MiB including newline; one response at most 16 MiB. The sequential
control reader naturally applies pipe backpressure, without an unbounded request queue.
Submissions and local diagnostics return promptly and run in cancellable workers. Do not
conflate a delivered cancellation request with an observed terminal state. A bounded response
may be refused rather than silently truncated.

Every envelope contains `protocol`, `id`, `op`, `params`. Each response echoes the id and current
bridge protocol and backend contract. Unknown versions, unknown operation fields, duplicate
JSON keys, nonfinite numbers and malformed UTF-8 are refused. `hello` negotiates the versions;
preview/submit require explicit acceptance of `FRONTEND_CONTRACT`. The protocol schema is
`protocol.schema.json`. Example request/response/error frames for every operation and each
diagnostic action are in `tests/golden/operations.json`; examples illustrate representative
results while full field lists remain the imported backend/dataclasses, not frozen copies here.

`describe` imports the Web's existing Gradio-free form groups, labels, stop and decision tables.
It also describes the actual request/route/policy dataclasses. It reports the Web text review's
contract separately. A pending Web review is visible; the bridge never rewrites its contract
or invents localized explanations. This package imports no Gradio, Torch or scientific program
merely to describe its form.

## Operations and durable ownership

| Operation | Behavior |
|---|---|
| hello / describe | Negotiate protocol/contract, capabilities and shared form presentation |
| preview | Core static validation; no scientific execution |
| submit | Public asynchronous `submit`, returning run_id; no generated-initial entrypoint |
| status / summary | Read durable backend records, including historical contract and stop_detail |
| cancel | Deliver only to the owning service token; client polls status afterwards |
| export | Copy verified sealed candidate roles into a new user-chosen directory; never overwrite |
| scene | Existing core viewer payload plus cell basis, units and candidate/seal identities |
| checkcif | Explicit `start`, `status`, `cancel` for a separate bounded local PLATON diagnostic |

Only original solver INS/HKL and declared priors enter reconstruction. Unknown request fields,
including reference answers, are refused. Resolve parent aliases such as macOS `/var` without
resolving leaf aliases that the core must reject. No broad exception disables integrity checks.

Candidate reads use the supported `crystalx_core.app.candidate_outputs` facade: validated run_id,
external seal pin, terminal phase equality, and rehashed sealed outputs. RES-only archives stay
inspectable/exportable with explicit missing roles; checkCIF cannot regenerate absent CIF/FCF.
Scenes use at most 1000 sites by default (explicit maximum 2000) and a 4 MiB model limit. Temporary
private copies are removed; geometry comes from `build_viewer_payload`. Display contacts and
elements remain unvalidated. This is not yet the S2 10,000-site GPU benchmark.

Diagnostic request/state/result records live under that run's `bridge/checkcif/<checkcif_id>/`.
Only thread/cancellation handles are volatile. Owner process death is shown as interrupted;
restarting the helper only reads evidence and does not resume, reconstruct or repeat a diagnostic.
Request ids correlate messages; they are **not idempotent submission keys**. If a connection dies
before an acknowledgement, inspect durable runs before any explicit resubmission. A native client
must not retry mutation requests automatically. EOF ends the helper; active work may then be
interrupted and must be inspected after restart. No service state is inferred from an export copy.

Local checkCIF is not the IUCr online service. Its result is distinct from worker completion;
`NOT_EVALUATED`, alerts and process failure never become a scientific PASS. The driver uses CPU,
as do all SHELX/PLATON adapters; MPS is solely the separately pinned inference device.

## Tests

```
PYTHONPATH=crystalx_core/src:crystalx_web/src:crystalx_bridge/src \
  .venv/bin/python -m pytest -q crystalx_bridge/tests
```

Tests use only built-in fictional data and supported providers, never core-test fixture imports.
They cover framing, version negotiation, shared presentation, async cancellation, crash/restart,
pin tampering, export/scene identity and separate local-diagnostic abstention. No engine accuracy,
native UI performance or signed application distribution is claimed by S0.
