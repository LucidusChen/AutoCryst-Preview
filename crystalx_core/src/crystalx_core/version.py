"""Single source of truth for the core version."""

from __future__ import annotations

__all__ = ["CORE_VERSION", "RECORD_SCHEMA_VERSION"]

#: Version of the ``crystalx_core`` package.
#:
#: ``pyproject.toml`` reads this attribute (``dynamic = ["version"]``) rather than repeating
#: the string, so the two can never disagree. ``test_packaging_metadata`` asserts it anyway.
#:
#: 0.1.0 -- P1: domain model, workspace, resolver, selector, branch machine, gates, ranking.
#: 0.2.0 -- P2.1: tool adapter with provenance, SHELX .ins writer and .res reader, SHELXT
#:          backend, plan-only heavy-atom routes, onion workflow, radiation merge guard.
#: 0.3.0 -- P2.2a: exact Sohncke/chirality analysis and the precondition that blocks on it,
#:          composition-trust detection for instrument-written .ins files, rule R10.
#: 0.4.0 -- P2.2b: optional-provider protocols, the gemmi adapter behind the [geo] extra,
#:          Niggli lattice comparison, and the SHELXT solution-output reader.
#: 0.4.1 -- answer-key leakage guard on ReferenceStructure; git preflight script.
#: 0.5.0 -- declared-element gap detection (a missing S/F cannot hold an NTf2 anion), P1 reported
#:          as trivially Sohncke, --element flag, Cell_Corrected staged as a second benchmark.
#: 0.5.1 -- macOS-safe editable-install bootstrap, CLI-install verification, and current status docs.
#: 0.5.2 -- live-SHELXT positive controls fail honestly; Rosetta marker status is not a launch claim.
#: 0.6.0 -- P3.0 Web MVP: frontend-independent application service (``crystalx_core.app``) and
#:          ``crystalx solve``; bounded SHELXL preliminary refinement and PLATON diagnostics with
#:          tolerant parsers; process-group cancellation and OS resource limits in the runner;
#:          viewer geometry; synthetic demo case and a mock toolchain that refuses real data.
#: 0.6.1 -- P3.1 corrective round (Codex review 2026-10-14): absolute executable paths checked
#:          against their recorded digest before launch; whole-process-group cleanup after
#:          cancellation, timeout and normal exit; convergence not assessed (raw mean and maximum
#:          shift/su); classified SHELXL messages; ``crystalx verify-run``; metric-complete
#:          boundary contacts; route-specific SHELXT input chirality; Python 3.10 serde fix.
CORE_VERSION = "0.6.1"

#: Version of the on-disk run-record schema.
#:
#: Bump this whenever the layout or the meaning of any file written into a run
#: workspace changes, so that an old record can still be identified and read.
#: 1 -- P1/P2 records: run.json, events.ndjson, inputs/, stages/, candidates/, result.json.
#: 2 -- solve runs add request.json, state.json, summary.json, refinement/, validation/, viewer/
#:      and errors/; every version-1 file keeps its meaning.
RECORD_SCHEMA_VERSION = 2
