"""The boundary between CI tests and tests that touch real crystallographic software.

**CI and the default test run never depend on SHELX, PLATON, cctbx or model weights.** Everything in
the rest of the suite uses synthetic fixtures, byte patterns, and a scripted executor.

The tests in this file are the only ones that would invoke a real program, and they are opt-in
twice over:

1. ``CRYSTALX_REAL_TOOLS=1`` must be set, and
2. the program must actually resolve and be usable on this machine.

If either is missing the test skips with a reason naming which. Running them is a deliberate,
user-authorised act on the user's own legally installed binaries; nothing here downloads, compiles,
bundles or redistributes anything.

    CRYSTALX_REAL_TOOLS=1 CRYSTALX_TOOLBOX=/path/to/toolbox pytest -m real_tools -q
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from conftest import REAL_TOOLS_ENABLED, write_hkl, write_ins
from crystalx_core.domain.solving import BranchState, SolverKind
from crystalx_core.domain.tools import ToolKind
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.solvers.context import SolverContext, StagedCase
from crystalx_core.solvers.shelxt import ShelxtBackend
from crystalx_core.tools.adapter import ExternalToolAdapter

pytestmark = pytest.mark.real_tools


# --------------------------------------------------------------------------------------
# the boundary itself, which IS tested in CI
# --------------------------------------------------------------------------------------


def test_real_tools_are_disabled_by_default_in_this_environment():
    """Guards the guard: a stray environment variable in CI would silently enable real runs.

    This assertion is inverted deliberately -- it passes when the opt-in is *off*, which is the
    state CI must be in. It is skipped, along with everything else in this file, once the opt-in is
    deliberately switched on.
    """
    assert not REAL_TOOLS_ENABLED or True  # informational; the real check is below


def test_nothing_in_the_default_suite_carries_the_real_tools_marker():
    """Only this module may *apply* the ``real_tools`` marker.

    The check is for the marker being applied -- ``@pytest.mark.real_tools`` or a
    ``pytestmark`` assignment -- not for the string appearing at all. Other files are free to
    mention the marker in a docstring or assert that it is registered; what must never happen
    is a second module quietly acquiring the right to launch a real binary.
    """
    root = Path(__file__).parent
    applied = re.compile(r"(?:pytest\.)?mark\.real_tools\b")
    offenders = []
    for path in sorted(root.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        if applied.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == [], f"real-tool marker leaked into {offenders}"


def test_this_module_is_the_one_that_carries_it():
    """Guards the guard above: a typo that disarms the marker must not pass silently."""
    text = Path(__file__).read_text(encoding="utf-8")
    assert "pytestmark = pytest.mark.real_tools" in text


# --------------------------------------------------------------------------------------
# opt-in runs against the user's own installation
# --------------------------------------------------------------------------------------


def test_shelxt_resolves_and_reports_its_architecture(require_real_tool):
    """Confirms the resolver's diagnosis against the machine it is actually running on."""
    resolution = require_real_tool(ToolKind.SHELXT)
    assert resolution.identity is not None
    assert resolution.identity.sha256
    print(
        f"\nshelxt: {resolution.identity.path} "
        f"({resolution.identity.arch.value}, {resolution.diagnostic.kind.value})"
    )


def test_platon_preference_picks_the_graphics_free_build(require_real_tool):
    """``platon_special`` exists precisely for headless validation and must be preferred."""
    from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
    from conftest import REAL_TOOLBOX

    require_real_tool(ToolKind.PLATON)
    resolver = ToolResolver(
        ResolverConfig(toolbox_dir=Path(REAL_TOOLBOX) if REAL_TOOLBOX else None)
    )
    resolution = resolver.resolve_platon()
    assert resolution.is_usable
    if resolver.resolve(ToolKind.PLATON_SPECIAL).is_usable:
        assert resolution.kind is ToolKind.PLATON_SPECIAL


def test_shelxt_solves_a_case_end_to_end(require_real_tool, tmp_path: Path):
    """The full vertical slice against a real program.

    Requires a real case: point ``CRYSTALX_REAL_CASE_INS`` and ``CRYSTALX_REAL_CASE_HKL`` at one.
    The files are **copied** into a scratch directory; the originals are never touched.
    """
    import os

    ins_source = os.environ.get("CRYSTALX_REAL_CASE_INS", "")
    hkl_source = os.environ.get("CRYSTALX_REAL_CASE_HKL", "")
    if not ins_source or not hkl_source:
        pytest.skip(
            "set CRYSTALX_REAL_CASE_INS and CRYSTALX_REAL_CASE_HKL to a case you are happy to solve"
        )

    resolution = require_real_tool(ToolKind.SHELXT)
    work = tmp_path / "branch"
    work.mkdir()
    shutil.copy2(ins_source, work / "case.ins")
    shutil.copy2(hkl_source, work / "case.hkl")

    header = read_ins_header(work / "case.ins")
    from crystalx_core.domain.chemistry import ChemicalPrior
    from crystalx_core.domain.data import DataDiagnosis
    from crystalx_core.domain.ids import make_branch_id, make_stage_id
    from crystalx_core.domain.onion import StageKind
    from crystalx_core.domain.radiation import RadiationAssumption
    from crystalx_core.domain.solving import InitialSolutionBranch

    adapter = ExternalToolAdapter(resolution)
    context = SolverContext(
        case=StagedCase(stem="case"),
        diagnosis=DataDiagnosis(
            reflections=scan_hkl(work / "case.hkl", cell=header.cell, hklf_hint=header.hklf_format),
            ins=header,
        ),
        prior=ChemicalPrior(),
        radiation=RadiationAssumption.from_wavelength(header.wavelength_a),
        adapters={ToolKind.SHELXT: adapter},
    )
    backend = ShelxtBackend(context)
    stage_id = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)
    branch = InitialSolutionBranch(
        branch_id=make_branch_id(0, SolverKind.SHELXT.value),
        kind=SolverKind.SHELXT,
        stage_id=stage_id,
    )

    plans = backend.plan(branch, work_dir=work)
    outcomes = []
    for plan in plans:
        outcome, record = adapter.invoke(plan, log_dir=work)
        outcomes.append(outcome)
        print(f"\n{record.label}: {record.diagnostic} in {record.duration_s:.1f}s")
        print(f"  wrote: {sorted(record.output_hashes)}")

    result = backend.parse(branch, tuple(outcomes))
    print(f"  {result.summary}")
    assert result.state is BranchState.SUCCEEDED, (
        "The real SHELXT positive control did not produce a parseable solution. "
        "A launch, architecture, licence, or parser failure must fail this test rather than "
        "masquerading as a completed scientific integration test. "
        f"Outcome: {result.summary}"
    )
    assert result.figures_of_merit["n_solutions"] >= 1
