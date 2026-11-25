"""Hand-authored synthetic SHELXT-style comments and tolerant listing regressions.

The fixture values and labels were invented for parser coverage. These tests establish software
behaviour only; they do not establish verification against real program output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crystalx_core.io.shelxt_output import (
    ShelxtSolutionSummary,
    parse_solution_rems,
    scan_lxt,
)

# Newly invented records; no archive identifier, composition or measured score is copied.
SYNTHETIC_REMS = (
    "Old TITL synthetic-rem-parser in P1",
    "SHELXT solution in P1: R1 0.271, Rweak 0.083, Alpha 0.000",
    "<I/s> 0.000 for 0 systematic absences, Orientation as input",
    "Formula found by SHELXT: C18 Cu2 N4 O6",
)


def test_reads_the_figures_of_merit_from_a_synthetic_solution():
    summary = parse_solution_rems(SYNTHETIC_REMS)

    assert summary.space_group == "P1"
    assert summary.r1 == pytest.approx(0.271)
    assert summary.r_weak == pytest.approx(0.083)
    assert summary.alpha == pytest.approx(0.000)
    assert summary.n_systematic_absences == 0
    assert summary.formula == "C18 Cu2 N4 O6"
    assert len(summary.raw_lines) == 3, "the 'Old TITL' line is not a figure of merit"


def test_unrecognised_rem_lines_are_ignored_not_guessed_at():
    summary = parse_solution_rems(("something else entirely", "REM more prose"))
    assert not summary.found_anything
    assert summary.raw_lines == ()


def test_a_high_rweak_is_reported_as_a_space_group_concern():
    """The failure this catches: a wrong space group with a respectable R1.

    Rweak is the diagnostic, because a group that imposes the wrong systematic absences leaves the
    reflections it expects to be weak looking strong.
    """
    summary = ShelxtSolutionSummary(space_group="P2(1)/m", r1=0.15, r_weak=0.55, alpha=0.0)
    concerns = summary.concerns()
    assert any("wrong space group" in c for c in concerns)


def test_a_strong_forbidden_reflection_is_reported():
    summary = ShelxtSolutionSummary(
        space_group="P2(1)", r1=0.15, r_weak=0.2, mean_i_over_sigma_absences=8.4,
        n_systematic_absences=37,
    )
    assert any("forbidden" in c and "wrong" in c for c in summary.concerns())


def test_a_clean_solution_raises_no_concerns():
    assert parse_solution_rems(SYNTHETIC_REMS).concerns() == ()


def test_concerns_are_separate_observations_not_a_score():
    """Compressing these into one number is how a wrong space group passes on a good R1."""
    summary = ShelxtSolutionSummary(
        space_group="X", r1=0.15, r_weak=0.55, alpha=0.9,
        mean_i_over_sigma_absences=8.0, n_systematic_absences=12,
    )
    assert len(summary.concerns()) == 3
    assert not hasattr(summary, "quality_score")


# --------------------------------------------------------------------------------------
# .lxt -- tolerant only. IDE-FILL(lxt-table) tightens this after the first real run.
# --------------------------------------------------------------------------------------


def test_a_missing_listing_is_reported_rather_than_raising():
    listing = scan_lxt(Path("/nonexistent/synthetic-example.lxt"))
    assert listing.trials == ()
    assert any("may not have written one" in w for w in listing.warnings)


def test_the_lxt_scanner_never_claims_its_format_is_verified(tmp_path: Path):
    """IDE-FILL(lxt-table) is not done, and the flag must say so.

    ``format_verified`` may only become ``True`` once a real SHELXT listing has been read and this
    parser checked against it. Until then no caller may make a selection decision from these rows.
    """
    path = tmp_path / "x.lxt"
    path.write_text("P2(1)      0.271  0.083  0.000  0.85\nsome prose line\n", encoding="utf-8")
    listing = scan_lxt(path)

    assert listing.format_verified is False
    assert any("has not been verified" in w for w in listing.warnings)


def test_unreadable_lines_are_kept_rather_than_dropped(tmp_path: Path):
    """The unparsed lines are the specification for IDE-FILL(lxt-table)."""
    path = tmp_path / "x.lxt"
    path.write_text("P2(1)   0.271  0.083\nnot a row at all\n\nP-1  0.30  0.40\n", encoding="utf-8")
    listing = scan_lxt(path)

    assert [t.space_group for t in listing.trials] == ["P2(1)", "P-1"]
    assert listing.unparsed_lines == ("not a row at all",)
