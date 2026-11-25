"""The optional-provider boundary, and the contracts the IDE is filling in.

Two jobs:

1. Prove the core still works with **no** provider registered — absence must be reported, never
   estimated around.
2. Hold the specification for the `IDE-FILL` stubs. Those tests are written *before* the code and
   marked ``xfail(strict=True)``: while the stub raises ``NotImplementedError`` they report as
   expected failures and the suite stays green; the moment an implementation makes one pass, strict
   mode turns the XPASS into a **failure** that says "this is done, delete the marker". That is the
   signal, and it is deliberately impossible to miss.

Tests that need gemmi skip cleanly when it is absent, so CI stays dependency-free.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from crystalx_core.domain.providers import (
    ChiralityCrossCheck,
    LatticeRelation,
    ReducedCell,
    SpaceGroupInfo,
    clear_symmetry_provider,
    get_symmetry_provider,
    register_symmetry_provider,
)

gemmi = pytest.importorskip("gemmi", reason="optional extra: pip install -e '.[geo]'")

from crystalx_core.providers import gemmi_provider  # noqa: E402


@pytest.fixture(autouse=True)
def _no_leaked_provider():
    clear_symmetry_provider()
    yield
    clear_symmetry_provider()


# --------------------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------------------


def test_no_provider_is_registered_by_default():
    """Importing the core must not install anything."""
    assert get_symmetry_provider() is None


def test_registering_and_clearing_round_trips():
    provider = gemmi_provider.GemmiProvider()
    register_symmetry_provider(provider)
    assert get_symmetry_provider() is provider
    clear_symmetry_provider()
    assert get_symmetry_provider() is None


def test_a_missing_reducer_reports_undecidable_rather_than_comparing_raw_parameters():
    comparison = gemmi_provider.compare_lattices(None, (10, 10, 10, 90, 90, 90))
    assert comparison.relation is LatticeRelation.UNDECIDABLE
    assert comparison.target is None


# --------------------------------------------------------------------------------------
# cell reduction and lattice comparison
# --------------------------------------------------------------------------------------

# Shapes from the project's own cases; the files themselves stay on the owner's machine.
TARGET_CELL = (19.29742, 55.30813, 29.34936, 90.0, 107.1844, 90.0)
REFERENCE_CELL = (18.309, 39.389, 42.459, 90.0, 101.747, 90.0)


def test_niggli_reduction_preserves_volume():
    reduced = gemmi_provider.GemmiProvider().niggli_reduce(TARGET_CELL)
    assert reduced.volume == pytest.approx(29926.3, rel=1e-4)


def test_a_cell_is_the_same_lattice_as_itself():
    assert (
        gemmi_provider.compare_lattices(TARGET_CELL, TARGET_CELL).relation
        is LatticeRelation.SAME_LATTICE
    )


def test_an_alternative_setting_is_recognised_as_the_same_lattice():
    """beta -> 180 - beta is a change of basis, not a different crystal."""
    alternative = (19.29742, 55.30813, 29.34936, 90.0, 180 - 107.1844, 90.0)
    assert (
        gemmi_provider.compare_lattices(TARGET_CELL, alternative).relation
        is LatticeRelation.SAME_LATTICE
    )


def test_equal_volume_does_not_make_two_lattices_the_same():
    """The finding that motivated this module.

    The project's target and its reference structure have volumes 0.18 % apart and are different
    lattices. Comparing the six cell parameters, or comparing volumes, would have got this wrong in
    opposite directions.
    """
    comparison = gemmi_provider.compare_lattices(TARGET_CELL, REFERENCE_CELL)

    assert comparison.relation is LatticeRelation.DIFFERENT_LATTICE
    assert comparison.volume_agrees is True, "the volumes really do agree; that is the trap"
    assert comparison.max_length_deviation_a > 10.0
    assert comparison.max_angle_deviation_deg > 10.0
    assert "coincidence" in comparison.detail


def test_space_group_lookup_by_name():
    provider = gemmi_provider.GemmiProvider()
    info = provider.space_group_by_name("P 1 21 1")
    assert info is not None
    assert (info.number, info.is_sohncke, info.is_centrosymmetric) == (4, True, False)

    centro = provider.space_group_by_name("P 1 21/m 1")
    assert centro is not None
    assert (centro.number, centro.is_sohncke) == (11, False)


def test_an_unknown_symbol_returns_none_rather_than_a_guess():
    assert gemmi_provider.GemmiProvider().space_group_by_name("not a space group") is None


# --------------------------------------------------------------------------------------
# the cross-check: neither side silently wins
# --------------------------------------------------------------------------------------


def test_the_cross_check_reports_no_second_opinion_rather_than_inventing_one():
    """While IDE-FILL(gemmi-sg-from-ops) is unimplemented the provider cannot answer, and the
    cross-check must say so instead of falling back on the core's verdict twice."""
    check = gemmi_provider.cross_check_chirality(1, ("-x, y+1/2,-z",))
    assert check.core_says_sohncke is False
    assert check.provider_says_sohncke is None
    assert check.agree is None


def test_a_disagreement_is_stated_loudly_and_neither_side_wins():
    check = ChiralityCrossCheck(
        core_says_sohncke=True, provider_says_sohncke=False, provider_name="gemmi"
    )
    assert check.agree is False
    assert "DISAGREEMENT" in check.explain()
    assert "Do not proceed" in check.explain()


# --------------------------------------------------------------------------------------
# IDE-FILL specifications: written first, marked strict-xfail
# --------------------------------------------------------------------------------------


@pytest.mark.xfail(reason="IDE-FILL(gemmi-sg-from-ops)", strict=True, raises=NotImplementedError)
@pytest.mark.parametrize(
    "latt, symm, expected_number, expected_sohncke",
    [
        (-1, (), 1, True),  # P1
        (1, (), 2, False),  # P-1
        (-1, ("-x, y+1/2, -z",), 4, True),  # P2(1)
        (1, ("-x, y+1/2, -z",), 11, False),  # P2(1)/m  <- the target's declared group
        (7, ("-x, y, 1/2-z",), 15, False),  # C2/c
    ],
)
def test_space_group_from_shelx_operators(latt, symm, expected_number, expected_sohncke):
    info = gemmi_provider.GemmiProvider().space_group_from_operators(latt, symm)
    assert info is not None, "a tabulated group should have been identified"
    assert info.number == expected_number
    assert info.is_sohncke is expected_sohncke


@pytest.mark.xfail(reason="IDE-FILL(gemmi-sg-from-ops)", strict=True, raises=NotImplementedError)
def test_an_operator_set_matching_nothing_returns_none_not_the_nearest_group():
    """Rounding to a plausible group is the exact failure mode this project exists to prevent."""
    info = gemmi_provider.GemmiProvider().space_group_from_operators(-1, ("x+1/3, y+1/7, z",))
    assert info is None


@pytest.mark.xfail(
    reason="IDE-FILL(gemmi-unique-reflections)", strict=True, raises=NotImplementedError
)
def test_unique_reflection_count_against_a_hand_checkable_case():
    """A cubic P cell, a=10 A, to d_min = 5 A.

    In P1 the only symmetry is Friedel, so the unique count is half the lattice points inside the
    resolution sphere, minus the origin. Checked as an order-of-magnitude bound rather than an
    exact integer, because rounding at the sphere boundary is implementation-defined.
    """
    n = gemmi_provider.GemmiProvider().n_unique_reflections((10, 10, 10, 90, 90, 90), "P 1", 5.0)
    assert 15 <= n <= 25


@pytest.mark.xfail(
    reason="IDE-FILL(gemmi-unique-reflections)", strict=True, raises=NotImplementedError
)
def test_a_nonpositive_resolution_is_refused():
    with pytest.raises(ValueError):
        gemmi_provider.GemmiProvider().n_unique_reflections((10, 10, 10, 90, 90, 90), "P 1", 0.0)


# --------------------------------------------------------------------------------------
# the meta-test: no stub may ship without a specification
# --------------------------------------------------------------------------------------

_MARKER = re.compile(r"IDE-FILL\(([a-z0-9\-]+)\)")


def test_every_ide_fill_stub_has_a_test_that_specifies_it():
    """A stub with no test is a hole with no shape.

    Scans ``src/`` for ``IDE-FILL(id)`` markers and requires each id to appear in ``tests/``. This
    is what makes the handoff safe: the implementing agent always has an executable specification,
    and nothing can be quietly declared finished.
    """
    root = Path(__file__).resolve().parents[1]
    src, tests = root / "src", root / "tests"
    if not src.is_dir():  # pragma: no cover - installed-wheel run
        pytest.skip("running against an installed distribution, not the source tree")

    declared: dict[str, str] = {}
    for path in sorted(src.rglob("*.py")):
        for identifier in _MARKER.findall(path.read_text(encoding="utf-8")):
            declared.setdefault(identifier, str(path.relative_to(root)))

    specified = set()
    for path in sorted(tests.rglob("*.py")):
        specified.update(_MARKER.findall(path.read_text(encoding="utf-8")))

    unspecified = {i: where for i, where in declared.items() if i not in specified}
    assert unspecified == {}, (
        f"these IDE-FILL stubs have no test specifying them: {unspecified}. "
        "Write the test first; it is the handoff contract."
    )
