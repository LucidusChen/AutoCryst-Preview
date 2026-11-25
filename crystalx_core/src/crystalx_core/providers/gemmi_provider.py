"""gemmi adapter: Niggli reduction, space-group tables, completeness.

Chosen over cctbx for size and installability -- a few MB, a native arm64 wheel, maintained by
CCP4. It is still an optional extra, reached only through
:mod:`crystalx_core.domain.providers`, and the core neither imports nor requires it.

## API notes, verified against gemmi 0.7.5 rather than assumed

* Cell reduction is **not** ``UnitCell.find_reduced_cell()``; no such method exists. It is
  ``gemmi.GruberVector(cell, centring, track_change_of_basis)`` followed by ``.niggli_reduce()``,
  reading the result from ``.cell_parameters()``.
* ``is_sohncke()`` and ``is_centrosymmetric()`` live on ``gemmi.SpaceGroup``, not on the
  ``GroupOps`` object returned by ``.operations()``.

Both were checked by running them before this module was written. Anything below marked
``IDE-FILL`` has *not* been, and says so.

## What this provider is allowed to decide

It may **name** a space group, reduce a cell, and count reflections. It may not overrule
:mod:`crystalx_core.domain.symmetry` on whether a structure's declared group is Sohncke: that
verdict comes from the operators actually present in the user's file, and is exact without any
library. Where the two differ, :func:`cross_check_chirality` reports a disagreement and neither
side wins. See :class:`~crystalx_core.domain.providers.ChiralityCrossCheck`.
"""

from __future__ import annotations

from dataclasses import dataclass

from crystalx_core.domain.providers import (
    ChiralityCrossCheck,
    LatticeComparison,
    LatticeRelation,
    ReducedCell,
    SpaceGroupInfo,
    register_symmetry_provider,
)
from crystalx_core.domain.symmetry import ChiralityCompatibility, analyse_symmetry

__all__ = ["GemmiProvider", "register", "is_available", "compare_lattices", "cross_check_chirality"]

#: Cell lengths are called equal below this, in angstroms. Reduced cells of the same lattice agree
#: to within experimental error; a real difference is orders of magnitude larger than this.
LENGTH_TOLERANCE_A = 0.05

#: Cell angles are called equal below this, in degrees.
ANGLE_TOLERANCE_DEG = 0.5


def is_available() -> bool:
    """Whether gemmi can be imported. Never raises."""
    try:
        import gemmi  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class GemmiProvider:
    """Implements :class:`~crystalx_core.domain.providers.SymmetryProvider` and ``CellReducer``."""

    name: str = "gemmi"

    def __post_init__(self) -> None:
        if not is_available():
            raise ImportError(
                "gemmi is not installed. Install the optional extra:  pip install -e '.[geo]'  "
                "The core does not require it; without it, cell reduction, space-group naming and "
                "completeness are reported as unavailable rather than estimated."
            )

    # ------------------------------------------------------------------ cell reduction
    def niggli_reduce(
        self, cell: tuple[float, float, float, float, float, float], *, centring: str = "P"
    ) -> ReducedCell:
        import gemmi

        unit_cell = gemmi.UnitCell(*cell)
        gruber = gemmi.GruberVector(unit_cell, centring, True)
        gruber.niggli_reduce()
        if not gruber.is_niggli():
            raise ValueError(
                f"Niggli reduction did not converge for cell {cell}. The cell is probably not a "
                "valid lattice; check the indexing before comparing it with anything."
            )
        a, b, c, alpha, beta, gamma = gruber.cell_parameters()
        return ReducedCell(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            volume=gemmi.UnitCell(a, b, c, alpha, beta, gamma).volume,
            change_of_basis=str(gruber.change_of_basis),
        )

    # ------------------------------------------------------------------ space groups
    def space_group_by_name(self, symbol: str) -> SpaceGroupInfo | None:
        import gemmi

        try:
            sg = gemmi.SpaceGroup(symbol)
        except (ValueError, RuntimeError):
            return None
        return _to_info(sg)

    def space_group_from_operators(
        self, latt: int | None, symm: tuple[str, ...]
    ) -> SpaceGroupInfo | None:
        """Identify the tabulated group whose operator set matches these SHELX records.

        IDE-FILL(gemmi-sg-from-ops): build a ``gemmi.GroupOps`` from the SHELX records and look up
        the matching space group.

        Contract:
          * Expand ``symm`` into ``gemmi.Op`` objects; SHELX omits the identity, so add ``x,y,z``.
          * A **positive** ``latt`` adds the inversion centre ``-x,-y,-z``; a negative one does not.
            ``abs(latt)`` selects the centring: 1=P 2=I 3=R 4=F 5=A 6=B 7=C. Add the corresponding
            centring translations (``gemmi.GroupOps.add_centring_vectors`` or by hand).
          * Find the tabulated group via ``gemmi.find_spacegroup_by_ops(ops)``.
          * Return ``None`` when no group matches. **Do not return the nearest match** -- an
            operator set that matches nothing is a real finding about the file, and rounding it to
            a plausible group is precisely the class of silent error this project exists to avoid.

        Verify against: P1 (latt -1, no symm) -> #1; P2(1)/m (latt 1, ``-x,y+1/2,-z``) -> #11;
        P2(1) (latt -1, same operator) -> #4; C2/c (latt 7, ``-x,y,1/2-z``) -> #15.
        The test in ``tests/test_providers.py`` already encodes these and is marked ``xfail``;
        when your implementation passes, the strict xfail turns into a failure telling you to
        delete the marker.
        """
        raise NotImplementedError("IDE-FILL(gemmi-sg-from-ops)")

    # ------------------------------------------------------------------ completeness
    def n_unique_reflections(
        self,
        cell: tuple[float, float, float, float, float, float],
        space_group: str,
        d_min: float,
    ) -> int:
        """Count the unique reflections symmetry allows out to ``d_min``.

        IDE-FILL(gemmi-unique-reflections): implement with
        ``gemmi.ReciprocalAsu`` / ``gemmi.make_miller_array`` or by generating the hkl range from
        the cell and reducing each to the asymmetric unit.

        Contract:
          * Count **unique** reflections in the reciprocal asymmetric unit, out to and including
            ``d_min``.
          * Exclude systematic absences (``GroupOps.is_systematically_absent``).
          * Exclude ``(0,0,0)``.
          * Raise ``ValueError`` for ``d_min <= 0`` rather than returning 0.

        This is the denominator of completeness, which the core currently reports as unavailable.
        Getting it wrong inflates or deflates a number the crystallographer will trust, so the test
        checks it against a hand-computable case (a cubic P cell) before anything else uses it.
        """
        raise NotImplementedError("IDE-FILL(gemmi-unique-reflections)")


def _to_info(sg) -> SpaceGroupInfo:
    return SpaceGroupInfo(
        number=sg.number,
        hermann_mauguin=sg.hm,
        is_sohncke=sg.is_sohncke(),
        is_centrosymmetric=sg.is_centrosymmetric(),
        laue_class=sg.laue_str(),
        crystal_system=sg.crystal_system_str(),
        n_operations=len(sg.operations()),
    )


def register() -> GemmiProvider:
    """Install the provider. Raises ``ImportError`` with an actionable message if gemmi is absent."""
    provider = GemmiProvider()
    register_symmetry_provider(provider)
    return provider


# --------------------------------------------------------------------------------------
# the two things the workflow actually asks for
# --------------------------------------------------------------------------------------


def compare_lattices(
    target: tuple[float, float, float, float, float, float] | None,
    reference: tuple[float, float, float, float, float, float] | None,
    *,
    target_centring: str = "P",
    reference_centring: str = "P",
    provider: GemmiProvider | None = None,
) -> LatticeComparison:
    """Are these two cells the same lattice?

    **Equal volume does not make two lattices the same**, and this function exists because that
    mistake was about to be made. The project's own target and its reference structure have
    volumes 29,926 and 29,979 cubic angstroms -- 0.18 % apart -- and are nonetheless different
    lattices: after Niggli reduction their b and c axes differ by 10 and 13 angstroms and two
    angles differ by more than 11 degrees. A coincidence of volume is chemically unremarkable when
    one salt swaps a small anion for a large one and loses some solvent.

    The consequence is concrete: the reference is a **topology** reference. It cannot seed
    isomorphous replacement or supply a starting model by re-indexing.
    """
    if target is None or reference is None:
        return LatticeComparison(
            relation=LatticeRelation.UNDECIDABLE,
            detail="One of the two cells is missing.",
        )
    reducer = provider or (GemmiProvider() if is_available() else None)
    if reducer is None:
        return LatticeComparison(
            relation=LatticeRelation.UNDECIDABLE,
            detail=(
                "No cell reducer is registered, so these cells can only be compared parameter by "
                "parameter, which is not a lattice comparison. Install the optional extra: "
                "pip install -e '.[geo]'"
            ),
        )

    reduced_target = reducer.niggli_reduce(target, centring=target_centring)
    reduced_reference = reducer.niggli_reduce(reference, centring=reference_centring)

    lengths = [
        abs(x - y) for x, y in zip(reduced_target.parameters[:3], reduced_reference.parameters[:3])
    ]
    angles = [
        abs(x - y) for x, y in zip(reduced_target.parameters[3:], reduced_reference.parameters[3:])
    ]
    same = max(lengths) <= LENGTH_TOLERANCE_A and max(angles) <= ANGLE_TOLERANCE_DEG
    ratio = reduced_target.volume / reduced_reference.volume if reduced_reference.volume else None

    if same:
        detail = (
            "The Niggli-reduced cells agree, so these are the same lattice in different settings."
        )
    else:
        detail = (
            "The Niggli-reduced cells differ, so these are different lattices — whatever their "
            f"volumes do. Largest length difference {max(lengths):.3f} A, largest angle difference "
            f"{max(angles):.3f} deg."
        )
        if ratio is not None and abs(ratio - 1.0) <= 0.02:
            detail += (
                f" Note that the volumes agree to {abs(ratio - 1.0) * 100:.2f} %, which is a "
                "coincidence and not evidence of a shared lattice."
            )

    return LatticeComparison(
        relation=LatticeRelation.SAME_LATTICE if same else LatticeRelation.DIFFERENT_LATTICE,
        target=reduced_target,
        reference=reduced_reference,
        volume_ratio=ratio,
        max_length_deviation_a=max(lengths),
        max_angle_deviation_deg=max(angles),
        detail=detail,
    )


def cross_check_chirality(
    latt: int | None, symm: tuple[str, ...], *, provider: GemmiProvider | None = None
) -> ChiralityCrossCheck:
    """Compare the core's operator analysis with gemmi's tables.

    Neither overrules the other. The core reads what is in the file; gemmi looks up what the file
    ought to be. Agreement raises confidence; disagreement means one of them is misreading the
    file, and that is worth a human's attention rather than an automatic tie-break.
    """
    core = analyse_symmetry(latt, symm)
    core_verdict: bool | None
    if core.compatibility is ChiralityCompatibility.SOHNCKE:
        core_verdict = True
    elif core.compatibility is ChiralityCompatibility.CONTAINS_IMPROPER:
        core_verdict = False
    else:
        core_verdict = None

    if provider is None and not is_available():
        return ChiralityCrossCheck(core_says_sohncke=core_verdict, provider_says_sohncke=None)

    active = provider or GemmiProvider()
    try:
        info = active.space_group_from_operators(latt, symm)
    except NotImplementedError:
        # IDE-FILL(gemmi-sg-from-ops) is not done yet. Report honestly rather than pretending.
        return ChiralityCrossCheck(
            core_says_sohncke=core_verdict,
            provider_says_sohncke=None,
            provider_name=active.name,
        )
    if info is None:
        return ChiralityCrossCheck(
            core_says_sohncke=core_verdict,
            provider_says_sohncke=None,
            provider_name=active.name,
        )
    return ChiralityCrossCheck(
        core_says_sohncke=core_verdict,
        provider_says_sohncke=info.is_sohncke,
        provider_name=active.name,
        space_group=info,
    )
