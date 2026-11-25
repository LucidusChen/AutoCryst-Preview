"""Ordering eligible candidates.

The ordering is a lexicographic key, not a weighted sum: R1, then wR2, then |GooF - 1|, then the
largest residual peak that is *not* adjacent to a heavy atom. There is no tunable weight to hide a
judgement in, and model confidence is absent on purpose -- it may order the search, never the result.

If nothing is eligible, the answer is :class:`NoTrustworthyCandidate`, which names the reason and the
next action. Reporting "no branch is trustworthy yet" is a correct outcome, not a failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from crystalx_core.domain.candidate import Candidate
from crystalx_core.domain.gates import EligibilityReport, RankingPolicy
from crystalx_core.domain.ids import CandidateId
from crystalx_core.domain.result import (
    NoTrustworthyCandidate,
    RankedCandidate,
    ReviewCheckpoint,
    ReviewCheckpointKind,
)

__all__ = ["RankingOutcome", "rank_candidates", "ORDERING_KEY_NAMES", "describe_ordering"]

#: The ordering, spelled out. Every key is "smaller is better".
ORDERING_KEY_NAMES: tuple[str, ...] = (
    "r1",
    "wr2",
    "goof_deviation",
    "max_residual_away_from_heavy",
)

_MISSING = math.inf
"""A missing ordering value sorts last. A candidate that cannot report R1 does not get to win by
default."""


@dataclass(frozen=True)
class RankingOutcome:
    """Ranked eligible candidates, the ineligible ones, and any review the run needs."""

    ranked: tuple[RankedCandidate, ...]
    ineligible: tuple[EligibilityReport, ...]
    review_checkpoints: tuple[ReviewCheckpoint, ...] = ()
    no_trustworthy_candidate: NoTrustworthyCandidate | None = None

    @property
    def best(self) -> RankedCandidate | None:
        return self.ranked[0] if self.ranked else None

    @property
    def has_winner(self) -> bool:
        return bool(self.ranked)


def rank_candidates(
    candidates: tuple[Candidate, ...],
    reports: tuple[EligibilityReport, ...],
    policy: RankingPolicy,
    *,
    heavy_elements: frozenset[str] = frozenset(),
) -> RankingOutcome:
    """Split candidates into eligible and not, then order the eligible ones."""
    by_id: dict[str, Candidate] = {str(c.candidate_id): c for c in candidates}
    eligible: list[tuple[tuple[float, ...], Candidate, EligibilityReport]] = []
    ineligible: list[EligibilityReport] = []
    checkpoints: list[ReviewCheckpoint] = []

    for report in reports:
        candidate = by_id.get(str(report.candidate_id))
        if candidate is None:
            continue
        if report.is_eligible:
            eligible.append((_sort_key(candidate, policy, heavy_elements), candidate, report))
        else:
            ineligible.append(report)
            if report.requires_review:
                checkpoints.append(_checkpoint_for(report))

    eligible.sort(key=lambda item: (item[0], str(item[1].candidate_id)))

    ranked = tuple(
        RankedCandidate(
            candidate_id=candidate.candidate_id,
            rank=index + 1,
            sort_key=key,
            key_names=ORDERING_KEY_NAMES,
            eligibility=report,
            summary=_summarise(candidate, key),
        )
        for index, (key, candidate, report) in enumerate(eligible)
    )

    none_trustworthy: NoTrustworthyCandidate | None = None
    if not ranked:
        none_trustworthy = NoTrustworthyCandidate(
            reason=(
                f"{len(ineligible)} candidate(s) were evaluated and none passed the eligibility "
                "gates."
                if ineligible
                else "No candidate has been produced yet."
            ),
            ineligible=tuple(ineligible),
            suggested_next_stage="shell_expansion",
            requires_human_expert=any(r.requires_review for r in ineligible),
        )

    return RankingOutcome(
        ranked=ranked,
        ineligible=tuple(ineligible),
        review_checkpoints=tuple(checkpoints),
        no_trustworthy_candidate=none_trustworthy,
    )


def _sort_key(
    candidate: Candidate, policy: RankingPolicy, heavy_elements: frozenset[str]
) -> tuple[float, ...]:
    metrics = candidate.metrics
    r1 = metrics.r1 if metrics and metrics.r1 is not None else _MISSING
    wr2 = metrics.wr2 if metrics and metrics.wr2 is not None else _MISSING
    goof_dev = (
        metrics.goof_deviation
        if metrics and metrics.goof_deviation is not None
        else _MISSING
    )
    residual = _max_residual_away_from_heavy(candidate, policy, heavy_elements)
    return (r1, wr2, goof_dev, residual)


def _max_residual_away_from_heavy(
    candidate: Candidate, policy: RankingPolicy, heavy_elements: frozenset[str]
) -> float:
    """Largest difference peak that is not sitting next to a heavy atom.

    A big peak beside a heavy atom is usually absorption or series-termination ripple. The same peak
    out in a void usually means the model is missing something, which is the case worth penalising.
    """
    metrics = candidate.metrics
    if metrics is None or metrics.residual_density is None:
        return _MISSING
    density = metrics.residual_density
    if density.peaks:
        offenders = density.significant_peaks_away_from_heavy(
            heavy_elements, threshold_e_per_a3=policy.residual_peak_threshold_e_per_a3
        )
        if offenders:
            return max(p.height_e_per_a3 for p in offenders)
        return 0.0
    if density.max_peak_e_per_a3 is not None:
        # No per-peak detail: fall back to the global maximum rather than claiming 0.
        return density.max_peak_e_per_a3
    return _MISSING


def _checkpoint_for(report: EligibilityReport) -> ReviewCheckpoint:
    reasons = [r for r in report.failures if r.routes_to_review]
    first = reasons[0] if reasons else report.failures[0]
    kind_map = {
        "g3_same_reflection_basis": ReviewCheckpointKind.DATA_QUALITY,
        "g5_coordination_plausible": ReviewCheckpointKind.COORDINATION_IMPLAUSIBLE,
        "g7_no_unresolved_severe_alert": ReviewCheckpointKind.SEVERE_VALIDATION_ALERT,
    }
    return ReviewCheckpoint(
        kind=kind_map.get(first.kind.value, ReviewCheckpointKind.DATA_QUALITY),
        question=(
            f"Candidate {report.candidate_id} failed {first.kind.value}: {first.message}. "
            "How should the workflow proceed?"
        ),
        candidates=(report.candidate_id,),
        options=(
            "acknowledge and continue with a recorded reason",
            "run another onion stage",
            "abandon this branch",
        ),
        recommendation="Inspect the evidence before acknowledging any level A alert.",
    )


def _summarise(candidate: Candidate, key: tuple[float, ...]) -> str:
    parts: list[str] = []
    for name, value in zip(ORDERING_KEY_NAMES, key):
        parts.append(f"{name}={'n/a' if value == _MISSING else f'{value:.4f}'}")
    metrics = candidate.metrics
    if metrics is not None:
        if metrics.n_restraints:
            parts.append(f"restraints={metrics.n_restraints}")
        if metrics.n_constraints:
            parts.append(f"constraints={metrics.n_constraints}")
        ratio = metrics.data_to_parameter_ratio
        if ratio is not None:
            parts.append(f"data/param={ratio:.1f}")
    return " ".join(parts)


def describe_ordering() -> str:
    """One line explaining the ordering, for the CLI and the run record."""
    return (
        "Eligible candidates are ordered lexicographically by "
        + " then ".join(ORDERING_KEY_NAMES)
        + ". There is no weighted score, and model confidence is not part of the key."
    )


def candidate_ids(outcome: RankingOutcome) -> tuple[CandidateId, ...]:
    return tuple(r.candidate_id for r in outcome.ranked)
