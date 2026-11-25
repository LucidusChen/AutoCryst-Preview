"""Iterative reconstruction: the Web adapter over ``ReconstructionBackendService``.

Initial route (SHELXT, protein C/D/E or small-molecule SHELXS/SHELXD) → pretrained CrystalX element
assignment → repeated SHELXL rounds, with local checkCIF/PLATON offered separately afterwards.

This module holds no solver logic and makes no numerical decision. It

* turns form values into the backend's typed request; the backend validates every value and its
  message is shown verbatim;
* lists the request and policy fields by reading the backend's own dataclasses, so a new backend
  field shows up here (and fails the label test until it is described);
* renders the backend's own state and summary records; a missing value stays visibly missing;
* reads candidate files only through the externally pinned seal (``<run_id>.seal-pin.json``),
  never through a path taken from a summary.

A numerical stop is never presented as a scientific result: stop codes are shown verbatim with a
short explanation, and nothing here says converged, validated or correct.
"""

from __future__ import annotations

import dataclasses
import html
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from crystalx_core.app.inference_protocols import INPUT_PROTOCOLS
from crystalx_core.app.reconstruction_backend import (
    FRONTEND_CONTRACT, ReconstructionBackendService, ReconstructionRequest,
)
from crystalx_core.app.solve_routes import InitialRouteConfig, ProteinPhasingConfig
from crystalx_core.private_acceptance.sealing import SealError, read_sealed_output, verify_seal
from crystalx_core.refinement.adaptive import AdaptivePolicy
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_web.reconstruction_text import decision_label, field_label, rt, stop_explanation

__all__ = [
    "FieldSpec", "FormError", "TERMINAL", "policy_fields", "cycle_groups", "parse_outer_rounds",
    "format_iteration_delta", "build_request", "preview_md", "status_md", "history_rows", "summary_md",
    "load_terminal_summary", "list_runs", "pinned_seal", "export_candidate", "viewer_payload_for",
    "nonpositive_u_warnings", "run_checkcif", "present_local_alerts", "DATASET_ROLES", "DATASET_FORMATS",
    "EXPECTED_CONTRACT", "contract_note", "demo_inputs", "MODES",
]

EXPECTED_CONTRACT = "2026-11-20"
"""The backend contract this page was written against (``FRONTEND_CONTRACT``). A different backend
value is shown on the page, and fails the tests until the page has been reviewed against it."""

MODES = ("real", "mock_demo")

TERMINAL = frozenset({"completed", "blocked", "failed", "cancelled", "interrupted"})
"""Phases after which the worker writes nothing more. ``interrupted`` is reported by the backend when
the owning process is gone; no automatic replay exists."""

DATASET_ROLES = ("SAD", "PEAK", "INFL", "HREM", "LREM", "NAT")
DATASET_FORMATS = ("SHELX_INTENSITIES", "SHELX_AMPLITUDES", "XDS_ASCII", "SCALEPACK", "MTZ")
CANDIDATE_ROLES = (("model", "ai_solved.res"), ("cif", "ai_solved.cif"), ("fcf", "ai_solved.fcf"))

# String fields whose admissible values the backend checks in __post_init__ without exporting a
# constant. tests/test_reconstruction_adapter.py proves each value is accepted and others refused.
CHOICES: dict[str, tuple[str, ...]] = {
    "heavy_preparation": ("current_loop", "upstream_heavy_v1"),
    "cycle.inference_protocol": tuple(INPUT_PROTOCOLS),
    "cycle.adaptive.prune_evidence_mode": ("any", "high_u_and_isolated"),
    "cycle.refinement_method": ("ls", "cgls"),
    "cycle.rejection_budget_scope": ("base_model_streak", "run_total"),
    "cycle.anchor_uiso_policy": ("unlock_or_block", "block_any", "report_only"),
    "cycle.stopping_policy": ("movable_and_growth", "legacy_important"),
    "cycle.adaptive.close_survivor_policy": ("lower_u", "reference_consistency"),
}

# Layout only: which CyclePolicy fields sit in which panel. A field missing from every group is
# still shown (in "other"), so a new backend field is never silently dropped from the form.
_CYCLE_GROUPS: dict[str, tuple[str, ...]] = {
    "rounds": ("iterations", "max_rounds", "shelxl_cycles", "stable_transitions",
               "coordinate_tolerance_angstrom", "r1_tolerance", "important_atom_ids",
               "max_wall_seconds", "trial_timeout_s"),
    "protocol": ("inference_protocol", "inference_seed", "stopping_policy",
                 "require_no_element_changes_for_stop", "record_diagnostics"),
    "safeguards": ("rejection_backoff", "rejection_budget_scope", "backoff_failure_budget",
                   "rejected_action_ttl_rounds", "anchor_uiso_guard", "anchor_uiso_policy",
                   "anchor_min_supported", "anchor_uiso_cap"),
    "refinement": ("refinement_method", "early_cgls_rounds", "damp_factor", "damp_shift_esd_limit",
                   "refinement_displacement_threshold_angstrom", "max_refinement_net_shift_angstrom", "anisotropic_metals"),
    "peaks": ("max_new_peaks_per_round", "minimum_peak_height", "minimum_peak_distance_angstrom",
              "minimum_assignment_probability", "max_r1_regression", "max_failures", "plan_peaks", "fourier_map", "metal_ripple_exclusion_angstrom"),
}

_KINDS = {"bool": "bool", "int": "int", "float": "float", "int|None": "optional_int",
          "float|None": "optional_float", "str": "text", "tuple[str,...]": "ids"}


@dataclass(frozen=True)
class FieldSpec:
    path: str
    """Dotted record path, e.g. ``cycle.max_rounds`` or ``cycle.adaptive.u_abs_high``."""
    kind: str
    """bool, int, float, optional_int, ids, text or choice; ``unsupported`` fails the tests."""
    default: Any
    choices: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.path.rsplit(".", 1)[-1]


class FormError(ValueError):
    """One or more messages for the person filling in the form (backend messages verbatim)."""

    def __init__(self, messages: Iterable[str]) -> None:
        self.messages = [str(m) for m in messages]
        super().__init__("; ".join(self.messages))


def _specs(cls: type, prefix: str, skip: tuple[str, ...] = ()) -> list[FieldSpec]:
    specs = []
    for item in dataclasses.fields(cls):
        if item.name in skip:
            continue
        path = f"{prefix}.{item.name}"
        default = item.default if item.default is not dataclasses.MISSING else item.default_factory()
        kind = _KINDS.get(str(item.type).replace(" ", ""), "unsupported")
        choices = CHOICES.get(path, ())
        if choices:
            kind = "choice"
        specs.append(FieldSpec(path, kind, default, choices))
    return specs


def policy_fields() -> dict[str, list[FieldSpec]]:
    """Every CyclePolicy and AdaptivePolicy field, read from the backend's dataclasses."""
    return {"cycle": _specs(CyclePolicy, "cycle", skip=("adaptive",)),
            "adaptive": _specs(AdaptivePolicy, "cycle.adaptive")}


def cycle_groups() -> dict[str, list[FieldSpec]]:
    """CyclePolicy fields by panel; ``adaptive`` holds every AdaptivePolicy field."""
    fields = policy_fields()
    by_name = {spec.name: spec for spec in fields["cycle"]}
    groups = {group: [by_name[n] for n in names if n in by_name] for group, names in _CYCLE_GROUPS.items()}
    placed = {n for names in _CYCLE_GROUPS.values() for n in names}
    groups["other"] = [spec for spec in fields["cycle"] if spec.name not in placed]
    groups["adaptive"] = fields["adaptive"]
    return groups


# ---------------------------------------------------------------------------- form values
def parse_outer_rounds(text: Any) -> int | None:
    """Blank → ``None`` (automatic numerical stopping); a positive whole number → that many rounds.

    Decimals, zero, negative values and words are refused. SHELXL cycles per round are separate.
    """
    if text is None:
        return None
    if isinstance(text, bool):
        raise ValueError("a whole number of rounds")
    if isinstance(text, (int, float)):
        if not math.isfinite(text) or text != int(text) or text < 1:
            raise ValueError("a whole number of rounds ≥ 1")
        return int(text)
    value = str(text).strip()
    if not value:
        return None
    if not re.fullmatch(r"\d+", value) or int(value) < 1:
        raise ValueError("a whole number of rounds ≥ 1")
    return int(value)


def _symbols(text: Any) -> list[str]:
    """Element symbols as typed, with only their capitalisation normalised (zn → Zn)."""
    parts = [p for p in re.split(r"[,\s，、;；]+", str(text or "")) if p]
    return [p[:1].upper() + p[1:].lower() for p in parts]


def _ids(text: Any) -> list[str]:
    """Atom labels exactly as typed (labels are case-sensitive)."""
    if isinstance(text, (list, tuple)):
        return [str(x) for x in text]
    return [p for p in re.split(r"[,\s，、;；]+", str(text or "")) if p]


def _whole(value: Any, label: str, *, optional: bool = False) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if optional:
            return None
        raise ValueError(f"{label}: a whole number is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: a whole number is required") from None
    if isinstance(value, bool) or not math.isfinite(number) or number != int(number):
        raise ValueError(f"{label}: a whole number is required")
    return int(number)


def _real(value: Any, label: str, *, optional: bool = False) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if optional:
            return None
        raise ValueError(f"{label}: a number is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: a number is required") from None
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{label}: a finite number is required")
    return number


def _field_value(spec: FieldSpec, value: Any) -> Any:
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "int":
        return _whole(value, spec.path)
    if spec.kind == "float":
        return _real(value, spec.path)
    if spec.kind == "optional_int":
        try:
            return parse_outer_rounds(value)
        except ValueError as exc:
            raise ValueError(f"{spec.path}: {exc}") from None
    if spec.kind == "optional_float":  # blank keeps the backend's "not set" (None)
        return _real(value, spec.path, optional=True)
    if spec.kind == "ids":
        return _ids(value)
    if spec.kind in ("choice", "text"):
        return "" if value is None else str(value)
    raise ValueError(f"{spec.path}: this backend field type is not supported by the form yet")


def _file(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    return _canonical_parent(str(value)) if value else None


def _canonical_parent(path: str) -> str:
    """The same file, named through its real folder.

    The backend refuses an input whose folder is reached through an alias (on macOS the temporary
    folder ``/var/...`` is an alias of ``/private/var/...``). Only the folder is resolved here: the
    file name is kept as given, so the backend still sees, and refuses, a file that is itself a link.
    """
    given = Path(path)
    try:
        return str(given.parent.resolve(strict=True) / given.name)
    except (OSError, RuntimeError):
        return path


def _route_record(form: Mapping[str, Any], problems: list[str]) -> dict[str, Any]:
    strategy = str(form.get("route.strategy") or "shelxt")
    record: dict[str, Any] = {"strategy": strategy}
    try:
        record["timeout_s"] = _real(form.get("route.timeout_s", InitialRouteConfig.timeout_s), "route.timeout_s")
        if strategy == "small_molecule":
            record["small_solver"] = str(form.get("route.small_solver") or "shelxs")
            record["small_method"] = str(form.get("route.small_method") or "direct")
            record["trials"] = _whole(form.get("route.trials"), "route.trials")
            record["find_sites"] = _whole(form.get("route.find_sites"), "route.find_sites", optional=True)
            record["plop"] = [_whole(p, "route.plop") for p in _ids(form.get("route.plop"))]
        elif strategy == "protein_cde":
            record["trials"] = _whole(form.get("route.trials"), "route.trials")
            record["protein"] = _protein_record(form)
    except ValueError as exc:
        problems.append(str(exc))
    return record


def _protein_record(form: Mapping[str, Any]) -> dict[str, Any]:
    datasets = []
    for index in range(len(DATASET_ROLES)):
        path = _file(form.get(f"protein.dataset.{index}.path"))
        role = form.get(f"protein.dataset.{index}.role")
        if not path and not role:
            continue
        if not path or not role:
            raise ValueError(f"protein dataset {index + 1}: choose both a role and a file")
        datasets.append({"role": str(role), "path": path,
                         "format": str(form.get(f"protein.dataset.{index}.format") or "SHELX_INTENSITIES"),
                         "friedel_pairs_retained": bool(form.get(f"protein.dataset.{index}.friedel"))})
    heavy = _symbols(form.get("protein.heavy_element"))
    if len(heavy) > 1:
        raise ValueError("protein.heavy_element: one element symbol")
    return {"protein_confirmed": bool(form.get("protein.protein_confirmed")),
            "experiment": str(form.get("protein.experiment") or "SAD"),
            "space_group": str(form.get("protein.space_group") or "").strip(),
            "find_sites": _whole(form.get("protein.find_sites"), "protein.find_sites", optional=True),
            "heavy_element": heavy[0] if heavy else "",
            "solvent_fraction": _real(form.get("protein.solvent_fraction"), "protein.solvent_fraction", optional=True),
            "datasets": datasets,
            "trace_cycles": _whole(form.get("protein.trace_cycles", ProteinPhasingConfig.trace_cycles), "protein.trace_cycles"),
            "density_cycles": _whole(form.get("protein.density_cycles", ProteinPhasingConfig.density_cycles),
                                     "protein.density_cycles"),
            "heavy_atoms_in_native": bool(form.get("protein.heavy_atoms_in_native", True)),
            "preferred_hand": str(form.get("protein.preferred_hand") or "auto")}


def request_record(form: Mapping[str, Any], *, mode: str = "real",
                   tool_paths: Iterable[tuple[str, Any]] = ()) -> dict[str, Any]:
    """The JSON record for ``ReconstructionRequest.from_record``. No expert or reference field exists."""
    problems: list[str] = []
    ins, hkl = _file(form.get("ins_path")), _file(form.get("hkl_path"))
    if ins is None:
        problems.append("ins_path: choose the original instruction (.ins) file")
    if hkl is None:
        problems.append("hkl_path: choose the measured reflection (.hkl) file")
    cycle: dict[str, Any] = {}
    adaptive: dict[str, Any] = {}
    fields = policy_fields()
    for spec in fields["cycle"]:
        if spec.path in form:
            try:
                cycle[spec.name] = _field_value(spec, form[spec.path])
            except ValueError as exc:
                problems.append(str(exc))
    for spec in fields["adaptive"]:
        if spec.path in form:
            try:
                adaptive[spec.name] = _field_value(spec, form[spec.path])
            except ValueError as exc:
                problems.append(str(exc))
    if adaptive:
        cycle["adaptive"] = adaptive
    record: dict[str, Any] = {
        "ins_path": ins or "", "hkl_path": hkl or "",
        "route": _route_record(form, problems),
        "cycle": cycle,
        "metal_only": bool(form.get("metal_only", False)),
        "anchor_elements": _symbols(form.get("anchor_elements")),
        "candidate_elements": _symbols(form.get("candidate_elements")),
        "candidate_element_source": str(form.get("candidate_element_source") or "original_ins"),
        "mode": mode,
        "strict_native": bool(form.get("strict_native", True)),
        "tool_paths": [[str(name), str(path)] for name, path in tool_paths],
        "device": str(form.get("device") or "auto"),
        "heavy_preparation": str(form.get("heavy_preparation") or "current_loop"),
        "label": str(form.get("label") or "").strip() or (Path(ins).stem if ins else "reconstruction"),
    }
    if problems:
        raise FormError(problems)
    return record


def build_request(form: Mapping[str, Any], *, mode: str = "real",
                  tool_paths: Iterable[tuple[str, Any]] = ()) -> ReconstructionRequest:
    """The backend's typed request. Its own validation messages are passed on verbatim."""
    record = request_record(form, mode=mode, tool_paths=tool_paths)
    try:
        return ReconstructionRequest.from_record(record)
    except (ValueError, TypeError) as exc:
        raise FormError([f"{type(exc).__name__}: {exc}"]) from None


# ---------------------------------------------------------------------------- presentation
def _code(value: Any) -> str:
    return "`" + html.escape(str(value)).replace("`", "'") + "`"


_TECHNICAL_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?::[A-Za-z0-9_.-]+)?")
_DETAIL_LIMIT = 600


def _detail_lines(detail: Any, lang: str) -> list[str]:
    """The backend's ``stop_detail`` verbatim (bounded), plus what any technical identifier in it means."""
    if detail is None or str(detail).strip() == "":
        return []
    text = " ".join(str(detail).split())
    shown = text if len(text) <= _DETAIL_LIMIT else text[:_DETAIL_LIMIT] + " …"
    lines = [rt("stop_detail_line", lang, detail=_code(shown))]
    explained: list[str] = []
    for token in _TECHNICAL_ID.findall(text):
        meaning = stop_explanation(token, lang)
        if meaning and token not in explained:
            explained.append(token)
            lines.append(f"- {_code(token)} — {meaning}")
        if len(explained) >= 5:
            break
    return ["\n".join(lines)]


def contract_note(lang: str) -> str:
    """Empty when the backend speaks the contract this page was written against."""
    if FRONTEND_CONTRACT == EXPECTED_CONTRACT:
        return ""
    return rt("contract_mismatch", lang, backend=_code(FRONTEND_CONTRACT), page=_code(EXPECTED_CONTRACT))


def demo_inputs(state_dir: Path) -> tuple[Path, Path]:
    """The built-in synthetic INS/HKL pair (a fictional structure), written below the page's state folder."""
    from crystalx_core.app.demo import write_demo_case

    return write_demo_case(Path(state_dir) / "reconstruction_demo_input")


def _codes_md(codes: Iterable[Any], lang: str) -> str:
    lines = []
    for code in codes:
        text = stop_explanation(str(code), lang)
        lines.append(f"- {_code(code)}" + (f" — {text}" if text else ""))
    return "\n".join(lines)


def format_iteration_delta(value: Any, *, digits: int = 4, unit: str = "", lang: str = "zh") -> str:
    """A per-round change. ``None`` (no preceding refined round) is shown as unavailable, never 0."""
    if value is None:
        return rt("delta_unavailable", lang)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return html.escape(str(value))
    return f"{number:.{digits}f}{unit}"


def device_lines(record: Mapping[str, Any] | None, lang: str) -> list[str]:
    """Requested and run-pinned inference device, the backend's reason, and any device blocker.

    Records written before the 2026-11-20 contract carry none of these fields; nothing is shown."""
    record = record or {}
    requested, used = record.get("device_requested"), record.get("device_used")
    reason, blocker = record.get("device_selection_reason"), record.get("device_blocker")
    if requested is None and used is None and not blocker:
        return []
    lines = [rt("device_line", lang, requested=_code(requested or "—"), used=_code(used or "—"),
                reason=html.escape(str(reason)) if reason else "—")]
    if blocker:
        meaning = stop_explanation(str(blocker), lang)
        lines.append(rt("device_blocked_line", lang, code=_code(blocker)) + (f" — {meaning}" if meaning else ""))
    return lines


def stage_lines(stages: Iterable[Mapping[str, Any]] | None, lang: str) -> list[str]:
    """Fixed (non-iterative) stages such as the initial heavy stage, with their own fit numbers."""
    lines = []
    for stage in stages or []:
        if not isinstance(stage, Mapping):
            continue
        lines.append(rt("stage_line", lang, stage=_code(stage.get("stage", "?")),
                        cycles=stage.get("shelxl_cycles", "—"),
                        aniso=rt("yes", lang) if stage.get("anisotropic") else rt("no", lang),
                        r1=format_iteration_delta(stage.get("r1"), lang=lang) if stage.get("r1") is not None else "—",
                        wr2=format_iteration_delta(stage.get("wr2"), lang=lang) if stage.get("wr2") is not None else "—",
                        goof=format_iteration_delta(stage.get("goof"), digits=3, lang=lang) if stage.get("goof") is not None else "—",
                        params="—" if stage.get("parameters") is None else str(stage.get("parameters"))))
        if stage.get("stage") == "upstream_heavy_v1":
            lines.append(rt("heavy_stage_note", lang))
    return ([rt("stages_head", lang)] + lines) if lines else []


def preview_md(preview: Mapping[str, Any] | None, lang: str) -> str:
    if not preview:
        return rt("preview_hint", lang)
    route = preview.get("route") or {}
    strategy = (route.get("strategy") or {}).get("strategy", "?")
    lines = [rt("preview_title", lang), "", rt("preview_strategy", lang, strategy=_code(strategy))]
    if preview.get("heavy_preparation") == "upstream_heavy_v1":
        lines.append(rt("preview_heavy_stage", lang))
    lines += device_lines(preview, lang)
    steps = [step.get("tool") for step in route.get("steps", []) if isinstance(step, dict)]
    if steps:
        lines.append(rt("preview_steps", lang, steps=" → ".join(_code(s) for s in steps)))
    tools = preview.get("tools") or []
    if tools:
        lines += ["", rt("preview_tools_head", lang), "", "| tool | static | diagnostic | message | path |",
                  "| --- | --- | --- | --- | --- |"]
        for tool in tools:
            ok = "✅" if tool.get("usable_static") else "⛔"
            lines.append(f"| {_code(tool.get('tool'))} | {ok} | {_code(tool.get('diagnostic'))} | "
                         f"{html.escape(str(tool.get('message') or ''))} | {html.escape(str(tool.get('path') or '—'))} |")
    blockers = preview.get("blockers") or []
    warnings = route.get("warnings") or []
    lines.append("")
    if blockers:
        lines += [rt("preview_blocked", lang), _codes_md(blockers, lang)]
    else:
        lines.append(rt("preview_ready_static", lang))
    if warnings:
        lines += ["", rt("preview_warnings", lang), _codes_md(warnings, lang)]
    lines += ["", rt("preview_scope", lang)]
    return "\n".join(lines)


def _phase(phase: Any, lang: str) -> str:
    return rt(f"phase_{phase}", lang) if phase in {"queued", "running", *TERMINAL} else html.escape(str(phase))


def status_md(state: Mapping[str, Any] | None, lang: str) -> str:
    if not state:
        return rt("no_recon_run", lang)
    lines = [rt("status_line", lang, run_id=_code(state.get("run_id", "?")), phase=_phase(state.get("phase"), lang),
                round=state.get("round", 0), inference=state.get("inference_calls", 0),
                tools=state.get("tool_calls", 0), stable=state.get("stable_transitions", 0))]
    lines += device_lines(state, lang)
    stop = state.get("stop_code")
    if stop:
        lines.append(rt("stop_line", lang, code=_code(stop), meaning=stop_explanation(str(stop), lang) or "—"))
    lines += _detail_lines(state.get("stop_detail"), lang)
    lines += stage_lines(state.get("stages"), lang)
    if state.get("message") and state.get("message") != stop:
        lines.append(rt("message_line", lang, message=html.escape(str(state["message"]))))
    if state.get("blockers"):
        lines += [rt("blockers_head", lang), _codes_md(state["blockers"], lang)]
    if state.get("phase") == "interrupted":
        lines.append(rt("interrupted_note", lang))
    lines.append(rt("not_validated_note", lang))
    return "\n\n".join(lines)


def _count(value: Any) -> str:
    """A per-round count; absent in records from before the 2026-11-20 contract (shown as —, never 0)."""
    return "—" if value is None else str(value)


def _main_rejection(counts: Any) -> str:
    """The most frequent peak-rejection reason of the round, as ``reason ×n`` (verbatim backend identifier)."""
    if not isinstance(counts, Mapping) or not counts:
        return "—"
    reason, n = max(((str(k), v) for k, v in counts.items() if isinstance(v, int)), key=lambda kv: (kv[1], kv[0]),
                    default=("", 0))
    return f"{reason} ×{n}" if reason else "—"


def history_rows(history: Iterable[Mapping[str, Any]] | None, lang: str) -> list[list[str]]:
    rows = []
    for entry in history or []:
        added = entry.get("accepted_new_sites_if_retained")
        deleted = entry.get("deleted_site_ids_if_retained") or ()
        stable = entry.get("stable_transition")
        rows.append([
            str(entry.get("round", "")),
            format_iteration_delta(entry.get("r1"), lang=lang),
            format_iteration_delta(entry.get("r1_change"), lang=lang),
            format_iteration_delta(entry.get("important_coordinate_max_change_angstrom"), digits=3, lang=lang),
            "—" if added is None else str(added),
            str(len(deleted)),
            "—" if entry.get("element_changes") is None else str(entry.get("element_changes")),
            rt("yes", lang) if stable else rt("no", lang),
            decision_label(str(entry.get("decision", "")), lang),
            str(entry.get("inference_protocol", "")),
            "—" if entry.get("backoff_level") is None else str(entry.get("backoff_level")),
            str(entry.get("refinement_method") or "—"),
            format_iteration_delta(entry.get("refinement_max_displacement_angstrom"), digits=2, lang=lang)
            if entry.get("refinement_max_displacement_angstrom") is not None else "—",
            "—" if entry.get("refinement_displacement_above_threshold_count") is None
            else str(entry.get("refinement_displacement_above_threshold_count")),
            _count(entry.get("listed_output_peak_count")),
            _count(entry.get("accepted_new_site_count")),
            _main_rejection(entry.get("peak_rejection_reason_counts")),
        ])
    return rows


def _candidate_label(summary: Mapping[str, Any], lang: str) -> str:
    """Named after the backend's stop category; records from before the contract fall back to the code."""
    category = summary.get("stop_category")
    stop = summary.get("stop_code")
    if category == "fixed_rounds" or (category is None and stop == "fixed_rounds_completed"):
        return rt("label_fixed_rounds", lang)
    if category == "numerical_stop" or (category is None and summary.get("phase") == "completed"):
        return rt("label_numerically_stable", lang)
    return rt("label_unfinished", lang)


def summary_md(summary: Mapping[str, Any] | None, lang: str, *, warnings: Mapping[int, list[str]] | None = None) -> str:
    if not summary:
        return rt("summary_pending", lang)
    lines = [rt("summary_title", lang), "",
             rt("summary_phase", lang, phase=_phase(summary.get("phase"), lang),
                code=_code(summary.get("stop_code") or "—"),
                meaning=stop_explanation(str(summary.get("stop_code") or ""), lang) or "—")]
    if summary.get("mode") == "mock_demo":
        lines.insert(1, rt("mock_result_note", lang))
    lines += _detail_lines(summary.get("stop_detail"), lang)
    lines += device_lines(summary, lang)
    lines += stage_lines(summary.get("stages"), lang)
    if summary.get("candidate_available"):
        lines += [rt("candidate_available", lang, label=_candidate_label(summary, lang),
                     seal=_code(str(summary.get("seal_sha256", ""))[:16]),
                     sha=_code(str(summary.get("candidate_sha256", ""))[:16]))]
        if summary.get("phase") != "completed":
            lines.append(rt("candidate_from_unfinished_run", lang))
    else:
        lines.append(rt("candidate_missing", lang))
    initial = summary.get("initial") or {}
    if initial:
        evidence = initial.get("evidence") or {}
        calls = evidence.get("attempted_invocations", len(initial.get("invocations") or ()))
        lines.append(rt("initial_line", lang, status=_code(initial.get("status", "?")), calls=calls))
        if initial.get("blockers"):
            lines.append(_codes_md(initial["blockers"], lang))
    if summary.get("blockers"):
        lines += [rt("blockers_head", lang), _codes_md(summary["blockers"], lang)]
    if summary.get("initial_reuse"):
        lines.append(rt("initial_reuse_line", lang))
    rejected = {key: summary.get(key) or 0 for key in ("distinct_rejections_total", "repeated_rejected_actions",
                                                        "skipped_rejected_actions", "skipped_rejected_proposals")}
    if any(rejected.values()):
        lines.append(rt("rejections_line", lang, distinct=rejected["distinct_rejections_total"],
                        repeated=rejected["repeated_rejected_actions"],
                        skipped=rejected["skipped_rejected_actions"] + rejected["skipped_rejected_proposals"]))
    waived = summary.get("scientific_gate_waivers") or {}
    if waived:
        lines.append(rt("waivers_line", lang, gates=", ".join(_code(k) for k in waived)))
    for round_number, ids in sorted((warnings or {}).items()):
        shown = ", ".join(_code(i) for i in ids[:20]) + (" …" if len(ids) > 20 else "")
        lines.append(rt("nonpositive_u_warning", lang, round=round_number, sites=shown))
    if summary.get("error_type"):
        lines.append(rt("error_line", lang, error=_code(summary["error_type"])))
    lines += ["", rt("not_validated_note", lang)]
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------- service calls
def load_terminal_summary(service: ReconstructionBackendService, run_id: str) -> tuple[dict, dict | None]:
    """``(state, summary)``. The summary is read only once the state is terminal, and may be absent."""
    state = service.status(run_id)
    if state.get("phase") not in TERMINAL:
        return state, None
    try:
        return state, service.summary(run_id)
    except (FileNotFoundError, json.JSONDecodeError):
        return state, None


def list_runs(service: ReconstructionBackendService, *, limit: int = 30) -> list[list[str]]:
    rows = []
    pattern = re.compile(r"(r-\d{8}T\d{6}Z-[a-f0-9]{8})(?:_(.*))?$")
    entries = sorted((p for p in service.runs_root.iterdir() if p.is_dir() and not p.is_symlink()),
                     key=lambda p: p.name, reverse=True)
    for path in entries:
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            state = service.status(match.group(1))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        rows.append([match.group(1), match.group(2) or "", str(state.get("phase", "")), str(state.get("round", 0)),
                     str(state.get("stop_code") or "")])
        if len(rows) >= limit:
            break
    return rows


def _run_root(service: ReconstructionBackendService, run_id: str) -> Path:
    """Use only the service's supported, validated run-directory accessor."""
    return service.run_root(run_id)


def pinned_seal(service: ReconstructionBackendService, run_id: str) -> tuple[Path, str]:
    """The run directory and the seal digest from the external pin record, cross-checked.

    The pin lives beside, not inside, the sealed run directory. It must name exactly the
    directory the service resolves for this run id; no path from a summary is trusted.
    """
    run_root = _run_root(service, run_id)
    pin_path = service.runs_root / f"{run_id}.seal-pin.json"
    if pin_path.is_symlink() or not pin_path.is_file():
        raise SealError("no external seal pin for this run")
    record = json.loads(pin_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or set(record) != {"run_root", "seal_sha256"}:
        raise SealError("external seal pin has an unexpected form")
    if Path(record["run_root"]).resolve() != run_root.resolve():
        raise SealError("external seal pin names a different run directory")
    digest = record["seal_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SealError("external seal pin digest is malformed")
    return run_root, digest


def export_candidate(service: ReconstructionBackendService, run_id: str, destination: Path) -> list[Path]:
    """Private download copies of the sealed candidate (RES, and CIF/FCF when sealed)."""
    run_root, digest = pinned_seal(service, run_id)
    seal = verify_seal(run_root, private_root=service.runs_root, expected_seal_sha256=digest)
    target = Path(destination) / run_id
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for role, name in CANDIDATE_ROLES:
        try:
            data = read_sealed_output(seal, private_root=service.runs_root, role=role)
        except SealError:
            if role == "model":
                raise
            continue
        path = target / name
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)
        written.append(path)
    return written


def viewer_payload_for(model_copy: Path, lang: str) -> dict | None:
    """3-D view of a private copy of the sealed candidate; coordinates exactly as written."""
    from crystalx_core.geometry.model import build_viewer_payload
    from crystalx_core.io.shelx_res import read_res

    try:
        payload = build_viewer_payload(read_res(model_copy), stage="refined", source_label=rt("viewer_source", "en"))
    except (ValueError, OSError):
        return None
    payload["element_note"] = rt("viewer_element_note", "en")
    payload["source"] = rt("viewer_source", lang)
    return payload


def nonpositive_u_warnings(service: ReconstructionBackendService, run_id: str) -> dict[int, list[str]]:
    """Sites whose Uiso ≤ 0 were kept for geometry-only inference, by round (from the run's own records)."""
    history = service.status(run_id).get("history") or []
    if history and all("nonpositive_u_site_ids" in row for row in history):
        return {int(row["round"]): list(row["nonpositive_u_site_ids"]) for row in history
                if row.get("nonpositive_u_site_ids")}
    run_root = _run_root(service, run_id)
    found: dict[int, list[str]] = {}
    cycles = run_root / "cycles"
    if not cycles.is_dir() or cycles.is_symlink():
        return found
    for directory in sorted(cycles.iterdir()):
        match = re.fullmatch(r"round-(\d{3})", directory.name)
        record = directory / "inference.json"
        if not match or directory.is_symlink() or record.is_symlink() or not record.is_file():
            continue
        if record.resolve().parent != directory.resolve() or directory.resolve().parent != cycles.resolve():
            continue
        try:
            provenance = json.loads(record.read_text(encoding="utf-8")).get("provenance") or {}
        except (OSError, ValueError, AttributeError):
            continue
        ids = provenance.get("nonpositive_u_sites_preserved_for_geometry_only_inference") or []
        if ids:
            found[int(match.group(1))] = [str(i) for i in ids]
    return found


def run_checkcif(service: ReconstructionBackendService, run_id: str, *, toolchain: Any, timeout_s: float,
                 allow_builtin_checkdef: bool, checkdef_path: Path | None = None,
                 clock: Callable[[], float] = time.time):
    """Separate local PLATON diagnostics on the pinned candidate (``execute=True``: the button)."""
    from crystalx_core.app.checkcif import PinnedArtifact, run_local_checkcif
    from crystalx_core.validation.platon import PlatonSettings
    from crystalx_core.workspace.hashing import sha256_file

    run_root, digest = pinned_seal(service, run_id)
    parent = service.runs_root / "checkcif"
    parent.mkdir(mode=0o700, exist_ok=True)
    output = parent / f"{run_id}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(clock()))}"
    suffix = 1
    while output.exists():
        suffix += 1
        output = parent / f"{run_id}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(clock()))}-{suffix}"
    checkdef = PinnedArtifact(Path(checkdef_path), sha256_file(Path(checkdef_path))) if checkdef_path else None
    return run_local_checkcif(run_root, private_root=service.runs_root, expected_seal_sha256=digest,
                              output_dir=output, toolchain=toolchain, settings=PlatonSettings(timeout_s=float(timeout_s)),
                              checkdef=checkdef, allow_builtin_checkdef=bool(allow_builtin_checkdef), execute=True)


def present_local_alerts(result: Any, lang: str) -> str:
    """The separate diagnostic's own status, reasons and alerts. No status here means PASS."""
    if result is None:
        return rt("checkcif_hint", lang)
    status = getattr(result.status, "value", str(result.status))
    lines = [rt("checkcif_status", lang, status=_code(status), meaning=rt(f"checkcif_{status}", lang)),
             _codes_md(result.reasons, lang)]
    report = result.report
    if report is not None and report.lines_read:
        counts = report.counts
        lines.append(rt("checkcif_counts", lang, **{k: counts[k] for k in "ABCG"}))
        if report.alerts:
            lines.append("\n".join(f"- {_code(a.token)} {html.escape(a.text)}" for a in report.alerts[:200]))
    else:
        lines.append(rt("checkcif_no_report", lang))
    if result.program:
        lines.append(rt("checkcif_program", lang, program=_code(result.program),
                        sha=_code((result.executable_sha256 or "")[:16]), calls=result.invocations,
                        verified=rt("yes", lang) if result.process_verified else rt("no", lang)))
    lines.append(rt("checkcif_candidate_kept", lang) if result.candidate_available else rt("checkcif_candidate_lost", lang))
    lines.append(rt("checkcif_scope", lang))
    return "\n\n".join(lines)


def field_labels_missing(lang: str = "en") -> list[str]:
    """Policy fields the form would show without a description (the label test keeps this empty)."""
    paths = [spec.path for group in policy_fields().values() for spec in group]
    return [p for p in paths if field_label(p, lang) is None]
