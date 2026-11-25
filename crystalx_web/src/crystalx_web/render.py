"""Turning service records into Markdown and tables for the interface, in zh / ja / en.

Every string that came from a user file or a program (titles, atom labels, messages) is escaped
before it is placed into Markdown, so a crafted file cannot inject markup into the page. Those
strings are shown verbatim and labelled as original text.
"""

from __future__ import annotations

import datetime as _dt
import html
from typing import Any, Iterable

from crystalx_web.i18n import (
    DEFAULT_LANG,
    alert_meaning,
    atom_columns,
    model_label,
    phase_label,
    readiness_label,
    refinement_label,
    stage_label,
    stage_state,
    t,
)

__all__ = [
    "esc",
    "code",
    "capabilities_md",
    "diagnosis_md",
    "preconditions_md",
    "plan_md",
    "status_md",
    "progress_fraction",
    "facts_md",
    "solution_md",
    "refinement_md",
    "platon_md",
    "events_text",
    "atom_rows",
    "ATOM_COLUMNS",
]

ATOM_COLUMNS = atom_columns(DEFAULT_LANG)
_MD_SPECIAL = str.maketrans({c: f"\\{c}" for c in "\\`*_[]|#~"})


def esc(value: Any) -> str:
    """Escape text for Markdown and HTML. ``None`` becomes a dash."""
    if value is None:
        return "—"
    text = html.escape(str(value), quote=False)
    return text.translate(_MD_SPECIAL).replace("\n", " ")


def code(value: Any) -> str:
    """Inline code. Markdown escapes are not processed inside code spans, so none are added."""
    text = html.escape(str(value), quote=False).replace("`", "'").replace("\n", " ")
    return f"`{text}`"


def _num(value: Any, digits: int = 4, lang: str = DEFAULT_LANG) -> str:
    if value is None:
        return t("unknown", lang)
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{value:.0f}"
    return esc(value)


def _table(header: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = list(header)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _sep(lang: str) -> str:
    return "：" if lang in ("zh", "ja") else ": "


# --------------------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------------------


def capabilities_md(report: dict[str, Any], lang: str = DEFAULT_LANG) -> str:
    host = report.get("host", {})
    rosetta = host.get("rosetta_marker_present")
    rosetta_text = t("rosetta_na" if rosetta is None else ("rosetta_marker" if rosetta else "rosetta_absent"), lang)
    source = t("src_mock" if report.get("mode") == "mock_demo" else "src_real", lang)
    lines = [
        f"**{t('core_version', lang)}** {esc(report.get('core_version'))} · **{t('tool_source', lang)}** {source} · "
        f"**{t('host', lang)}** {esc(host.get('system'))} {esc(host.get('arch'))} · "
        f"**Rosetta 2** {rosetta_text} · **Python** {esc(host.get('python'))}",
        "",
        t("tools_static", lang),
        "",
    ]
    rows = [
        [
            code(name),
            t("pass" if tool.get("usable_by_static_inspection") else "unusable", lang),
            esc(tool.get("diagnostic")),
            esc(tool.get("arch") or "—"),
            esc(tool.get("path") or "—"),
            esc(tool.get("message")),
        ]
        for name, tool in report.get("tools", {}).items()
    ]
    note = t("col_note", lang) + t("original", lang)
    lines.append(_table([t(k, lang) for k in ("col_program", "col_static", "col_diagnostic", "col_arch", "col_path")] + [note], rows))
    lines += ["", f"> {esc(report.get('static_inspection_note'))}", "", t("readiness_title", lang), ""]
    lines.append(
        _table(
            [t("col_component", lang), t("col_status", lang), note],
            [[esc(c["name"]), readiness_label(c["readiness"], lang), esc(c["note"])] for c in report.get("components", [])],
        )
    )
    inference = report.get("crystalx_inference", {})
    lines += [
        "",
        t(
            "inference",
            lang,
            pkg=t("installed" if inference.get("apple_engine_installed") else "not_installed", lang),
            torch=t("installed" if inference.get("torch_installed") else "not_installed", lang),
        ),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# diagnosis, preconditions, plan
# --------------------------------------------------------------------------------------


def diagnosis_md(d: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    if not d:
        return t("not_diagnosed", lang)
    s = _sep(lang)
    cell = d.get("cell")
    sg = d.get("space_group", {})
    comp = d.get("composition", {})
    refl = d.get("reflections", {})
    lines = [f"**{t('d_title', lang)}** {esc(d.get('title') or t('none', lang))}"]
    if cell:
        lines.append(
            f"**{t('d_cell', lang)}** a={cell['a']:.4f} b={cell['b']:.4f} c={cell['c']:.4f} Å · "
            f"α={cell['alpha']:.3f} β={cell['beta']:.3f} γ={cell['gamma']:.3f}° · V={cell['volume']} Å³"
        )
    else:
        lines.append(f"**{t('d_cell', lang)}** {t('d_cell_missing', lang)}")
    lines.append(
        f"**{t('d_wavelength', lang)}** {_num(d.get('wavelength_a'), 5, lang)} Å "
        f"({esc(d.get('radiation'))}{t('d_assumed', lang) if d.get('radiation_assumed') else ''})"
    )
    lines.append(
        f"**{t('d_symmetry', lang)}** {esc(sg.get('summary'))} · {t('d_chirality', lang)}{s}{esc(sg.get('chirality'))}  \n"
        f"{esc(sg.get('explanation'))}"
    )
    override = comp.get("override") or []
    lines.append(
        f"**{t('d_composition', lang)}** SFAC {esc(' '.join(comp.get('sfac', [])) or t('none', lang))} · "
        f"UNIT {esc(' '.join(f'{u:g}' for u in comp.get('unit', [])) or t('none', lang))} · "
        f"{t('d_trusted' if comp.get('trusted') else 'd_placeholder', lang)}"
        + (f" · {t('d_override', lang)}{s}{esc(' '.join(f'{x} {c:g}' for x, c in override))}" if override else "")
    )
    lines.append(
        f"**{t('d_reflections', lang)}** {t('d_count', lang, n=refl.get('count'))} · HKLF {refl.get('hklf')} · "
        f"{t('d_anomalous', lang)}{s}{esc(refl.get('anomalous'))} ({t('d_friedel', lang, n=refl.get('friedel_pairs'))}) · "
        f"{t('d_resolution', lang)} {_num(refl.get('d_max_a'), 2, lang)}–{_num(refl.get('d_min_a'), 2, lang)} Å · "
        f"{t('d_mean_ios', lang)} {_num(refl.get('mean_i_over_sigma'), 1, lang)}"
    )
    if refl.get("completeness_unavailable"):
        lines.append(f"**{t('d_completeness', lang)}** {t('d_not_computed', lang)}{s}{esc(refl['completeness_unavailable'])}")
    enantio = d.get("enantiopure", {})
    lines.append(
        f"**{t('d_handedness', lang)}** {t('d_enantiopure' if enantio.get('value') else 'd_not_enantiopure', lang)}"
        f"{s}{esc(enantio.get('reason'))}"
    )
    text = "  \n".join(lines)
    warnings = d.get("warnings") or []
    if warnings:
        text += f"\n\n**{t('warnings', lang)}{t('original', lang)}**\n" + "\n".join(f"- {esc(w)}" for w in warnings)
    return text


def preconditions_md(p: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    if not p:
        return ""
    s = _sep(lang)
    head = t("pre_ok" if p.get("may_start") else "pre_refused", lang)
    lines = [f"**{t('pre_title', lang)}** {head} — {esc(p.get('summary'))}"]
    for item in p.get("items", []):
        tag = t("pre_block" if item["blocking"] else "pre_note", lang)
        lines.append(
            f"- **[{tag}] {esc(item['kind'])}** {esc(item['summary'])}  \n"
            f"  {t('pre_evidence', lang)}{s}{esc(item['evidence'])}  \n  {t('pre_remedy', lang)}{s}{esc(item['remedy'])}"
            + (f"  \n  {t('pre_override', lang)}{s}{esc(item['override_hint'])}" if item.get("override_hint") else "")
        )
    return "\n".join(lines)


def plan_md(plan: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    if not plan:
        return ""
    s = _sep(lang)
    rows = [
        [
            code(n["solver"]),
            t("will_run" if n["will_run"] else "not_run", lang),
            readiness_label(n["readiness"], lang),
            str(n["priority"]),
            esc("; ".join(n["reasons"])),
        ]
        for n in plan.get("nominated", [])
    ]
    header = [t(k, lang) for k in ("col_route", "col_execute", "col_readiness", "col_priority")]
    lines = [t("plan_routes", lang), "", _table(header + [t("col_reasons", lang) + t("original", lang)], rows)]
    excluded = plan.get("excluded", [])
    if excluded:
        lines += ["", t("plan_excluded", lang)] + [f"- {code(e['solver'])}{s}{esc(e['reason'])}" for e in excluded]
    lines += ["", t("plan_downstream", lang)]
    for step in plan.get("downstream", []):
        budget = f"{step['time_budget_s']:g}"
        lines.append(f"- {code(step['step'])}{s}{esc(step['description'])} ({t('time_limit', lang, s=budget)})")
    shelxt = (plan.get("limits") or {}).get("shelxt") or {}
    if shelxt:
        lines.append(
            "- "
            + t(
                "plan_limits",
                lang,
                cpu=shelxt.get("cpu_seconds"),
                mib=(shelxt.get("max_file_bytes") or 0) // (1024 * 1024),
                files=shelxt.get("max_open_files"),
                nice=shelxt.get("nice_increment"),
            )
        )
    if plan.get("laue_note"):
        lines += ["", f"> {esc(plan['laue_note'])}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# run status
# --------------------------------------------------------------------------------------


def _clock(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return _dt.datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ""


def status_md(state: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    if not state:
        return t("no_run", lang)
    lines = [
        f"**{t('s_run', lang)}** {code(state.get('run_id'))}"
        + (f" · {t('s_mock', lang)}" if state.get("mode") == "mock_demo" else "")
        + f" · **{t('s_state', lang)}** {phase_label(state.get('phase', 'unknown'), lang)}",
        esc(state.get("message")),
    ]
    if state.get("note"):
        lines.append(f"_{esc(state['note'])}_")
    rows = []
    for stage in state.get("stages", []):
        icon, label = stage_state(stage["state"], lang)
        rows.append(
            [
                f"{icon} {stage_label(stage['name'], lang)}",
                label,
                _clock(stage.get("started_at")),
                _clock(stage.get("finished_at")),
                esc(stage.get("message", "")),
            ]
        )
    header = [t(k, lang) for k in ("col_step", "col_status", "col_start", "col_end")]
    return "  \n".join(lines) + "\n\n" + _table(header + [t("col_note", lang) + t("original", lang)], rows)


def progress_fraction(state: dict[str, Any] | None) -> float:
    if not state or not state.get("stages"):
        return 0.0
    done = sum(1 for s in state["stages"] if s["state"] not in ("pending", "running"))
    return done / len(state["stages"])


def events_text(events: list[dict[str, Any]], lang: str = DEFAULT_LANG) -> str:
    s = _sep(lang)
    lines = []
    for event in events:
        stamp = _clock(event.get("ts"))
        kind = event.get("kind", "")
        message = event.get("message", "")
        if kind == "app.stage":
            _, label = stage_state(event.get("state", ""), lang)
            lines.append(f"{stamp}  {stage_label(event.get('stage', ''), lang)}{s}{label} — {message}")
        elif kind == "app.phase":
            lines.append(f"{stamp}  {t('e_phase', lang)}{s}{phase_label(event.get('phase', ''), lang)} — {message}")
        elif kind == "warning":
            lines.append(f"{stamp}  ⚠ {message}")
        elif kind == "command.started":
            lines.append(f"{stamp}  {t('e_exec', lang)}{s}{message}")
        elif kind == "command.finished":
            lines.append(
                f"{stamp}  {t('e_exit', lang)}{s}{event.get('diagnostic', '')} "
                f"({event.get('duration_s', '?')} {t('seconds', lang)}) — {message}"
            )
        elif kind == "input.staged":
            digest = str(event.get("sha256", ""))[:12]
            lines.append(f"{stamp}  {t('e_staged', lang, role=event.get('role', ''), digest=digest)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------


def facts_md(summary: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    if not summary:
        return t("no_results", lang)
    process = summary.get("process", {})
    eligibility = summary.get("scientific_eligibility", {})
    lines = [t("r_mock", lang), ""] if summary.get("mock") else []
    lines.append(
        _table(
            [t("col_fact", lang), t("col_conclusion", lang)],
            [
                [t("f_process", lang), f"{phase_label(process.get('phase', ''), lang)} — {esc(process.get('message'))}"],
                [t("f_model", lang), model_label(summary.get("model_availability", ""), lang)],
                [t("f_refinement", lang), refinement_label(summary.get("refinement_status", ""), lang)],
                [t("f_eligibility", lang), f"{t('f_not_evaluated', lang)} — {esc(eligibility.get('reason'))}"],
            ],
        )
    )
    warnings = summary.get("warnings") or []
    if warnings:
        lines += ["", f"**{t('warnings', lang)}{t('original', lang)}**"] + [f"- {esc(w)}" for w in warnings]
    return "\n".join(lines)


def solution_md(summary: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    solution = (summary or {}).get("solution")
    if not solution:
        return t("no_solution", lang)
    rows = [
        [
            f"{esc(x['letter'])}{t('primary', lang) if x['is_primary'] else ''}",
            esc(x.get("space_group") or t("sg_unstated", lang)),
            _num(x.get("r1"), 3, lang),
            _num(x.get("r_weak"), 3, lang),
            _num(x.get("alpha"), 3, lang),
            str(x.get("n_atoms")),
            esc(" ".join(x.get("elements", []))),
            esc(x.get("chirality")),
        ]
        for x in solution.get("solutions", [])
    ]
    header = [t("col_solution", lang), t("col_sg", lang), "R1", "Rweak", "Alpha",
              t("col_atoms", lang), t("col_elements", lang), t("col_chirality", lang)]
    lines = [_table(header, rows), "", f"> {esc(solution.get('ordering_note'))} {esc(solution.get('element_note'))}"]
    concerns = [c for x in solution.get("solutions", []) for c in x.get("concerns", [])]
    if concerns:
        lines += ["", t("concerns", lang) + t("original", lang)] + [f"- {esc(c)}" for c in concerns]
    lxt = solution.get("lxt", {})
    if lxt.get("found"):
        v = t("verified" if lxt.get("format_verified") else "unverified", lang)
        lines.append("\n" + t("lxt_rows", lang, n=len(lxt.get("trials", [])), v=v))
    return "\n".join(lines)


def refinement_md(summary: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    refinement = (summary or {}).get("refinement")
    if not refinement:
        return t("no_refinement", lang)
    s = _sep(lang)
    if refinement.get("blocked"):
        prefix = t("tool_unavailable" if refinement.get("tool_unavailable") else "refinement_not_done", lang)
        return f"**{prefix}**{s}{esc(refinement.get('reason', ''))}"
    listing = refinement.get("listing", {})
    res = refinement.get("res_r_factors") or {}

    def row(key, field, res_field=None, digits=4):
        return [t(key, lang) if key.startswith("ref_") else key, _num(listing.get(field), digits, lang),
                _num(res.get(res_field), digits, lang) if res_field else "—"]

    rows = [
        row("ref_r1obs", "r1_observed", "r1_observed"),
        row("ref_r1all", "r1_all", "r1_all"),
        row("wR2", "wr2", "wr2"),
        row("GooF", "goof", "goof", 3),
        row("ref_mean_shift", "mean_shift_su", None, 3),
        row("ref_max_shift", "max_shift_su", None, 3),
        row("ref_peak", "highest_peak", "highest_peak", 2),
        row("ref_hole", "deepest_hole", "deepest_hole", 2),
        [t("ref_params", lang), _num(listing.get("parameters_refined"), 4, lang), _num(res.get("n_parameters"), 4, lang)],
        [t("ref_cycles", lang), _num(listing.get("cycles_run"), 4, lang), "—"],
    ]
    settings = refinement.get("settings", {})
    verified = t("ref_verified_yes" if listing.get("format_verified") else "ref_verified_no", lang)
    lines = [
        f"**{t('f_refinement', lang)}** {refinement_label(refinement.get('status', ''), lang)} · "
        f"{t('ref_settings', lang, cycles=settings.get('cycles'), timeout=settings.get('timeout_s'))} · "
        f"{t('ref_diag', lang)} {esc(refinement.get('diagnostic'))} ({_num(refinement.get('duration_s'), 1, lang)} {t('seconds', lang)})",
        "",
        _table([t("ref_value", lang), t("ref_from_lst", lang), t("ref_from_res", lang)], rows),
        "",
        t("ref_unknown_note", lang, v=verified),
    ]
    disagreements = (refinement.get("agreement") or {}).get("disagreements") or []
    if disagreements:
        lines += ["", t("ref_disagree", lang) + t("original", lang)] + [f"- {esc(x)}" for x in disagreements]
    if refinement.get("changes"):
        lines += ["", t("ref_changes", lang)] + [f"- {esc(c)}" for c in refinement["changes"]]
    lines += ["", f"> {esc(refinement.get('reflections_note'))}. {esc(refinement.get('preliminary_note'))}"]
    if refinement.get("status") == "failed":
        lines.insert(0, f"**{t('failed', lang)}**{s}{esc(refinement.get('diagnostic_message'))}\n")
    return "\n".join(lines)


def platon_md(summary: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> str:
    platon = (summary or {}).get("platon")
    if not platon:
        return t("no_platon", lang)
    s = _sep(lang)
    if platon.get("skipped"):
        return f"**{t('skipped', lang)}**{s}{esc(platon.get('reason'))}"
    if platon.get("blocked"):
        prefix = t("tool_unavailable" if platon.get("tool_unavailable") else "blocked", lang)
        return f"**{prefix}**{s}\n" + "\n".join(f"- {esc(r)}" for r in platon.get("reasons", []))
    report = platon.get("report", {})
    counts = report.get("counts", {})
    lines = [
        f"**{t('p_program', lang)}** {code(platon.get('program'))} · **{t('col_diagnostic', lang)}** "
        f"{esc(platon.get('diagnostic'))} · " + " · ".join(f"**{lv}** {counts.get(lv, 0)}" for lv in "ABCG"),
        "",
    ]
    alerts = report.get("alerts", [])
    if alerts:
        header = [t("col_level", lang), t("col_code", lang), t("col_type", lang), t("col_text", lang) + t("original", lang)]
        lines.append(_table(header, [[a["level"], code(a["token"]), str(a["type"]), esc(a["text"])] for a in alerts]))
    else:
        lines.append(t("p_no_alerts", lang))
    if report.get("unrecognised_sample"):
        lines += ["", t("p_unrecognised", lang) + t("original", lang)] + [f"- {esc(x)}" for x in report["unrecognised_sample"]]
    v = t("ref_verified_yes" if report.get("format_verified") else "ref_verified_no", lang)
    lines += [
        "",
        f"{t('p_levels', lang)}{s}" + "; ".join(alert_meaning(lv, lang) for lv in "ABCG"),
        f"> {esc(platon.get('note'))} {t('p_format', lang, v=v)}",
    ]
    return "\n".join(lines)


def atom_rows(payload: dict[str, Any] | None, lang: str = DEFAULT_LANG) -> list[list[Any]]:
    """An accessible, sortable alternative to the 3-D view."""
    if not payload:
        return []
    kind_label = {"atom": t("kind_atom", lang), "unknown": t("kind_unknown", lang), "q_peak": t("kind_q", lang)}
    rows = []
    for atom in payload.get("atoms", []):
        u = atom.get("u_iso") if atom.get("u_iso") is not None else atom.get("u_eq")
        rows.append(
            [
                atom["label"],
                atom["element"],
                kind_label.get(atom["kind"], atom["kind"]),
                *atom["frac"],
                atom.get("occupancy"),
                round(u, 4) if isinstance(u, float) else u,
                atom.get("height"),
            ]
        )
    return rows
