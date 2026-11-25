"""The native-tool status panel and the atom-assignment panel.

The tool panel renders an inventory from ``crystalx_native_tools`` (static inspection: files are
read, nothing is run). It never starts a launch probe; it only shows a probe record that the owner
created explicitly from the command line. Technical messages stay verbatim (English); labels and
guidance are translated.

The assignment panel states what assigns elements today (SHELXT peak heights, unverified) and that
the experimental AI backend is unavailable. Whether that backend is installed is read from package
metadata only -- the package (and PyTorch) is never imported here.
"""

from __future__ import annotations

import importlib.metadata
from collections import Counter
from pathlib import Path
from typing import Any

from crystalx_web.i18n import DEFAULT_LANG, blocker_label, native_label, t
from crystalx_web.render import _table, code, esc

__all__ = ["inventory_record", "native_status_md", "engine_status", "assignment_md", "PIPELINE_TOOLS"]

PIPELINE_TOOLS = ("shelxt", "shelxl", "platon_special", "platon")
_ENGINE_DISTRIBUTION = "crystalx-engine-apple-silicon"


def inventory_record(
    tools_config: Any,
    *,
    strict_native: bool,
    probe_store: Path | None,
    search_path: tuple[Path, ...] | None = None,
) -> tuple[dict | None, str]:
    """Build the inventory; returns ``(record, problem)``. Never raises into the page."""
    try:
        from crystalx_native_tools.inventory import build_inventory
        from crystalx_native_tools.probe import ProbeStore
    except ImportError:
        return None, "missing_package"
    try:
        store = ProbeStore(probe_store) if probe_store is not None else None
        inventory = build_inventory(
            tools_config,
            strict_native=strict_native,
            probe_store=store,
            search_path=list(search_path) if search_path is not None else None,
        )
        return inventory.to_record(), ""
    except Exception as exc:  # pragma: no cover - defensive: the page must still render
        return None, f"{type(exc).__name__}: {exc}"


def native_status_md(record: dict | None, lang: str = DEFAULT_LANG, problem: str = "") -> str:
    if record is None:
        detail = t("native_missing_pkg", lang) if problem == "missing_package" else esc(problem)
        return t("native_unavailable", lang, detail=detail)
    host = record.get("host", {})
    lines = [
        t("native_policy_strict" if record.get("strict_native") else "native_policy_translation", lang),
        f"{t('native_host', lang)} {esc(host.get('system'))} {esc(host.get('hardware_arch'))}"
        f"{t('native_translated', lang) if host.get('process_translated') else ''} · macOS "
        f"{esc(host.get('macos_version') or '—')} · "
        + t("native_rosetta_markers", lang, n=len(host.get("rosetta_markers") or [])),
        "",
    ]
    if record.get("pipeline") == "ready_static":
        lines.append(t("native_ready", lang))
    else:
        lines.append(t("native_blocked", lang, tools=", ".join(code(x) for x in record.get("pipeline_blockers", []))))
    config_problems = (record.get("config") or {}).get("problems") or []
    if config_problems:
        lines += ["", t("native_config_problems", lang)]
        lines.extend(f"- {esc(problem)}" for problem in config_problems)
    lines.append("")
    rows = []
    tools = {tool["tool"]: tool for tool in record.get("tools", [])}
    for name in PIPELINE_TOOLS:
        tool = tools.get(name)
        if tool is None:
            continue
        selected = tool.get("selected") or {}
        candidate = selected or (tool.get("candidates") or [{}])[0]
        rows.append(
            [
                code(name),
                native_label(tool.get("static", ""), lang),
                native_label((tool.get("launch") or {}).get("status", "not_tested"), lang),
                native_label(tool.get("validated", "not_verified"), lang),
                native_label(candidate.get("native", ""), lang) if candidate else "—",
                esc(selected.get("path") or "—"),
            ]
        )
    lines.append(_table([t(k, lang) for k in ("col_tool", "col_static_check", "col_launch", "col_validated",
                                                "col_native", "col_selected")], rows))
    blocker_lines = []
    for name in PIPELINE_TOOLS:
        tool = tools.get(name)
        if not tool:
            continue
        for blocker in tool.get("blockers", []):
            blocker_lines.append(
                f"- {code(name)} — **{esc(blocker_label(blocker.get('code', ''), lang))}**  \n"
                f"  {esc(blocker.get('message'))}"
                + (f"  \n  → {esc(blocker.get('next_step'))}" if blocker.get("next_step") else "")
            )
        availability = tool.get("availability")
        if tool.get("blockers") and availability:
            blocker_lines.append(
                "  " + t("native_evidence", lang,
                         build=native_label(availability.get("native_arm64_build", "unknown"), lang),
                         source=native_label(availability.get("source_code", "unknown"), lang))
            )
    if blocker_lines:
        lines += ["", t("native_blockers", lang), "", *blocker_lines]
    for name in ("platon_special", "platon"):
        tool = tools.get(name)
        resources = (tool or {}).get("resources")
        if not resources:
            continue
        found = [c for c in resources.get("checkdef", []) if c.get("exists")]
        found_text = ", ".join(f"{esc(c['path'])} (v{esc(c.get('version_date') or '?')})" for c in found) or t(
            "native_checkdef_none", lang
        )
        lines += ["", f"{code(name)} " + t("native_checkdef", lang, found=found_text,
                                            match=native_label(resources.get("version_match", "unknown"), lang))]
        lines += [f"- {esc(note)}" for note in resources.get("notes", [])]
    lines += ["", t("native_note", lang)]
    return "\n".join(lines)


def engine_status() -> dict[str, Any]:
    """Installed version of the experimental Apple backend, from metadata; never imports it."""
    try:
        version = importlib.metadata.version(_ENGINE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = None
    except Exception:  # pragma: no cover - broken metadata
        version = None
    return {"installed": version is not None, "version": version, "integrated": False}


def assignment_md(summary: dict | None, refined: dict | None, lang: str = DEFAULT_LANG,
                  engine: dict | None = None) -> str:
    engine = engine if engine is not None else engine_status()
    engine_text = (
        t("engine_present", lang, version=esc(engine.get("version"))) if engine.get("installed")
        else t("engine_absent", lang)
    )
    lines = [t("assign_traditional", lang), "", t("assign_ai", lang, engine=engine_text), ""]
    solutions = ((summary or {}).get("solution") or {}).get("solutions") or []
    primary = next((s for s in solutions if s.get("is_primary")), solutions[0] if solutions else None)
    if primary:
        lines.append(t("assign_elements", lang, letter=esc(primary.get("letter")),
                       elements=esc(" ".join(primary.get("elements", [])) or "—")))
    if refined:
        counts = Counter(a.get("element") for a in refined.get("atoms", []) if a.get("kind") == "atom")
        if counts:
            lines.append("")
            lines.append(t("assign_refined", lang,
                           counts=esc(", ".join(f"{el} {n}" for el, n in sorted(counts.items())))))
    if not primary and not refined:
        lines.append(t("assign_no_run", lang))
    return "\n".join(lines)
