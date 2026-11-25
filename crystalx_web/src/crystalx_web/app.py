"""The Gradio interface, in Chinese, Japanese and English. Every action goes through the service.

This module builds the page and wires events to :class:`crystalx_core.app.RunService`; it holds no
workflow logic. The page polls the service with a timer while a run is active and stops polling
when the run ends. The language selector relabels every component and re-renders every panel; the
programs' own messages stay verbatim.
"""

from __future__ import annotations

import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from crystalx_core.app import (
    TERMINAL_PHASES,
    ChemicalPriorInput,
    RequestInvalid,
    RunConfig,
    RunMode,
    RunService,
    SolveRequest,
    UnknownRun,
    write_demo_case,
)
from crystalx_web.i18n import (
    DEFAULT_LANG,
    LANG_CHOICES,
    atom_columns,
    issue_message,
    phase_label,
    runs_columns,
    t,
)
from crystalx_web.render import (
    atom_rows,
    capabilities_md,
    diagnosis_md,
    events_text,
    facts_md,
    plan_md,
    platon_md,
    preconditions_md,
    progress_fraction,
    refinement_md,
    solution_md,
    status_md,
)
from crystalx_web.native_panel import assignment_md, inventory_record, native_status_md
from crystalx_web.reconstruction_tab import build_reconstruction_tab
from crystalx_web.reconstruction_text import rt
from crystalx_web.viewer import empty_viewer, viewer_html

__all__ = ["WebSettings", "build_app", "launch_options", "run_tool_kwargs", "TITLE"]

TITLE = "CrystalX"
_TERMINAL = {p.value for p in TERMINAL_PHASES}

CSS = """
.cx-banner{background:#fdf4ff;border:1px solid #f0abfc;color:#701a75;border-radius:8px;padding:8px 12px;margin:4px 0}
.cx-errors p{color:#b91c1c}
.cx-mono textarea{font-family:ui-monospace,Menlo,Consolas,monospace!important;font-size:12px!important}
"""


@dataclass(frozen=True)
class WebSettings:
    """Where the interface keeps its files. All of them live in ignored local directories."""

    state_dir: Path
    """Holds the demo input copy and the per-run download copies."""

    toolbox_dir: Path | None = None
    poll_seconds: float = 1.0
    default_lang: str = DEFAULT_LANG
    tools_config: Any = None
    """A ``crystalx_native_tools.config.ToolsConfig`` (explicit tool paths, search directories)."""

    strict_native_default: bool = True
    """Strict native mode (no Rosetta 2) is on unless the user turns it off."""

    probe_store: Path | None = None
    """Launch-probe records written by ``crystalx-tools probe``; read, never written, here."""

    inventory_search_path: tuple[Path, ...] | None = None
    """Directories standing in for ``PATH`` in the tool inventory. ``None`` (normal use) reads the
    environment; tests pass ``()`` so the page never inspects programs installed on the machine."""

    quick_start: bool = False
    """Show the one-click, clearly labelled synthetic demonstration first (the desktop start does)."""

    desktop_stop: Any = None
    """A callable that stops this desktop server. When set, the page shows a stop control."""

    repository: Path | None = None
    """The developer checkout whose ignored ``runs/`` holds iterative-reconstruction runs. ``None``
    (the packaged app) shows the reconstruction tab as unavailable."""

    reconstruction_service_factory: Any = None
    """Builds the ``ReconstructionBackendService`` for real local programs (default: from ``repository``)."""

    reconstruction_demo_service_factory: Any = None
    """Builds the service for the labelled synthetic demo: the backend's stand-in providers, which
    accept only the built-in data (default: from ``repository``, with its own run folder)."""

    checkcif_toolchain: Any = None
    """``strict -> Toolchain`` for the separate local PLATON run; ``None`` uses the real tool settings."""


def launch_options(host: str = "127.0.0.1", port: int | None = None) -> dict[str, Any]:
    """The keyword arguments every launch uses: local only, no sharing, no telemetry, no CDN fonts."""
    system_fonts = [
        "system-ui", "-apple-system", "PingFang SC", "Hiragino Sans", "Hiragino Sans GB",
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "Yu Gothic", "sans-serif",
    ]
    return {
        "server_name": host,
        "server_port": port,
        "share": False,
        "inbrowser": False,
        "show_error": True,
        "footer_links": [],
        "run_history": False,
        "enable_monitoring": False,
        "ssr_mode": False,
        "pwa": False,
        "mcp_server": False,
        "max_file_size": "500mb",
        "theme": gr.themes.Soft(
            primary_hue="blue",
            font=system_fonts,
            font_mono=["ui-monospace", "Menlo", "Consolas", "monospace"],
        ),
        "css": CSS,
        "head": '<meta name="referrer" content="no-referrer">',
    }


def _split_symbols(text: str) -> tuple[str, ...]:
    return tuple(x for x in re.split(r"[,\s，、;；]+", text or "") if x)


def _file_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    return Path(value) if value else None


class _InputError(ValueError):
    def __init__(self, messages: list[str]) -> None:
        super().__init__("; ".join(messages))
        self.messages = messages


def run_tool_kwargs(settings: WebSettings, strict: bool) -> dict[str, Any]:
    """``RunConfig`` fields for a run: the same locations and policy the native-tool panel inspects."""
    cfg = settings.tools_config
    return {
        "toolbox_dir": settings.toolbox_dir if settings.toolbox_dir is not None else getattr(cfg, "toolbox", None),
        "tool_paths": tuple(sorted((getattr(cfg, "tools", None) or {}).items())),
        "extra_tool_dirs": tuple(getattr(cfg, "search_dirs", ()) or ()),
        "strict_native": bool(strict),
        "include_olex2": bool(getattr(cfg, "include_olex2", True)),
        "include_known_locations": bool(getattr(cfg, "include_known_locations", True)),
    }


def build_app(service: RunService, settings: WebSettings) -> gr.Blocks:
    """The whole page, bound to one service."""
    downloads_root = settings.state_dir / "downloads"
    L0 = settings.default_lang

    def banner_html(lang: str) -> str:
        return f'<div class="cx-banner" role="status">{t("mock_banner", lang)}</div>'

    def mode_choices(lang: str):
        return [(t("mode_real", lang), RunMode.REAL.value), (t("mode_mock", lang), RunMode.MOCK_DEMO.value)]

    def view_choices(lang: str):
        return [(t("view_refined", lang), "refined"), (t("view_solution", lang), "solution")]

    # ------------------------------------------------------------------ helpers
    def tool_kwargs(strict: bool) -> dict[str, Any]:
        return run_tool_kwargs(settings, strict)

    inventory_cache: dict[bool, tuple[dict | None, str]] = {}

    def native_md(strict: bool, lang: str, refresh: bool = False) -> str:
        """Render the tool inventory; a language switch re-renders the cached one, a recheck rebuilds it."""
        key = bool(strict)
        if not refresh and key in inventory_cache:
            return native_status_md(*inventory_cache[key][:1], lang, inventory_cache[key][1])
        cfg = settings.tools_config
        if cfg is not None and settings.toolbox_dir is not None and getattr(cfg, "toolbox", None) != settings.toolbox_dir:
            cfg = cfg.with_overrides(toolbox=settings.toolbox_dir)
        if cfg is None and settings.toolbox_dir is not None:
            try:
                from crystalx_native_tools.config import ToolsConfig

                cfg = ToolsConfig(toolbox=settings.toolbox_dir)
            except ImportError:
                cfg = None
        record, problem = inventory_record(cfg, strict_native=key, probe_store=settings.probe_store,
                                           search_path=settings.inventory_search_path)
        inventory_cache[key] = (record, problem)
        return native_status_md(record, lang, problem)

    def request_from(lang, ins, hkl, mode, metals, elements, peptide, racemic, composition,
                     shelxt_timeout, shelxl_cycles, shelxl_timeout, platon_timeout, strict) -> SolveRequest:
        ins_path, hkl_path = _file_path(ins), _file_path(hkl)
        problems = []
        if ins_path is None:
            problems.append(t("need_ins", lang))
        if hkl_path is None:
            problems.append(t("need_hkl", lang))
        try:
            pairs = ChemicalPriorInput.parse_composition(composition or "")
        except ValueError as exc:
            problems.append(t("bad_composition", lang, detail=exc))
            pairs = ()
        try:
            config = RunConfig(
                shelxt_timeout_s=float(shelxt_timeout),
                shelxl_cycles=int(shelxl_cycles),
                shelxl_timeout_s=float(shelxl_timeout),
                platon_timeout_s=float(platon_timeout),
                **tool_kwargs(strict),
            )
        except (TypeError, ValueError) as exc:
            problems.append(t("bad_limits", lang, detail=exc))
            config = RunConfig(**tool_kwargs(strict))
        if problems:
            raise _InputError(problems)
        return SolveRequest(
            ins_path=ins_path,
            hkl_path=hkl_path,
            prior=ChemicalPriorInput(
                metals=_split_symbols(metals),
                elements=_split_symbols(elements),
                peptide=bool(peptide),
                racemic=bool(racemic),
                composition=pairs,
            ),
            config=config,
            mode=RunMode(mode),
            label=Path(ins_path).stem,
        )

    def errors_md(messages: list[str]) -> str:
        return "\n".join(f"⛔ {html.escape(m)}" for m in messages)

    def invalid_md(exc: RequestInvalid, lang: str) -> str:
        return errors_md([issue_message(i.code, i.field, i.message, lang) for i in exc.issues])

    def progress_html(state: dict | None, lang: str) -> str:
        total = len((state or {}).get("stages", [])) or 8
        done = round(progress_fraction(state) * total)
        return (
            f'<div style="display:flex;gap:8px;align-items:center"><progress value="{done}" max="{total}" '
            f'style="flex:1;height:14px" aria-label="{t("progress_aria", lang)}"></progress>'
            f'<span>{t("progress", lang, done=done, total=total)}</span></div>'
        )

    def copy_downloads(run_id: str) -> list[str]:
        target = downloads_root / run_id
        paths = []
        for relative in service.artefacts(run_id):
            source = service.artefact_path(run_id, relative)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                shutil.copyfile(source, destination)
            paths.append(str(destination))
        return paths

    def final_log(run_id: str, lang: str) -> str:
        logs = [a for a in service.artefacts(run_id) if a.endswith(".stdout.log")]
        if not logs:
            return t("run_ended_no_output", lang)
        text = service.artefact_path(run_id, logs[-1]).read_text(encoding="utf-8", errors="replace")[-20000:]
        return f"[{logs[-1]}] {t('run_ended_log', lang)}\n{text}"

    def viewer_for(run_id: str, choice: str, lang: str):
        if not run_id:
            return empty_viewer(lang), gr.update(value=[], headers=atom_columns(lang))
        try:
            payload = service.viewer_payload(run_id, choice) or (
                service.viewer_payload(run_id, "solution") if choice == "refined" else None
            )
        except UnknownRun:
            payload = None
        return viewer_html(payload, lang=lang), gr.update(value=atom_rows(payload, lang), headers=atom_columns(lang))

    def assignment_for(run_id: str, summary: dict | None, lang: str) -> str:
        refined = None
        if run_id and summary:
            try:
                refined = service.viewer_payload(run_id, "refined")
            except UnknownRun:
                refined = None
        return assignment_md(summary or None, refined, lang)

    def results_for(run_id: str, choice: str, lang: str) -> tuple:
        """facts, solution, refinement, platon, viewer, atoms, assignment -- from the run record on disk."""
        summary = (service.summary(run_id) if run_id else None) or {}
        terminal = bool(run_id) and service.status(run_id).get("phase") in _TERMINAL
        if not terminal:
            view, rows = empty_viewer(lang), gr.update(value=[], headers=atom_columns(lang))
            return (facts_md(None, lang), solution_md(None, lang), refinement_md(None, lang), platon_md(None, lang),
                    view, rows, assignment_md(None, None, lang))
        view, rows = viewer_for(run_id, choice, lang)
        return (facts_md(summary, lang), solution_md(summary, lang), refinement_md(summary, lang),
                platon_md(summary, lang), view, rows, assignment_for(run_id, summary, lang))

    # ------------------------------------------------------------------ handlers
    def on_mode(mode: str, lang: str):
        visible = mode == RunMode.MOCK_DEMO.value
        return gr.update(value=banner_html(lang) if visible else "", visible=visible)

    def on_demo():
        ins, hkl = write_demo_case(settings.state_dir / "demo_input")
        return str(ins), str(hkl), RunMode.MOCK_DEMO.value, "Zn", "", False, False, ""

    def on_capabilities(mode: str, lang: str, strict: bool = True):
        if RunMode(mode) is RunMode.MOCK_DEMO:
            return capabilities_md(service.capabilities(RunMode.MOCK_DEMO), lang)
        return capabilities_md(service.capabilities(RunMode.REAL, config=RunConfig(**tool_kwargs(strict))), lang)

    def on_native(strict: bool, lang: str):
        return native_md(strict, lang)

    def on_native_recheck(strict: bool, lang: str):
        return native_md(strict, lang, refresh=True)

    def on_preview(lang, *fields):
        try:
            preview = service.preview(request_from(lang, *fields))
        except _InputError as exc:
            return errors_md(exc.messages), gr.update(), gr.update(), gr.update(), gr.update()
        except RequestInvalid as exc:
            return invalid_md(exc, lang), gr.update(), gr.update(), gr.update(), gr.update()
        data = {k: preview[k] for k in ("diagnosis", "preconditions", "plan")}
        return ("", diagnosis_md(data["diagnosis"], lang), preconditions_md(data["preconditions"], lang),
                plan_md(data["plan"], lang), data)

    def on_run(lang, *fields):
        empty = (facts_md(None, lang), solution_md(None, lang), refinement_md(None, lang), platon_md(None, lang),
                 empty_viewer(lang), gr.update(value=[], headers=atom_columns(lang)), assignment_md(None, None, lang),
                 None, None)
        try:
            handle = service.submit(request_from(lang, *fields))
        except _InputError as exc:
            return (errors_md(exc.messages), gr.update(), gr.update(), gr.update(active=False), *empty)
        except RequestInvalid as exc:
            return (invalid_md(exc, lang), gr.update(), gr.update(), gr.update(active=False), *empty)
        return ("", handle.run_id, status_md(service.status(handle.run_id), lang), gr.update(active=True), *empty)

    def on_cancel(run_id: str, lang: str):
        if not run_id:
            return t("no_active_run", lang)
        try:
            accepted = service.cancel(run_id)
        except UnknownRun:
            return t("no_active_run", lang)
        note = t("cancel_requested" if accepted else "cancel_too_late", lang)
        return status_md(service.status(run_id), lang) + f"\n\n_{note}_"

    def on_tick(run_id: str, choice: str, lang: str):
        keep = gr.update()
        if not run_id:
            return (gr.update(active=False), *(keep,) * 16)
        try:
            state = service.status(run_id)
        except UnknownRun:
            return (gr.update(active=False), t("run_not_found", lang), *(keep,) * 15)
        summary = service.summary(run_id) or {}
        log_name, log_text = service.log_tail(run_id)
        terminal = state.get("phase") in _TERMINAL
        common = (
            status_md(state, lang),
            progress_html(state, lang),
            final_log(run_id, lang) if terminal else ((f"[{log_name}]\n" + log_text) if log_name else keep),
            events_text(service.events(run_id), lang),
            diagnosis_md(summary["diagnosis"], lang) if summary.get("diagnosis") else keep,
            preconditions_md(summary["preconditions"], lang) if summary.get("preconditions") else keep,
            plan_md(summary["plan"], lang) if summary.get("plan") else keep,
        )
        if not terminal:
            return (gr.update(active=True), *common, *(keep,) * 9)
        return (gr.update(active=False), *common, *results_for(run_id, choice, lang),
                copy_downloads(run_id) or None, None)

    def on_bundle(run_id: str, lang: str):
        if not run_id:
            return None, t("need_finished_run", lang)
        try:
            bundle = service.export_bundle(run_id)
        except (RuntimeError, UnknownRun) as exc:
            return None, t("export_failed", lang, detail=html.escape(str(exc)))
        destination = downloads_root / run_id / bundle.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle, destination)
        return str(destination), t("bundle_ready", lang)

    def on_runs(lang: str):
        rows = [
            [r["run_id"], r["label"], r["mode"], phase_label(r["phase"], lang), r["message"], r["created_at"]]
            for r in service.list_runs()
        ]
        return gr.update(value=rows, headers=runs_columns(lang))

    def on_load(run_id: str, lang: str):
        run_id = (run_id or "").strip()
        try:
            state = service.status(run_id)
        except UnknownRun:
            return "", t("run_not_found", lang), gr.update(active=False)
        return run_id, status_md(state, lang), gr.update(active=True)

    def quick_md(strict: bool, lang: str) -> str:
        """The first-use card: the synthetic demo, and one honest line about real computations."""
        native_md(strict, lang)  # fills the inventory cache when it is still empty
        record, problem = inventory_cache.get(bool(strict), (None, "missing_package"))
        if record is None:
            real = t("quick_real_unknown", lang, detail=html.escape(problem or "-"))
        elif record.get("pipeline") == "ready_static":
            real = t("quick_real_ready", lang)
        else:
            tools = ", ".join(f"`{name}`" for name in record.get("pipeline_blockers", [])) or "-"
            real = t("quick_real_blocked", lang, tools=tools)
        return "\n\n".join([t("quick_title", lang), t("quick_text", lang), real])

    def on_stop_request():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    def on_stop_keep():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    def on_stop_confirm(lang: str):
        if settings.desktop_stop is None:
            return gr.update(visible=False), gr.update(visible=False), t("stop_unavailable", lang)
        settings.desktop_stop()
        return gr.update(visible=False), gr.update(visible=False), t("stop_done", lang, runs=html.escape(
            str(settings.state_dir.parent)))

    def reconstruction_factory():
        """``None`` without a developer checkout: the tab then explains why it is unavailable."""
        if settings.reconstruction_service_factory is not None:
            return settings.reconstruction_service_factory
        if settings.repository is None:
            return None

        def make():
            from crystalx_core.app.reconstruction_backend import ReconstructionBackendService

            repository = Path(settings.repository)
            return ReconstructionBackendService(repository, repository / "runs" / "reconstruction")
        return make

    def reconstruction_demo_factory():
        """The synthetic demo keeps its runs apart from real ones; ``None`` without a developer checkout."""
        if settings.reconstruction_demo_service_factory is not None:
            return settings.reconstruction_demo_service_factory
        if settings.repository is None:
            return None

        def make():
            from crystalx_core.app.reconstruction_backend import ReconstructionBackendService
            from crystalx_core.app.reconstruction_synthetic import assigner_factory, toolchain_factory

            repository = Path(settings.repository)
            return ReconstructionBackendService(repository, repository / "runs" / "reconstruction_demo",
                                                toolchain_factory=toolchain_factory, assigner_factory=assigner_factory)
        return make

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title=TITLE, analytics_enabled=False, fill_width=True) as demo:
        lang = gr.State(L0)
        run_id = gr.State("")
        preview_data = gr.State(None)
        desktop = settings.desktop_stop is not None
        with gr.Row():
            header = gr.Markdown(t("header", L0))
            with gr.Column(scale=0, min_width=260):
                lang_choice = gr.Radio(choices=LANG_CHOICES, value=L0, label=t("language", L0))
                stop_btn = gr.Button(t("stop_server", L0), size="sm", visible=desktop, elem_id="cx-stop")
                stop_confirm = gr.Button(t("stop_confirm", L0), size="sm", variant="stop", visible=False,
                                         elem_id="cx-stop-confirm")
                stop_keep = gr.Button(t("stop_keep", L0), size="sm", visible=False, elem_id="cx-stop-keep")
                stop_note = gr.Markdown("", elem_id="cx-stop-note")
        with gr.Group(visible=settings.quick_start, elem_id="cx-quick"):
            quick = gr.Markdown(quick_md(settings.strict_native_default, L0) if settings.quick_start else "",
                                elem_id="cx-quick-text")
            quick_btn = gr.Button(t("quick_button", L0), variant="primary", elem_id="cx-quick-run")
        banner = gr.HTML("", visible=False, js_on_load=None)
        with gr.Accordion(t("native_accordion", L0), open=not settings.quick_start) as native_acc:
            native = gr.Markdown(native_md(settings.strict_native_default, L0), elem_id="cx-native")
            native_btn = gr.Button(t("recheck", L0), size="sm", elem_id="cx-native-recheck")
        with gr.Accordion(t("caps_accordion", L0), open=False) as caps_acc:
            caps = gr.Markdown(on_capabilities(RunMode.REAL.value, L0, settings.strict_native_default))
            caps_btn = gr.Button(t("recheck", L0), size="sm")
        with gr.Tabs(elem_id="cx-modes"):
            with gr.Tab(rt("tab_solve", L0), id="cx-tab-solve") as solve_tab:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        sec_input = gr.Markdown(t("sec_input", L0))
                        ins = gr.File(label=t("ins_file", L0), file_types=[".ins", ".res"], type="filepath", elem_id="cx-ins")
                        hkl = gr.File(label=t("hkl_file", L0), file_types=[".hkl"], type="filepath", elem_id="cx-hkl")
                        demo_btn = gr.Button(t("demo_button", L0), size="sm", elem_id="cx-demo")
                        mode = gr.Radio(choices=mode_choices(L0), value=RunMode.REAL.value, label=t("mode", L0), elem_id="cx-mode")
                        strict = gr.Checkbox(value=settings.strict_native_default, label=t("strict_native", L0), elem_id="cx-strict")
                        metals = gr.Textbox(label=t("metals", L0), placeholder=t("metals_ph", L0))
                        elements = gr.Textbox(label=t("elements", L0), placeholder=t("elements_ph", L0))
                        with gr.Row():
                            peptide = gr.Checkbox(label=t("peptide", L0))
                            racemic = gr.Checkbox(label=t("racemic", L0))
                        composition = gr.Textbox(label=t("composition", L0), placeholder=t("composition_ph", L0))
                        with gr.Accordion(t("limits_accordion", L0), open=False) as limits_acc:
                            shelxt_timeout = gr.Number(value=3600, label=t("shelxt_timeout", L0), minimum=1)
                            shelxl_cycles = gr.Slider(1, 20, value=4, step=1, label=t("shelxl_cycles", L0))
                            shelxl_timeout = gr.Number(value=900, label=t("shelxl_timeout", L0), minimum=1)
                            platon_timeout = gr.Number(value=600, label=t("platon_timeout", L0), minimum=1)
                            cpu_note = gr.Markdown(t("cpu_note", L0))
                        input_errors = gr.Markdown(elem_classes=["cx-errors"])
                        with gr.Row():
                            preview_btn = gr.Button(t("btn_preview", L0), elem_id="cx-preview")
                            run_btn = gr.Button(t("btn_run", L0), variant="primary", elem_id="cx-run")
                            cancel_btn = gr.Button(t("btn_cancel", L0), variant="stop", elem_id="cx-cancel")
                    with gr.Column(scale=6, min_width=420):
                        sec_plan = gr.Markdown(t("sec_plan", L0))
                        diagnosis = gr.Markdown(t("plan_hint", L0))
                        preconditions = gr.Markdown()
                        plan = gr.Markdown()
                sec_run = gr.Markdown(t("sec_run", L0))
                status = gr.Markdown(t("no_run", L0))
                progress = gr.HTML(progress_html(None, L0), js_on_load=None)
                with gr.Row():
                    log = gr.Textbox(label=t("log_box", L0), lines=12, max_lines=12, autoscroll=True,
                                     interactive=False, elem_classes=["cx-mono"])
                    events = gr.Textbox(label=t("events_box", L0), lines=12, max_lines=12, autoscroll=True,
                                        interactive=False, elem_classes=["cx-mono"])
                sec_results = gr.Markdown(t("sec_results", L0))
                facts = gr.Markdown(facts_md(None, L0))
                with gr.Tabs():
                    with gr.Tab(t("tab_viewer", L0), id="tab-viewer") as tab_viewer:
                        view_choice = gr.Radio(choices=view_choices(L0), value="refined", label=t("view_choice", L0))
                        viewer = gr.HTML(empty_viewer(L0), js_on_load=None)
                        atoms = gr.Dataframe(headers=atom_columns(L0), value=[], interactive=False, wrap=True,
                                             label=t("atom_table", L0))
                    with gr.Tab(t("tab_shelxt", L0), id="tab-shelxt") as tab_shelxt:
                        solution = gr.Markdown(solution_md(None, L0))
                    with gr.Tab(t("tab_shelxl", L0), id="tab-shelxl") as tab_shelxl:
                        refinement = gr.Markdown(refinement_md(None, L0))
                    with gr.Tab(t("tab_platon", L0), id="tab-platon") as tab_platon:
                        platon = gr.Markdown(platon_md(None, L0))
                    with gr.Tab(t("tab_assign", L0), id="tab-assign") as tab_assign:
                        assign = gr.Markdown(assignment_md(None, None, L0), elem_id="cx-assign")
                    with gr.Tab(t("tab_downloads", L0), id="tab-downloads") as tab_downloads:
                        downloads = gr.File(label=t("downloads", L0), file_count="multiple", interactive=False)
                        bundle_btn = gr.Button(t("btn_bundle", L0), elem_id="cx-bundle")
                        bundle_note = gr.Markdown()
                        bundle = gr.File(label=t("bundle_file", L0), interactive=False)
                with gr.Accordion(t("history", L0), open=False) as history_acc:
                    runs = gr.Dataframe(headers=runs_columns(L0), value=[], interactive=False, wrap=True)
                    with gr.Row():
                        load_id = gr.Textbox(label=t("run_id", L0), placeholder="r-20261013T000000Z-xxxxxxxx")
                        load_btn = gr.Button(t("btn_load", L0))
                        runs_btn = gr.Button(t("btn_refresh", L0))
            with gr.Tab(rt("tab_recon", L0), id="cx-tab-recon") as recon_tab:
                recon = build_reconstruction_tab(
                    lang0=L0, service_factory=reconstruction_factory(), state_dir=settings.state_dir,
                    tool_kwargs=tool_kwargs, strict_default=settings.strict_native_default,
                    poll_seconds=settings.poll_seconds, demo_service_factory=reconstruction_demo_factory(),
                    checkcif_toolchain=settings.checkcif_toolchain)
        timer = gr.Timer(settings.poll_seconds, active=False)

        fields = [ins, hkl, mode, metals, elements, peptide, racemic, composition,
                  shelxt_timeout, shelxl_cycles, shelxl_timeout, platon_timeout, strict]
        results = [facts, solution, refinement, platon, viewer, atoms, assign, downloads, bundle]

        # ---------------------------------------------------------- language switch
        relabelled = [header, lang_choice, banner, caps_acc, caps, caps_btn, sec_input, ins, hkl, demo_btn, mode,
                      metals, elements, peptide, racemic, composition, limits_acc, shelxt_timeout, shelxl_cycles,
                      shelxl_timeout, platon_timeout, cpu_note, preview_btn, run_btn, cancel_btn, sec_plan,
                      diagnosis, preconditions, plan, sec_run, status, progress, log, events, sec_results,
                      tab_viewer, view_choice, tab_shelxt, tab_shelxl, tab_platon, tab_downloads, downloads,
                      bundle_btn, bundle, history_acc, runs, load_id, load_btn, runs_btn,
                      native_acc, native, native_btn, strict, tab_assign,
                      quick, quick_btn, stop_btn, stop_confirm, stop_keep,
                      facts, solution, refinement, platon, viewer, atoms, assign]

        def on_language(new: str, mode_value: str, rid: str, choice: str, preview: dict | None, strict_value: bool):
            L = new if new in ("zh", "ja", "en") else DEFAULT_LANG
            state = service.status(rid) if rid else None
            summary = (service.summary(rid) if rid else None) or {}
            source = summary if summary.get("diagnosis") else (preview or {})
            mock = mode_value == RunMode.MOCK_DEMO.value
            updates = [
                gr.update(value=t("header", L)),
                gr.update(label=t("language", L)),
                gr.update(value=banner_html(L) if mock else "", visible=mock),
                gr.update(label=t("caps_accordion", L)),
                on_capabilities(mode_value, L, strict_value),
                gr.update(value=t("recheck", L)),
                gr.update(value=t("sec_input", L)),
                gr.update(label=t("ins_file", L)),
                gr.update(label=t("hkl_file", L)),
                gr.update(value=t("demo_button", L)),
                gr.update(label=t("mode", L), choices=mode_choices(L), value=mode_value),
                gr.update(label=t("metals", L), placeholder=t("metals_ph", L)),
                gr.update(label=t("elements", L), placeholder=t("elements_ph", L)),
                gr.update(label=t("peptide", L)),
                gr.update(label=t("racemic", L)),
                gr.update(label=t("composition", L), placeholder=t("composition_ph", L)),
                gr.update(label=t("limits_accordion", L)),
                gr.update(label=t("shelxt_timeout", L)),
                gr.update(label=t("shelxl_cycles", L)),
                gr.update(label=t("shelxl_timeout", L)),
                gr.update(label=t("platon_timeout", L)),
                gr.update(value=t("cpu_note", L)),
                gr.update(value=t("btn_preview", L)),
                gr.update(value=t("btn_run", L)),
                gr.update(value=t("btn_cancel", L)),
                gr.update(value=t("sec_plan", L)),
                diagnosis_md(source.get("diagnosis"), L) if source.get("diagnosis") else t("plan_hint", L),
                preconditions_md(source.get("preconditions"), L),
                plan_md(source.get("plan"), L),
                gr.update(value=t("sec_run", L)),
                status_md(state, L),
                progress_html(state, L),
                gr.update(label=t("log_box", L)),
                gr.update(label=t("events_box", L), value=events_text(service.events(rid), L) if rid else ""),
                gr.update(value=t("sec_results", L)),
                gr.update(label=t("tab_viewer", L)),
                gr.update(label=t("view_choice", L), choices=view_choices(L), value=choice),
                gr.update(label=t("tab_shelxt", L)),
                gr.update(label=t("tab_shelxl", L)),
                gr.update(label=t("tab_platon", L)),
                gr.update(label=t("tab_downloads", L)),
                gr.update(label=t("downloads", L)),
                gr.update(value=t("btn_bundle", L)),
                gr.update(label=t("bundle_file", L)),
                gr.update(label=t("history", L)),
                on_runs(L),
                gr.update(label=t("run_id", L)),
                gr.update(value=t("btn_load", L)),
                gr.update(value=t("btn_refresh", L)),
                gr.update(label=t("native_accordion", L)),
                native_md(strict_value, L),
                gr.update(value=t("recheck", L)),
                gr.update(label=t("strict_native", L)),
                gr.update(label=t("tab_assign", L)),
                quick_md(strict_value, L) if settings.quick_start else "",
                gr.update(value=t("quick_button", L)),
                gr.update(value=t("stop_server", L)),
                gr.update(value=t("stop_confirm", L)),
                gr.update(value=t("stop_keep", L)),
                *results_for(rid, choice, L),
            ]
            return (L, *updates)

        lang_choice.change(on_language, [lang_choice, mode, run_id, view_choice, preview_data, strict], [lang, *relabelled])
        lang_choice.change(lambda L: (gr.update(label=rt("tab_solve", L)), gr.update(label=rt("tab_recon", L))),
                           lang_choice, [solve_tab, recon_tab])
        lang_choice.change(recon.relabel, [lang_choice, *recon.relabel_inputs], recon.relabel_outputs)
        mode.change(on_mode, [mode, lang], banner)
        mode.change(on_capabilities, [mode, lang, strict], caps)
        caps_btn.click(on_capabilities, [mode, lang, strict], caps)
        strict.change(on_native, [strict, lang], native)
        strict.change(on_capabilities, [mode, lang, strict], caps)
        native_btn.click(on_native_recheck, [strict, lang], native)
        demo_btn.click(on_demo, None, [ins, hkl, mode, metals, elements, peptide, racemic, composition])
        demo_fields = [ins, hkl, mode, metals, elements, peptide, racemic, composition]
        quick_btn.click(on_demo, None, demo_fields).then(
            on_run, [lang, *fields], [input_errors, run_id, status, timer, *results])
        strict.change(lambda s, L: quick_md(s, L) if settings.quick_start else "", [strict, lang], quick)
        stop_btn.click(on_stop_request, None, [stop_btn, stop_confirm, stop_keep])
        stop_keep.click(on_stop_keep, None, [stop_btn, stop_confirm, stop_keep])
        stop_confirm.click(on_stop_confirm, lang, [stop_confirm, stop_keep, stop_note])
        preview_btn.click(on_preview, [lang, *fields], [input_errors, diagnosis, preconditions, plan, preview_data])
        run_btn.click(on_run, [lang, *fields], [input_errors, run_id, status, timer, *results])
        cancel_btn.click(on_cancel, [run_id, lang], status)
        timer.tick(on_tick, [run_id, view_choice, lang],
                   [timer, status, progress, log, events, diagnosis, preconditions, plan, *results])
        view_choice.change(viewer_for, [run_id, view_choice, lang], [viewer, atoms])
        bundle_btn.click(on_bundle, [run_id, lang], [bundle, bundle_note])
        runs_btn.click(on_runs, lang, runs)
        load_btn.click(on_load, [load_id, lang], [run_id, status, timer])
        demo.load(on_runs, lang, runs)
    return demo
