"""The iterative-reconstruction tab: Gradio components wired to :mod:`crystalx_web.reconstruction`.

Every action goes through ``ReconstructionBackendService`` (preview, submit, status, summary,
cancel) or the separate local checkCIF function; nothing here runs a crystallographic program
itself. The policy controls are generated from the backend's dataclasses, so the form follows the
backend contract instead of keeping its own copy of it.
"""

from __future__ import annotations

import html
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from crystalx_web import reconstruction as R
from crystalx_web.i18n import t
from crystalx_web.reconstruction_text import choice_label, field_label, history_columns, rt, run_columns
from crystalx_web.viewer import empty_viewer, viewer_html

__all__ = ["ReconTab", "build_reconstruction_tab"]


@dataclass
class ReconTab:
    """What the page needs from the tab: its relabel hook, plus the handlers (tests call them)."""

    relabel_outputs: list = field(default_factory=list)
    relabel: Callable[..., list] | None = None
    relabel_inputs: list = field(default_factory=list)
    handlers: dict[str, Callable[..., Any]] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)


def build_reconstruction_tab(*, lang0: str, service_factory: Callable[[], Any] | None, state_dir: Path,
                             tool_kwargs: Callable[[bool], dict], strict_default: bool, poll_seconds: float = 1.0,
                             demo_service_factory: Callable[[], Any] | None = None,
                             checkcif_toolchain: Callable[[bool], Any] | None = None) -> ReconTab:
    """Build the tab's components inside the caller's ``gr.Tab`` context.

    ``service_factory`` builds the service for real local programs; ``demo_service_factory`` builds
    a second one whose providers are the backend's synthetic stand-ins (built-in data only). Either
    may be ``None``; a run always stays with the service of the mode it was started in.
    """
    tab = ReconTab()
    labelled: list[tuple[Any, Callable[[str], dict]]] = []
    form: list[tuple[str, Any]] = []
    downloads_root = Path(state_dir) / "downloads" / "reconstruction"
    factories = {"real": service_factory, "mock_demo": demo_service_factory}
    modes = tuple(m for m in R.MODES if factories[m] is not None) or ("real",)
    holder: dict[str, Any] = {}
    warning_cache: dict[str, dict[int, list[str]]] = {}  # terminal runs no longer change
    lock = threading.Lock()

    def service(mode: str | None = None):
        """One service per mode and page (its worker is reused); created on first use."""
        mode = mode if mode in R.MODES else modes[0]
        with lock:
            if mode not in holder:
                factory = factories[mode]
                if factory is None:
                    raise RuntimeError("no developer checkout" if mode == "real" else "synthetic demo unavailable")
                holder[mode] = factory()
            return holder[mode]

    def lab(component, make: Callable[[str], dict]):
        labelled.append((component, make))
        return component

    def label_of(key: str) -> Callable[[str], dict]:
        return lambda L: {"label": rt(key, L)}

    def field_component(spec: R.FieldSpec):
        label = lambda L: field_label(spec.path, L) or spec.path  # noqa: E731 - a new field shows its raw name
        if spec.kind == "bool":
            comp = gr.Checkbox(value=bool(spec.default), label=label(lang0))
        elif spec.kind in ("int", "float"):
            comp = gr.Number(value=spec.default, label=label(lang0), precision=0 if spec.kind == "int" else None)
        elif spec.kind in ("optional_int", "optional_float"):
            comp = gr.Textbox(value="" if spec.default is None else str(spec.default), label=label(lang0),
                              placeholder="auto" if spec.kind == "optional_int" else "—")
        elif spec.kind == "choice":
            choices = lambda L: [(choice_label(spec.path, c, L), c) for c in spec.choices]  # noqa: E731
            comp = gr.Radio(choices=choices(lang0), value=spec.default, label=label(lang0))
            labelled.append((comp, lambda L: {"label": label(L), "choices": choices(L)}))
            form.append((spec.path, comp))
            return comp
        elif spec.kind == "ids":
            comp = gr.Textbox(value=", ".join(spec.default), label=label(lang0))
        else:
            comp = gr.Textbox(value="" if spec.default is None else str(spec.default), label=label(lang0))
        labelled.append((comp, lambda L: {"label": label(L)}))
        form.append((spec.path, comp))
        return comp

    def radio(path: str, values: tuple[str, ...], default: str, key: str):
        choices = lambda L: [(choice_label(path, v, L), v) for v in values]  # noqa: E731
        comp = gr.Radio(choices=choices(lang0), value=default, label=rt(key, lang0))
        labelled.append((comp, lambda L: {"label": rt(key, L), "choices": choices(L)}))
        form.append((path, comp))
        return comp

    def plain(path: str, comp, key: str):
        labelled.append((comp, label_of(key)))
        form.append((path, comp))
        return comp

    available = any(factory is not None for factory in factories.values())
    groups = R.cycle_groups()

    def banner_html(mode: str, lang: str) -> str:
        return f'<div class="cx-banner" role="status">{rt("recon_mock_banner", lang)}</div>' if mode == "mock_demo" else ""

    # ------------------------------------------------------------------ layout
    lab(gr.Markdown(rt("recon_intro", lang0)), lambda L: {"value": rt("recon_intro", L)})
    unavailable = gr.Markdown(rt("recon_unavailable", lang0), visible=service_factory is None,
                              elem_id="cx-recon-unavailable")
    labelled.append((unavailable, lambda L: {"value": rt("recon_unavailable", L)}))
    contract = gr.Markdown(R.contract_note(lang0), visible=bool(R.contract_note(lang0)), elem_id="cx-recon-contract")
    labelled.append((contract, lambda L: {"value": R.contract_note(L)}))
    mode_choices = lambda L: [(choice_label("mode", m, L), m) for m in modes]  # noqa: E731
    mode_radio = gr.Radio(choices=mode_choices(lang0), value=modes[0], label=rt("recon_mode", lang0),
                          visible=len(modes) > 1 or modes[0] == "mock_demo", elem_id="cx-recon-mode")
    labelled.append((mode_radio, lambda L: {"label": rt("recon_mode", L), "choices": mode_choices(L)}))
    form.append(("mode", mode_radio))
    banner = gr.HTML(banner_html(modes[0], lang0), visible=modes[0] == "mock_demo", js_on_load=None,
                     elem_id="cx-recon-banner")
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=340):
            lab(gr.Markdown(rt("sec_recon_input", lang0)), lambda L: {"value": rt("sec_recon_input", L)})
            ins_file = plain("ins_path", gr.File(label=rt("ins_original", lang0), file_types=[".ins"], type="filepath",
                                                 elem_id="cx-recon-ins"), "ins_original")
            hkl_file = plain("hkl_path", gr.File(label=rt("hkl_measured", lang0), file_types=[".hkl"], type="filepath",
                                                 elem_id="cx-recon-hkl"), "hkl_measured")
            demo_btn = lab(gr.Button(rt("btn_recon_demo", lang0), size="sm", visible=modes[0] == "mock_demo",
                                     elem_id="cx-recon-demo"), lambda L: {"value": rt("btn_recon_demo", L)})
            strategy = radio("route.strategy", ("shelxt", "protein_cde", "small_molecule"), "shelxt", "strategy")
            radio("heavy_preparation", R.CHOICES["heavy_preparation"], "current_loop", "heavy_prep")
            with gr.Group(visible=False) as small_group:
                radio("route.small_solver", ("shelxs", "shelxd"), "shelxs", "small_solver")
                radio("route.small_method", ("direct", "patterson"), "direct", "small_method")
                plain("route.find_sites", gr.Number(value=None, precision=0, label=rt("find_sites_small", lang0)),
                      "find_sites_small")
                plain("route.plop", gr.Textbox(label=rt("plop", lang0)), "plop")
            with gr.Group(visible=False) as trials_group:  # SHELXS/SHELXD and protein substructure search
                plain("route.trials", gr.Number(value=100, precision=0, label=rt("trials", lang0)), "trials")
            plain("route.timeout_s", gr.Number(value=600, label=rt("route_timeout", lang0)), "route_timeout")
            with gr.Group(visible=False) as protein_group:
                lab(gr.Markdown(rt("protein_head", lang0)), lambda L: {"value": rt("protein_head", L)})
                plain("protein.protein_confirmed", gr.Checkbox(value=False, label=rt("protein_confirmed", lang0)),
                      "protein_confirmed")
                radio("protein.experiment", ("SAD", "MAD"), "SAD", "experiment")
                plain("protein.space_group", gr.Textbox(label=rt("space_group", lang0)), "space_group")
                plain("protein.find_sites", gr.Number(value=None, precision=0, label=rt("protein_find_sites", lang0)),
                      "protein_find_sites")
                plain("protein.heavy_element", gr.Textbox(label=rt("heavy_element", lang0)), "heavy_element")
                plain("protein.solvent_fraction", gr.Number(value=None, label=rt("solvent_fraction", lang0)),
                      "solvent_fraction")
                for index in range(len(R.DATASET_ROLES)):
                    n = index + 1
                    with gr.Row():
                        role = gr.Dropdown(choices=["", *R.DATASET_ROLES], value="", label=rt("dataset_role", lang0, n=n))
                        labelled.append((role, lambda L, n=n: {"label": rt("dataset_role", L, n=n)}))
                        form.append((f"protein.dataset.{index}.role", role))
                        path = gr.File(label=rt("dataset_file", lang0, n=n), type="filepath")
                        labelled.append((path, lambda L, n=n: {"label": rt("dataset_file", L, n=n)}))
                        form.append((f"protein.dataset.{index}.path", path))
                        fmt = gr.Dropdown(choices=list(R.DATASET_FORMATS), value="SHELX_INTENSITIES",
                                          label=rt("dataset_format", lang0, n=n))
                        labelled.append((fmt, lambda L, n=n: {"label": rt("dataset_format", L, n=n)}))
                        form.append((f"protein.dataset.{index}.format", fmt))
                        friedel = gr.Checkbox(value=False, label=rt("dataset_friedel", lang0, n=n))
                        labelled.append((friedel, lambda L, n=n: {"label": rt("dataset_friedel", L, n=n)}))
                        form.append((f"protein.dataset.{index}.friedel", friedel))
                plain("protein.trace_cycles", gr.Number(value=3, precision=0, label=rt("trace_cycles", lang0)), "trace_cycles")
                plain("protein.density_cycles", gr.Number(value=20, precision=0, label=rt("density_cycles", lang0)),
                      "density_cycles")
                plain("protein.heavy_atoms_in_native", gr.Checkbox(value=True, label=rt("heavy_atoms_in_native", lang0)),
                      "heavy_atoms_in_native")
                radio("protein.preferred_hand", ("auto", "original", "inverted"), "auto", "preferred_hand")
            with gr.Accordion(rt("metal_acc", lang0), open=False) as metal_acc:
                labelled.append((metal_acc, label_of("metal_acc")))
                lab(gr.Markdown(rt("metal_note", lang0)), lambda L: {"value": rt("metal_note", L)})
                plain("metal_only", gr.Checkbox(value=False, label=rt("metal_only", lang0), elem_id="cx-recon-metal"),
                      "metal_only")
                plain("anchor_elements", gr.Textbox(label=rt("anchor_elements", lang0)), "anchor_elements")
            candidates = gr.Textbox(label=rt("candidate_elements", lang0), placeholder=rt("candidate_ph", lang0))
            labelled.append((candidates, lambda L: {"label": rt("candidate_elements", L), "placeholder": rt("candidate_ph", L)}))
            form.append(("candidate_elements", candidates))
            radio("candidate_element_source", ("original_ins", "initial_seed"), "original_ins", "candidate_source")
            for group, key, is_open in (("rounds", "rounds_acc", True), ("protocol", "protocol_acc", False),
                                        ("safeguards", "safeguards_acc", False), ("refinement", "refinement_acc", False),
                                        ("peaks", "peaks_acc", False), ("other", "other_acc", False)):
                if not groups[group]:
                    continue
                with gr.Accordion(rt(key, lang0), open=is_open) as acc:
                    labelled.append((acc, label_of(key)))
                    for spec in groups[group]:
                        field_component(spec)
            with gr.Accordion(rt("adaptive_acc", lang0), open=False) as adaptive_acc:
                labelled.append((adaptive_acc, label_of("adaptive_acc")))
                lab(gr.Markdown(rt("adaptive_note", lang0)), lambda L: {"value": rt("adaptive_note", L)})
                for spec in groups["adaptive"]:
                    field_component(spec)
            radio("device", ("auto", "cpu", "mps"), "auto", "device")
            strict = gr.Checkbox(value=strict_default, label=t("strict_native", lang0), elem_id="cx-recon-strict")
            labelled.append((strict, lambda L: {"label": t("strict_native", L)}))
            form.append(("strict_native", strict))
            plain("label", gr.Textbox(label=rt("label_field", lang0)), "label_field")
            errors = gr.Markdown(elem_classes=["cx-errors"])
            with gr.Row():
                preview_btn = lab(gr.Button(rt("btn_recon_preview", lang0), interactive=available, elem_id="cx-recon-preview"),
                                  lambda L: {"value": rt("btn_recon_preview", L)})
                start_btn = lab(gr.Button(rt("btn_recon_start", lang0), variant="primary", interactive=available,
                                          elem_id="cx-recon-start"), lambda L: {"value": rt("btn_recon_start", L)})
                cancel_btn = lab(gr.Button(rt("btn_recon_cancel", lang0), variant="stop", interactive=available,
                                           elem_id="cx-recon-cancel"), lambda L: {"value": rt("btn_recon_cancel", L)})
        with gr.Column(scale=6, min_width=420):
            lab(gr.Markdown(rt("sec_recon_status", lang0)), lambda L: {"value": rt("sec_recon_status", L)})
            preview = gr.Markdown(rt("preview_hint", lang0), elem_id="cx-recon-preview-md")
            status = gr.Markdown(rt("no_recon_run", lang0), elem_id="cx-recon-status")
            history = gr.Dataframe(headers=history_columns(lang0), value=[], interactive=False, wrap=True,
                                   label=rt("history_label", lang0))
            summary = gr.Markdown(rt("summary_pending", lang0), elem_id="cx-recon-summary")
            export_btn = lab(gr.Button(rt("btn_export", lang0), elem_id="cx-recon-export"),
                             lambda L: {"value": rt("btn_export", L)})
            export_note = gr.Markdown()
            files = gr.File(label=rt("downloads", lang0), file_count="multiple", interactive=False,
                            elem_id="cx-recon-files")
            labelled.append((files, label_of("downloads")))
            viewer = gr.HTML(empty_viewer(lang0), js_on_load=None)
            with gr.Accordion(rt("checkcif_acc", lang0), open=False) as checkcif_acc:
                labelled.append((checkcif_acc, label_of("checkcif_acc")))
                allow_builtin = lab(gr.Checkbox(value=False, label=rt("allow_builtin", lang0)), label_of("allow_builtin"))
                checkdef = lab(gr.File(label=rt("checkdef_file", lang0), type="filepath"), label_of("checkdef_file"))
                platon_timeout = lab(gr.Number(value=600, label=rt("platon_timeout", lang0)), label_of("platon_timeout"))
                checkcif_btn = lab(gr.Button(rt("btn_checkcif", lang0), elem_id="cx-recon-checkcif"),
                                   lambda L: {"value": rt("btn_checkcif", L)})
                checkcif_md = gr.Markdown(rt("checkcif_hint", lang0), elem_id="cx-recon-checkcif-md")
    with gr.Accordion(rt("runs_acc", lang0), open=False) as runs_acc:
        labelled.append((runs_acc, label_of("runs_acc")))
        runs = gr.Dataframe(headers=run_columns(lang0), value=[], interactive=False, wrap=True)
        with gr.Row():
            load_id = lab(gr.Textbox(label=rt("load_run", lang0), placeholder="r-20261112T000000Z-xxxxxxxx"),
                          label_of("load_run"))
            load_btn = lab(gr.Button(rt("btn_load", lang0)), lambda L: {"value": rt("btn_load", L)})
            runs_btn = lab(gr.Button(rt("btn_runs_refresh", lang0)), lambda L: {"value": rt("btn_runs_refresh", L)})
    run_id = gr.State("")
    run_mode = gr.State(modes[0])  # the mode the shown run was started in; its service owns the run
    lang_state = gr.State(lang0)
    last_preview = gr.State(None)
    last_checkcif = gr.State(None)
    timer = gr.Timer(poll_seconds, active=False)
    paths = [p for p, _ in form]
    inputs = [c for _, c in form]
    tab.paths = paths

    # ------------------------------------------------------------------ handlers
    def form_of(values) -> dict[str, Any]:
        return dict(zip(paths, values))

    def request_of(values):
        """``(mode, typed request)``. Tool locations belong to real runs only."""
        values = form_of(values)
        mode = values.get("mode") if values.get("mode") in modes else modes[0]
        tool_paths = tool_kwargs(bool(values.get("strict_native", True)))["tool_paths"] if mode == "real" else ()
        return mode, R.build_request(values, mode=mode, tool_paths=tool_paths)

    def fail(messages) -> str:
        return "\n".join(f"⛔ {html.escape(str(m))}" for m in messages)

    def on_strategy(value):
        return (gr.update(visible=value == "small_molecule"), gr.update(visible=value == "protein_cde"),
                gr.update(visible=value in ("small_molecule", "protein_cde")))

    def on_mode(mode, lang):
        mock = mode == "mock_demo"
        return gr.update(value=banner_html(mode, lang), visible=mock), gr.update(visible=mock), on_runs(lang, mode)

    def on_demo():
        ins, hkl = R.demo_inputs(Path(state_dir))
        return str(ins), str(hkl)

    def on_preview(lang, *values):
        try:
            mode, request = request_of(values)
            result = service(mode).preview(request)
        except R.FormError as exc:
            return fail(exc.messages), gr.update(), None
        except Exception as exc:  # backend refusal: shown verbatim, nothing is retried or substituted
            return fail([f"{type(exc).__name__}: {exc}"]), gr.update(), None
        return "", R.preview_md(result, lang), result

    def table(rows, lang):
        return gr.update(value=rows, headers=history_columns(lang), label=rt("history_label", lang))

    def render(rid: str, lang: str, mode: str | None = None):
        """Status, per-round table, summary and whether to keep polling, from the backend's records."""
        if not rid:
            return rt("no_recon_run", lang), table([], lang), rt("summary_pending", lang), False
        owner = service(mode)
        state, summary_record = R.load_terminal_summary(owner, rid)
        rows = R.history_rows((summary_record or {}).get("history") or state.get("history"), lang)
        if summary_record and rid not in warning_cache:
            warning_cache[rid] = R.nonpositive_u_warnings(owner, rid)
        warnings = warning_cache.get(rid) if summary_record else None
        terminal = state.get("phase") in R.TERMINAL
        summary_text = R.summary_md(summary_record, lang, warnings=warnings) if terminal else rt("summary_pending", lang)
        status_text = R.status_md(state, lang)
        if owner is holder.get("mock_demo"):  # a stand-in run can never be mistaken for a real one
            status_text = rt("mock_result_note", lang) + "\n\n" + status_text
        return status_text, table(rows, lang), summary_text, not terminal

    def on_start(lang, *values):
        keep = gr.update()
        try:
            mode, request = request_of(values)
            handle = service(mode).submit(request)
        except R.FormError as exc:
            return (fail(exc.messages), keep, keep, keep, keep, keep, gr.update(active=False))
        except Exception as exc:
            return (fail([f"{type(exc).__name__}: {exc}"]), keep, keep, keep, keep, keep, gr.update(active=False))
        text, rows, summary_text, active = render(handle.run_id, lang, mode)
        return "", handle.run_id, mode, text, rows, summary_text, gr.update(active=active)

    def on_tick(rid, lang, mode=None):
        if not rid:
            return gr.update(active=False), gr.update(), gr.update(), gr.update()
        try:
            text, rows, summary_text, active = render(rid, lang, mode)
        except (ValueError, OSError, RuntimeError) as exc:
            return gr.update(active=False), fail([f"{type(exc).__name__}: {exc}"]), gr.update(), gr.update()
        return gr.update(active=active), text, rows, summary_text

    def on_cancel(rid, lang, mode=None):
        if not rid:
            return rt("no_selection", lang)
        try:
            delivered = service(mode).cancel(rid)
            state = service(mode).status(rid)
        except (ValueError, OSError, RuntimeError) as exc:
            return fail([f"{type(exc).__name__}: {exc}"])
        return R.status_md(state, lang) + "\n\n_" + rt("cancel_requested" if delivered else "cancel_not_active", lang) + "_"

    def on_export(rid, lang, mode=None):
        if not rid:
            return None, rt("no_selection", lang), empty_viewer(lang)
        try:
            written = R.export_candidate(service(mode), rid, downloads_root)
        except Exception as exc:  # no pin, a broken seal or a missing candidate: say so, offer nothing
            return None, rt("export_failed", lang, detail=html.escape(f"{type(exc).__name__}: {exc}")), empty_viewer(lang)
        if not written:
            return None, rt("export_none", lang), empty_viewer(lang)
        model = next((p for p in written if p.name == "ai_solved.res"), None)
        payload = R.viewer_payload_for(model, lang) if model else None
        return [str(p) for p in written], "", viewer_html(payload, lang=lang) if payload else empty_viewer(lang)

    def on_checkcif(rid, lang, allow, checkdef_file, timeout, strict_value, mode=None):
        if not rid:
            return rt("no_selection", lang), None
        try:
            owner = service(mode)
            # A stand-in run gets the stand-in toolchain: the backend then reports NOT_EVALUATED
            # instead of running the real PLATON on placeholder files.
            make = checkcif_toolchain or (_mock_toolchain if owner is holder.get("mock_demo") else _real_toolchain(tool_kwargs))
            toolchain = make(bool(strict_value))
            result = R.run_checkcif(owner, rid, toolchain=toolchain, timeout_s=float(timeout or 600),
                                    allow_builtin_checkdef=bool(allow),
                                    checkdef_path=Path(checkdef_file) if checkdef_file else None)
        except Exception as exc:
            return fail([f"{type(exc).__name__}: {exc}"]), None
        return R.present_local_alerts(result, lang), result

    def on_runs(lang, mode=None):
        """Runs of the selected mode (each mode keeps its own run folder)."""
        mode = mode if mode in modes else modes[0]
        if factories[mode] is None:
            return gr.update(value=[], headers=run_columns(lang))
        try:
            rows = R.list_runs(service(mode))
        except Exception:  # the list is a convenience; the status box carries the real errors
            rows = []
        return gr.update(value=rows, headers=run_columns(lang))

    def on_load(value, lang, mode=None):
        rid = (value or "").strip()
        mode = mode if mode in modes else modes[0]
        keep = gr.update()
        try:
            text, rows, summary_text, active = render(rid, lang, mode)
        except (ValueError, OSError, RuntimeError) as exc:
            return keep, keep, fail([f"{type(exc).__name__}: {exc}"]), keep, keep, gr.update(active=False)
        return rid, mode, text, rows, summary_text, gr.update(active=active)

    strategy.change(on_strategy, strategy, [small_group, protein_group, trials_group])
    mode_radio.change(on_mode, [mode_radio, lang_state], [banner, demo_btn, runs])
    demo_btn.click(on_demo, None, [ins_file, hkl_file])
    preview_btn.click(on_preview, [lang_state, *inputs], [errors, preview, last_preview])
    start_btn.click(on_start, [lang_state, *inputs], [errors, run_id, run_mode, status, history, summary, timer])
    timer.tick(on_tick, [run_id, lang_state, run_mode], [timer, status, history, summary])
    cancel_btn.click(on_cancel, [run_id, lang_state, run_mode], status)
    export_btn.click(on_export, [run_id, lang_state, run_mode], [files, export_note, viewer])
    checkcif_btn.click(on_checkcif, [run_id, lang_state, allow_builtin, checkdef, platon_timeout, strict, run_mode],
                       [checkcif_md, last_checkcif])
    runs_btn.click(on_runs, [lang_state, mode_radio], runs)
    load_btn.click(on_load, [load_id, lang_state, mode_radio], [run_id, run_mode, status, history, summary, timer])

    # ------------------------------------------------------------------ language
    def relabel(lang, rid, mode_of_run, selected_mode, preview_record, checkcif_result):
        updates = [gr.update(**make(lang)) for _, make in labelled]
        try:
            text, rows, summary_text, _ = render(rid if available else "", lang, mode_of_run)
        except (ValueError, OSError, RuntimeError):
            text, rows, summary_text = rt("no_recon_run", lang), table([], lang), rt("summary_pending", lang)
        return [lang, *updates, gr.update(value=banner_html(selected_mode, lang)), R.preview_md(preview_record, lang),
                text, rows, summary_text, R.present_local_alerts(checkcif_result, lang), on_runs(lang, selected_mode)]

    tab.relabel_outputs = [lang_state, *[c for c, _ in labelled], banner, preview, status, history, summary,
                           checkcif_md, runs]
    tab.relabel_inputs = [run_id, run_mode, mode_radio, last_preview, last_checkcif]
    tab.relabel = relabel
    tab.handlers = {"preview": on_preview, "start": on_start, "tick": on_tick, "cancel": on_cancel,
                    "export": on_export, "checkcif": on_checkcif, "runs": on_runs, "load": on_load,
                    "strategy": on_strategy, "relabel": relabel, "service": service, "mode": on_mode,
                    "demo": on_demo}
    return tab


def _mock_toolchain(strict: bool):
    """The core's stand-in toolchain; local PLATON diagnostics refuse to count it as an evaluation."""
    from crystalx_core.app.mock_toolchain import MockToolchain

    return MockToolchain(step_seconds=0.001)


def _real_toolchain(tool_kwargs: Callable[[bool], dict]) -> Callable[[bool], Any]:
    """The same tool locations and native policy as the tool panel (for the separate PLATON run)."""
    def make(strict: bool):
        from crystalx_core.app import RunConfig
        from crystalx_core.app.toolchain import RealToolchain

        try:
            from crystalx_native_tools.inspect import candidate_check_for

            check = candidate_check_for()
        except ImportError:
            check = None
        return RealToolchain.for_config(RunConfig(**tool_kwargs(strict)), candidate_check=check)
    return make
