"""End-to-end: the iterative-reconstruction tab in a headless browser, against an in-process server.

The synthetic side uses the backend's stand-in providers (built-in data only). The real side is a
real toolchain confined to an empty toolbox and a checkout without weights, so it can only report
blockers: no crystallographic program exists to be launched and nothing on this machine is
inspected. Skipped when Playwright or a Chromium build is unavailable; nothing is downloaded.
"""

from __future__ import annotations

import socket
import subprocess
import time

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

from crystalx_core.app import RunService  # noqa: E402
from crystalx_core.app.reconstruction_backend import ReconstructionBackendService  # noqa: E402
from crystalx_core.app.reconstruction_synthetic import assigner_factory, toolchain_factory  # noqa: E402
from crystalx_core.app.toolchain import RealToolchain  # noqa: E402
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver  # noqa: E402
from crystalx_web.app import WebSettings, build_app, launch_options  # noqa: E402

pytestmark = pytest.mark.e2e

TAB = "迭代重构（预训练指认 + SHELXL 循环）"
MOCK = "合成演示（MOCK，仅内置数据）"
REAL = "本机真实程序"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("recon").resolve() / "checkout"
    root.mkdir()
    # The upload cache is reached through an alias, as the temporary folder is on macOS
    # (/var is an alias of /private/var). The page has to name the files through the real folder.
    cache = tmp_path_factory.mktemp("upload-cache").resolve()
    alias = cache.with_name(cache.name + "-alias")
    alias.symlink_to(cache, target_is_directory=True)
    environment = pytest.MonkeyPatch()
    environment.setenv("GRADIO_TEMP_DIR", str(alias))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("/runs/\n")
    toolbox = root / "toolbox"
    toolbox.mkdir()

    def confined(request):
        return RealToolchain(resolver=ToolResolver(ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False)))

    real = ReconstructionBackendService(root, root / "runs" / "reconstruction", toolchain_factory=confined)
    demo = ReconstructionBackendService(root, root / "runs" / "reconstruction_demo",
                                        toolchain_factory=toolchain_factory, assigner_factory=assigner_factory)
    from crystalx_native_tools.config import ToolsConfig

    tools = ToolsConfig(toolbox=toolbox, include_olex2=False, include_known_locations=False)
    app = build_app(
        RunService(root / "runs"),
        WebSettings(state_dir=root / "runs" / "_web", poll_seconds=0.5, toolbox_dir=toolbox, tools_config=tools,
                    inventory_search_path=(), reconstruction_service_factory=lambda: real,
                    reconstruction_demo_service_factory=lambda: demo),
    )
    app.queue()
    port = _free_port()
    app.launch(prevent_thread_lock=True, quiet=True, **launch_options("127.0.0.1", port))
    yield {"url": f"http://127.0.0.1:{port}/", "root": root, "real": real, "demo": demo}
    app.close()
    environment.undo()


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            instance = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - no browser on this machine
            pytest.skip(f"Chromium is not available: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, server):
    context = browser.new_context(viewport={"width": 1400, "height": 1000})
    page = context.new_page()
    page.requests_seen = []
    page.on("request", lambda r: page.requests_seen.append(r.url))
    page.goto(server["url"], wait_until="load")
    page.get_by_role("tab", name=TAB).click()
    page.get_by_text("1 · 输入与初解路线").wait_for()
    yield page
    context.close()


def _text(page, selector: str) -> str:
    return page.locator(selector).inner_text()


def _wait_in(page, selector: str, text: str, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        shown = _text(page, selector)
        if text in shown:
            return shown
        page.wait_for_timeout(200)
    raise AssertionError(f"{text!r} did not appear in {selector}: {_text(page, selector)[:400]!r}")


def _load_demo(page) -> None:
    page.locator("#cx-recon-mode").get_by_text(MOCK, exact=True).click()
    _wait_in(page, "#cx-recon-banner", "不是真实结果")
    page.get_by_role("button", name="载入内置合成数据（虚构结构）").click()
    _wait_in(page, "#cx-recon-hkl", "crystalx_demo")
    _wait_in(page, "#cx-recon-ins", "crystalx_demo")


def test_the_synthetic_demo_runs_in_the_page_and_is_marked_as_stand_in(page, server):
    assert page.locator("#cx-recon-banner").is_hidden()  # real mode first: no banner, no demo button
    assert page.get_by_role("button", name="载入内置合成数据（虚构结构）").is_hidden()
    _load_demo(page)
    page.get_by_label("总重构轮数（留空 = 自动数值停止）").fill("2")
    page.locator("#cx-recon-preview").click()
    preview = _wait_in(page, "#cx-recon-preview-md", "静态检查通过")
    assert "这不是启动测试" in preview and "专家参考：未使用" in preview
    page.locator("#cx-recon-start").click()
    summary = _wait_in(page, "#cx-recon-summary", "运行摘要", timeout=90)
    status = _text(page, "#cx-recon-status")
    assert status.startswith("🧪 合成演示（MOCK）") and "fixed_rounds_completed" in status
    assert "合成演示（MOCK）" in summary and "固定轮数已完成" in summary and "不是收敛判断" in summary
    assert "数值候选，未经科学验证" in summary
    for forbidden in ("已收敛", "已验证", "结构正确", "PASS"):
        assert forbidden not in summary and forbidden not in status

    page.locator("#cx-recon-export").click()
    _wait_in(page, "#cx-recon-files", "ai_solved")
    links = page.locator("#cx-recon-files a[download]").evaluate_all("els => els.map(e => e.getAttribute('href'))")
    assert any(href and href.endswith("ai_solved.res") for href in links)
    page.wait_for_selector('iframe[sandbox="allow-scripts"]', timeout=15000)

    page.get_by_text("单独的本机 checkCIF / PLATON 诊断").click()
    page.locator("#cx-recon-checkcif").click()
    checkcif = _wait_in(page, "#cx-recon-checkcif-md", "NOT_EVALUATED")
    assert "不是“零警告”" in checkcif and "警告计数 A" not in checkcif and "候选仍可用" in checkcif

    assert list((server["root"] / "runs" / "reconstruction_demo").glob("r-*"))
    assert not list((server["root"] / "runs" / "reconstruction").glob("r-*"))  # nothing ran on the real side
    local = ("http://127.0.0.1", "data:", "blob:", "about:")
    assert [u for u in page.requests_seen if not u.startswith(local)] == []


def test_real_mode_reports_its_blockers_and_starts_nothing(page, server):
    _load_demo(page)
    page.locator("#cx-recon-mode").get_by_text(REAL, exact=True).click()
    page.locator("#cx-recon-banner").wait_for(state="hidden")
    page.locator("#cx-recon-preview").click()
    preview = _wait_in(page, "#cx-recon-preview-md", "被阻断")
    assert "tool_unavailable:shelxt" in preview and "pretrained_checkpoint_missing" in preview
    assert "没有找到预训练权重文件" in preview
    page.locator("#cx-recon-start").click()
    status = _wait_in(page, "#cx-recon-status", "preview_blocked")
    assert "被阻断" in status and "详情（后端原文）" in status and "合成演示" not in status
    summary = _wait_in(page, "#cx-recon-summary", "没有产生候选文件")
    assert "预检未通过" in summary
    blocked = list((server["root"] / "runs" / "reconstruction").glob("r-*/"))
    assert len(blocked) == 1 and not (blocked[0] / "cycles").exists() and not (blocked[0] / "initial").exists()


def test_the_tab_follows_the_interface_language(page):
    _load_demo(page)
    page.get_by_text("English", exact=True).click()
    _wait_in(page, "#cx-recon-banner", "not a real result")
    page.get_by_role("button", name="Start reconstruction").wait_for()
    assert page.get_by_label("Total reconstruction rounds (blank = automatic numerical stop)").is_visible()
    page.get_by_text("日本語", exact=True).click()
    _wait_in(page, "#cx-recon-banner", "実際の結果ではありません")
    page.get_by_role("button", name="再構成を開始").wait_for()
    page.get_by_text("中文", exact=True).click()
    page.get_by_role("button", name="开始重构").wait_for()
