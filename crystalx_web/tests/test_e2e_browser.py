"""End-to-end: the real page in a headless browser, against an in-process server.

The server uses the mock toolchain for the demo (slowed down so cancellation can be exercised) and
the real toolchain pointed at an empty toolbox for real data, so no crystallographic program
exists to be launched. Skipped when Playwright or a Chromium build is unavailable; nothing is
downloaded.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

from crystalx_core.app import RunMode, RunService  # noqa: E402
from crystalx_core.app.mock_toolchain import MockToolchain  # noqa: E402
from crystalx_core.app.toolchain import RealToolchain  # noqa: E402
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver  # noqa: E402
from crystalx_web.app import WebSettings, build_app, launch_options  # noqa: E402

pytestmark = pytest.mark.e2e

DEMO_BUTTON = "载入内置合成演示数据（虚构结构，配合模拟工具链）"
INS_TEXT = (
    "TITL ordinary\nCELL 0.71073 10 10 10 90 90 90\nZERR 2 0.001 0.001 0.001 0 0 0\n"
    "LATT 1\nSYMM -X, -Y, Z\nSFAC C N O Ag\nUNIT 24 4 8 2\nHKLF 4\nEND\n"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    toolbox = root / "toolbox"
    toolbox.mkdir()

    def factory(request):
        if request.mode is RunMode.MOCK_DEMO:
            return MockToolchain(step_seconds=1.5)
        return RealToolchain(
            resolver=ToolResolver(
                ResolverConfig(toolbox_dir=toolbox, search_path=[], include_known_locations=False,
                               include_olex2=False)
            )
        )

    service = RunService(root / "runs", toolchain_factory=factory)
    from crystalx_native_tools.config import ToolsConfig

    confined = ToolsConfig(toolbox=toolbox, include_olex2=False, include_known_locations=False)
    app = build_app(
        service,
        WebSettings(state_dir=root / "state", poll_seconds=0.5, toolbox_dir=toolbox, tools_config=confined,
                    inventory_search_path=()),
    )
    app.queue()
    port = _free_port()
    app.launch(prevent_thread_lock=True, quiet=True, **launch_options("127.0.0.1", port))
    yield {"url": f"http://127.0.0.1:{port}/", "root": root, "service": service}
    service.shutdown(timeout=10)
    app.close()


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
    page.get_by_text("1 · 输入与化学先验").wait_for()
    yield page
    context.close()


def _body(page) -> str:
    return page.inner_text("body")


def _wait_for_text(page, text: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if text in _body(page):
            return
        page.wait_for_timeout(200)
    raise AssertionError(f"{text!r} did not appear")


def _viewer_frame(page, timeout: float = 15.0):
    page.wait_for_selector('iframe[sandbox="allow-scripts"]', timeout=timeout * 1000)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = next((f for f in page.frames if f.url == "about:srcdoc"), None)
        if frame is not None and frame.query_selector("#note") and frame.inner_text("#note"):
            return frame
        page.wait_for_timeout(200)
    raise AssertionError("the 3-D viewer did not load")


def _upload(page, ins: Path, hkl: Path) -> None:
    page.locator("#cx-ins input[type=file]").set_input_files(str(ins))
    page.locator("#cx-hkl input[type=file]").set_input_files(str(hkl))
    _wait_for_text(page, f"{hkl.stem}\n.hkl")


def _ordinary_case(tmp_path: Path) -> tuple[Path, Path]:
    ins = tmp_path / "ordinary.ins"
    hkl = tmp_path / "ordinary.hkl"
    ins.write_text(INS_TEXT, encoding="utf-8")
    rows = [f"{h:4d}{k:4d}{l:4d}{100.0 / (h + k + l + 1):8.2f}{2.0:8.2f}" for h in range(1, 6) for k in range(4) for l in range(3)]
    hkl.write_text("\n".join(rows + ["   0   0   0    0.00    0.00"]) + "\n", encoding="utf-8")
    return ins, hkl


def test_the_demo_runs_from_upload_to_result_without_leaving_the_machine(page, server):
    page.get_by_role("button", name=DEMO_BUTTON).click()
    _wait_for_text(page, "模拟演示模式")
    page.get_by_role("button", name="诊断与规划").click()
    _wait_for_text(page, "将运行")
    page.get_by_role("button", name="开始运行").click()
    _wait_for_text(page, "初步精修模型（未验证）", timeout=60)

    body = _body(page)
    assert "已完成" in body and "未评估，不可排序或称为已验证" in body
    assert "不是真实结果" in body

    frame = _viewer_frame(page)
    canvas = frame.evaluate("() => { const c = document.getElementById('c'); return [c.width, c.height]; }")
    assert canvas[0] > 100 and canvas[1] > 100
    assert "跨晶胞边界" in frame.inner_text("#note")
    assert "置信度：不可用" in frame.inner_text("#legend")
    sandbox = page.locator('iframe[title="晶体结构三维视图"]').get_attribute("sandbox")
    assert sandbox == "allow-scripts"
    assert "Zn1" in body  # the accessible site table

    page.get_by_role("tab", name="SHELXL 初步精修").click()
    _wait_for_text(page, "0.0412")
    page.get_by_role("tab", name="PLATON 诊断").click()
    _wait_for_text(page, "MOCK001_ALERT_3_B")
    page.get_by_role("tab", name="下载").click()
    _wait_for_text(page, "crystalx_demo\n.cif")  # the file list renders name and suffix apart
    links = page.locator("a[download]").evaluate_all("els => els.map(e => e.getAttribute('href'))")
    assert any(href and href.endswith("crystalx_demo.cif") for href in links)
    page.get_by_role("button", name="导出完整运行记录（zip）").click()
    _wait_for_text(page, "已生成运行记录压缩包")
    page.get_by_role("tab", name="原子指认").click()
    _wait_for_text(page, "AI 元素指认（CrystalX）：不可用")
    assert "SHELXT 初解（解 a）的元素" in _body(page)
    assert "不显示原子置信度" in _body(page)

    local = ("http://127.0.0.1", "data:", "blob:", "about:")
    assert [u for u in page.requests_seen if not u.startswith(local)] == []


def test_a_run_can_be_cancelled_from_the_page(page):
    page.get_by_role("button", name=DEMO_BUTTON).click()
    _wait_for_text(page, "模拟演示模式")
    page.get_by_role("button", name="开始运行").click()
    _wait_for_text(page, "SHELXT is searching")
    page.get_by_role("button", name="取消运行").click()
    _wait_for_text(page, "已取消", timeout=20)
    assert "初步精修模型（未验证）" not in _body(page)


def test_real_data_is_refused_in_mock_mode(page, tmp_path):
    ins, hkl = _ordinary_case(tmp_path)
    _upload(page, ins, hkl)
    page.get_by_text("模拟演示（仅限内置合成数据）").click()
    page.get_by_role("button", name="开始运行").click()
    _wait_for_text(page, "模拟演示只能使用内置的合成演示数据")


def test_missing_programs_block_a_real_run_before_anything_starts(page, tmp_path):
    ins, hkl = _ordinary_case(tmp_path)
    _upload(page, ins, hkl)
    page.get_by_role("button", name="开始运行").click()
    _wait_for_text(page, "已阻止（未启动程序）", timeout=30)
    assert "shelxt was not found" in _body(page)


def test_missing_inputs_are_explained_in_chinese(page):
    page.get_by_role("button", name="开始运行").click()
    _wait_for_text(page, "请上传 INS/RES 指令文件")


def test_the_interface_switches_between_chinese_japanese_and_english(page):
    page.get_by_text("English", exact=True).click()
    _wait_for_text(page, "Start run")
    assert "Load the built-in synthetic demo" in _body(page)
    page.get_by_role("button", name="Load the built-in synthetic demo (fictional structure, mock toolchain)").click()
    _wait_for_text(page, "Mock demo mode")
    page.get_by_role("button", name="Start run").click()
    _wait_for_text(page, "preliminary refined model (unvalidated)", timeout=60)
    frame = _viewer_frame(page)
    assert "Bonds across a cell boundary" in frame.inner_text("#note")
    page.get_by_text("日本語", exact=True).click()
    _wait_for_text(page, "実行開始")
    _wait_for_text(page, "予備精密化モデル（未検証）")
    page.get_by_role("tab", name="PLATON 診断").click()
    _wait_for_text(page, "レベルの意味")
    page.get_by_text("中文", exact=True).click()
    _wait_for_text(page, "开始运行")


def test_native_tool_status_and_the_strict_switch(page):
    body = _body(page)
    assert "原生工具状态与阻断原因" in body
    assert "严格原生（Apple 芯片上不使用 Rosetta 2）" in body
    assert "已阻断" in body and "未找到" in body
    assert "网页不会自动执行" in body
    page.get_by_text("严格原生模式：拒绝仅含 x86_64 代码的程序，不经 Rosetta 2 运行").click()
    _wait_for_text(page, "允许转译")
    page.get_by_text("English", exact=True).click()
    _wait_for_text(page, "Native tools: status and blockers")
    assert "Strict native mode: refuse x86_64-only programs" in _body(page)
    page.get_by_text("日本語", exact=True).click()
    _wait_for_text(page, "ネイティブツールの状態と阻害要因")
    page.get_by_text("中文", exact=True).click()
    _wait_for_text(page, "开始运行")
