"""The page served by a launcher-started desktop server, in a headless browser.

Unlike ``test_e2e_browser.py`` (which calls ``build_app`` in-process), the server here is started
by the real desktop launcher in a Finder-like environment, so the first-use route, cancellation,
the real-mode strict checks and the owned shutdown are exercised the way a double click reaches
them. The tool folder is a tripwire: nothing may start any of its programs.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

from crystalx_web import desktop  # noqa: E402
from desktop_support import Installation, free_port  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(desktop.fcntl is None, reason="POSIX only")]

INS_TEXT = (
    "TITL ordinary\nCELL 0.71073 10 10 10 90 90 90\nZERR 2 0.001 0.001 0.001 0 0 0\n"
    "LATT 1\nSYMM -X, -Y, Z\nSFAC C N O Ag\nUNIT 24 4 8 2\nHKLF 4\nEND\n"
)


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
def started(tmp_path):
    install = Installation(tmp_path)
    result = install.run("launch", "--port", str(free_port()))
    assert result.returncode == 0, result.stdout + result.stderr
    yield install, install.record()
    install.cleanup()
    assert not install.marker.exists(), "a crystallographic program (tripwire) was started"


@pytest.fixture
def page(browser, started):
    install, record = started
    context = browser.new_context(viewport={"width": 1400, "height": 1000})
    page = context.new_page()
    page.requests_seen = []
    page.on("request", lambda r: page.requests_seen.append(r.url))
    page.goto(record["url"], wait_until="load")
    page.get_by_text("快速开始 · 合成演示（MOCK）").wait_for()
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


def test_first_use_is_the_labelled_synthetic_demo(page, started):
    body = _body(page)
    assert "不需要** SHELX" not in body  # rendered, not raw Markdown
    assert "不需要 SHELX、PLATON、XQuartz 或模型权重" in body and "不是真实结果" in body
    assert "真实计算：目前被阻断" in body  # the tripwire folder holds no verified native program
    page.get_by_role("button", name="▶ 一键运行合成演示（MOCK）").click()
    _wait_for_text(page, "模拟演示模式")
    _wait_for_text(page, "初步精修模型（未验证）", timeout=90)
    assert "已完成" in _body(page)
    local = (started[1]["url"].rstrip("/"), "data:", "blob:", "about:")
    assert [u for u in page.requests_seen if not u.startswith(local)] == []


def test_a_demo_run_can_be_cancelled(page):
    page.get_by_role("button", name="▶ 一键运行合成演示（MOCK）").click()
    _wait_for_text(page, "模拟演示模式")
    page.get_by_role("button", name="取消运行").click()  # the mock run takes several seconds
    _wait_for_text(page, "已取消", timeout=30)
    assert "初步精修模型（未验证）" not in _body(page)


def test_real_data_stays_out_of_the_mock_tools_and_strict_checks_hold(page, tmp_path):
    ins, hkl = tmp_path / "ordinary.ins", tmp_path / "ordinary.hkl"
    ins.write_text(INS_TEXT, encoding="utf-8")
    rows = [f"{h:4d}{k:4d}{l:4d}{100.0 / (h + k + l + 1):8.2f}{2.0:8.2f}"
            for h in range(1, 6) for k in range(4) for l in range(3)]
    hkl.write_text("\n".join(rows + ["   0   0   0    0.00    0.00"]) + "\n", encoding="utf-8")
    page.locator("#cx-ins input[type=file]").set_input_files(str(ins))
    page.locator("#cx-hkl input[type=file]").set_input_files(str(hkl))
    _wait_for_text(page, "ordinary\n.hkl")
    page.get_by_role("radio", name="模拟演示（仅限内置合成数据）").check()
    page.get_by_role("button", name="开始运行").click()
    _wait_for_text(page, "模拟演示只能使用内置的合成演示数据")
    page.get_by_role("radio", name="本机真实程序").check()
    page.get_by_role("button", name="开始运行").click()
    _wait_for_text(page, "已阻止（未启动程序）", timeout=60)


def test_the_stop_control_ends_the_owned_server_and_keeps_the_records(page, started):
    install, record = started
    page.get_by_role("button", name="▶ 一键运行合成演示（MOCK）").click()
    _wait_for_text(page, "初步精修模型（未验证）", timeout=90)
    runs_before = sorted(p.name for p in install.paths.runs_dir.iterdir())
    page.get_by_role("button", name="停止本地服务").click()
    page.get_by_role("button", name="确认停止（进行中的运行会被取消，记录保留）").click()
    _wait_for_text(page, "本地服务正在停止")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and desktop.lock_is_held(install.paths.instance_lock):
        time.sleep(0.2)
    assert not desktop.lock_is_held(install.paths.instance_lock) and install.record() is None
    assert not desktop.prove_identity(record["port"], record["secret"], timeout=1)
    assert sorted(p.name for p in install.paths.runs_dir.iterdir()) == runs_before  # nothing deleted
    again = install.run("launch", "--port", str(free_port()))  # a later double click starts afresh
    assert again.returncode == 0 and install.record()["pid"] != record["pid"]
