"""The provisioned mock demo works with no network at all: no DNS, no route, loopback only.

Linux only: the launcher, the server and a headless browser run inside a fresh network namespace
(``unshare -rn``) whose only interface is loopback. The demo is run end to end from the first-use
button. Elsewhere (macOS) this is skipped; the proxy-poisoning and request-locality checks in the
other desktop tests still apply there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from desktop_support import Installation, finder_env, free_port

pytestmark = pytest.mark.e2e

_DRIVER = textwrap.dedent('''
    import fcntl, json, socket, struct, subprocess, sys, time
    data, opener, port, home = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    result = {"dns_blocked": False, "outbound_blocked": False}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # bring loopback up
        flags = struct.unpack("16sh14x", fcntl.ioctl(s.fileno(), 0x8913, struct.pack("16sh14x", b"lo", 0)))[1]
        fcntl.ioctl(s.fileno(), 0x8914, struct.pack("16sh14x", b"lo", flags | 1))
    try:
        socket.getaddrinfo("pypi.org", 443)
    except OSError:
        result["dns_blocked"] = True
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
    except OSError:
        result["outbound_blocked"] = True
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": home, "TMPDIR": home}
    launch = subprocess.run([sys.executable, "-m", "crystalx_web.desktop", "launch", "--data-dir", data,
                             "--opener", opener, "--port", port], env=env, cwd="/", capture_output=True,
                            text=True, timeout=180)
    result["launch_exit"] = launch.returncode
    result["launch_output"] = launch.stdout[-2000:] + launch.stderr[-2000:]
    if launch.returncode == 0:
        from crystalx_web.desktop import paths_for, read_record
        record = read_record(paths_for(data_dir=data).record)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            seen = []
            page.on("request", lambda r: seen.append(r.url))
            page.goto(record["url"], wait_until="load")
            page.get_by_role("button", name="▶ 一键运行合成演示（MOCK）").click()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and "初步精修模型（未验证）" not in page.inner_text("body"):
                page.wait_for_timeout(300)
            result["demo_finished"] = "初步精修模型（未验证）" in page.inner_text("body")
            local = (record["url"].rstrip("/"), "data:", "blob:", "about:")
            result["non_local_requests"] = [u for u in seen if not u.startswith(local)]
            browser.close()
        stop = subprocess.run([sys.executable, "-m", "crystalx_web.desktop", "stop", "--data-dir", data],
                              env=env, cwd="/", capture_output=True, text=True, timeout=90)
        result["stop_exit"] = stop.returncode
    print(json.dumps(result))
''')


def _namespaces_available() -> bool:
    if not sys.platform.startswith("linux") or shutil.which("unshare") is None:
        return False
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _namespaces_available(), reason="needs Linux user/network namespaces (unshare -rn)")
def test_the_mock_demo_runs_with_no_network(tmp_path: Path):
    pytest.importorskip("playwright.sync_api")
    install = Installation(tmp_path)
    driver = tmp_path / "offline_driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    env = finder_env(install.home)
    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        env["PLAYWRIGHT_BROWSERS_PATH"] = os.environ["PLAYWRIGHT_BROWSERS_PATH"]
    done = subprocess.run(["unshare", "-rn", sys.executable, str(driver), str(install.data), str(install.opener),
                           str(free_port()), str(install.home)], env=env, cwd="/", capture_output=True, text=True,
                          timeout=400)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["dns_blocked"] and result["outbound_blocked"]  # really offline
    assert result["launch_exit"] == 0, result["launch_output"]
    assert result["demo_finished"] and result["non_local_requests"] == []
    assert result["stop_exit"] == 0
    assert not install.marker.exists()
