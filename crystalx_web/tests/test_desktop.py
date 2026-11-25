"""Desktop start: identity, ports, stale state, proxies, failures and the double-click launcher.

The process tests start real servers through the real launcher, in a Finder-like environment, and
stop them through the authenticated stop. These are the acceptance tests for the desktop start;
the browser tests that call ``build_app`` directly do not replace them.
"""

from __future__ import annotations

import http.server
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from crystalx_web import desktop
from desktop_support import Bystander, Installation, dead_pid, finder_env, free_port, listening_addresses

REPO_WEB = Path(__file__).resolve().parents[1]
COMMAND_FILE = REPO_WEB / "CrystalX Web Demo (developer).command"

pytestmark = pytest.mark.skipif(desktop.fcntl is None, reason="desktop start is POSIX only")


@pytest.fixture
def install(tmp_path):
    installation = Installation(tmp_path)
    yield installation
    installation.cleanup()
    assert not installation.marker.exists(), "a crystallographic program (tripwire) was started"


# ------------------------------------------------------------------------------------- units


def test_loopback_is_exempted_from_proxies_and_existing_exclusions_are_kept():
    env = desktop.loopback_proxy_exemption({"HTTP_PROXY": "http://proxy:3128", "no_proxy": "corp.example,.lan",
                                            "NO_PROXY": ".lan,10.0.0.0/8"})
    assert env["HTTP_PROXY"] == "http://proxy:3128"  # the proxy itself is left alone
    assert env["NO_PROXY"] == env["no_proxy"] == ".lan,10.0.0.0/8,corp.example,127.0.0.1,localhost,::1"
    assert desktop.loopback_proxy_exemption({})["NO_PROXY"] == "127.0.0.1,localhost,::1"


def test_an_occupied_port_is_skipped_and_its_listener_left_alone():
    bystander = Bystander()
    try:
        assert not desktop.port_is_free(bystander.port)
        port, skipped = desktop.choose_port(bystander.port, span=20)
        assert port != bystander.port and skipped[0] == bystander.port and desktop.port_is_free(port)
        assert bystander.alive()
    finally:
        bystander.close()


def test_the_record_is_private_and_malformed_records_are_ignored(tmp_path):
    record = tmp_path / "instance.json"
    desktop.write_record(record, {"schema": desktop.SCHEMA, "pid": 1, "port": 2, "secret": "s" * 43})
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert desktop.read_record(record)["port"] == 2
    for bad in ({"schema": "other", "pid": 1, "port": 2, "secret": "s" * 43},
                {"schema": desktop.SCHEMA, "pid": "1", "port": 2, "secret": "s" * 43},
                {"schema": desktop.SCHEMA, "pid": 1, "port": 2, "secret": "short"}):
        record.write_text(json.dumps(bad))
        assert desktop.read_record(record) is None
    record.write_text("{not json")
    assert desktop.read_record(record) is None


def test_the_instance_lock_is_released_when_its_holder_dies(tmp_path):
    lock = tmp_path / "instance.lock"
    holder = subprocess.Popen([sys.executable, "-c", (
        "import sys, time; from pathlib import Path; from crystalx_web.desktop import FileLock; "
        f"l = FileLock(Path({str(lock)!r})); assert l.acquire(5); print('held', flush=True); time.sleep(60)")],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert desktop.lock_is_held(lock)
    finally:
        holder.kill()  # the test's own child
        holder.wait()
    assert not desktop.lock_is_held(lock)


class _Impostor(http.server.BaseHTTPRequestHandler):
    proof = "0" * 64

    def do_GET(self):  # noqa: N802
        body = json.dumps({"app": desktop.APP_ID, "proof": self.proof}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_a_listener_without_the_secret_does_not_prove_identity():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Impostor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert not desktop.prove_identity(server.server_address[1], "x" * 43)
    finally:
        server.shutdown()
        server.server_close()
    bystander = Bystander()  # not HTTP at all
    try:
        assert not desktop.prove_identity(bystander.port, "x" * 43, timeout=1)
    finally:
        bystander.close()
    assert not desktop.prove_identity(free_port(), "x" * 43, timeout=1)  # nothing listening


def test_profiles_keep_data_out_of_the_application(tmp_path):
    dev = desktop.paths_for("dev", base=tmp_path)
    assert dev.runs_dir == tmp_path / "runs" and dev.desktop_dir == tmp_path / "runs" / "_web" / "desktop"
    app = desktop.paths_for("app", home=tmp_path, system="Darwin")
    assert app.runs_dir == tmp_path / "Library" / "Application Support" / "CrystalX Demo" / "runs"
    assert app.log_dir == tmp_path / "Library" / "Logs" / "CrystalX Demo"
    custom = desktop.paths_for("app", data_dir=tmp_path / "x")
    assert custom.profile == "custom" and custom.record == tmp_path / "x" / "desktop" / "instance.json"
    assert desktop.message_lang({}) == "zh" and desktop.message_lang({"LANG": "ja_JP.UTF-8"}) == "ja"
    assert desktop.message_lang({"LC_ALL": "en_US.UTF-8", "LANG": "zh_CN.UTF-8"}) == "en"


# ------------------------------------------------------------------------- real processes


def test_finder_like_start_reuse_and_stop(install):
    port = free_port()
    started = time.monotonic()
    first = install.run("launch", "--port", str(port))
    first_seconds = time.monotonic() - started
    assert first.returncode == 0, first.stdout + first.stderr
    record = install.record()
    url = f"http://127.0.0.1:{record['port']}/"
    assert record["url"] == url and install.opened_urls() == [url]
    assert "已就绪" in first.stdout and "停止本地服务" in first.stdout  # Finder gives no locale: Chinese
    assert desktop.prove_identity(record["port"], record["secret"])
    assert stat.S_IMODE(install.paths.record.stat().st_mode) == 0o600
    addresses = listening_addresses(record["port"])
    if addresses:  # Linux can show what is listening; it must be loopback only
        assert addresses == {"127.0.0.1"}
    page = desktop._DIRECT.open(url, timeout=10).read().decode("utf-8", "replace")
    assert "CrystalX" in page

    started = time.monotonic()
    second = install.run("launch", "--port", str(port))
    assert second.returncode == 0 and "复用" in second.stdout
    assert time.monotonic() - started < first_seconds  # no second server was started
    assert install.record()["pid"] == record["pid"] and install.opened_urls() == [url, url]

    stopped = install.run("stop")
    assert stopped.returncode == 0 and "已停止" in stopped.stdout
    assert not desktop.lock_is_held(install.paths.instance_lock) and install.record() is None
    assert install.paths.runs_dir.is_dir()  # records stay
    assert install.run("status").returncode == 3


class _Console:
    """The developer launcher in a pseudo-terminal, like the Terminal window a double click opens."""

    def __init__(self, install: Installation, port: int) -> None:
        import pty

        self.master, slave = pty.openpty()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "crystalx_web.desktop", "launch", "--data-dir", str(install.data),
             "--opener", str(install.opener), "--port", str(port), "--attach"],
            env=finder_env(install.home), cwd="/", stdin=slave, stdout=slave, stderr=slave,
            start_new_session=True)  # its own session and process group, as in a Terminal tab
        os.close(slave)
        self.output = b""
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and "关闭此窗口" not in self.text():
            self.read(0.3)
        assert "关闭此窗口" in self.text(), self.text()

    def read(self, wait: float) -> None:
        import select

        ready, _, _ = select.select([self.master], [], [], wait)
        if ready:
            try:
                self.output += os.read(self.master, 65536)
            except OSError:
                pass

    def text(self) -> str:
        return self.output.decode("utf-8", "replace")

    def hang_up(self) -> None:
        """Close the window: the terminal goes away and its process group gets SIGHUP."""
        os.close(self.master)
        os.killpg(self.process.pid, signal.SIGHUP)

    def interrupt(self) -> None:
        """Ctrl-C in the window."""
        os.killpg(self.process.pid, signal.SIGINT)

    def wait(self, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.process.poll() is None:
            if self.master >= 0:
                self.read(0.2)
            else:
                time.sleep(0.2)
        return self.process.poll()


def _stopped_within(install: Installation, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and desktop.lock_is_held(install.paths.instance_lock):
        time.sleep(0.2)
    return not desktop.lock_is_held(install.paths.instance_lock)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")
def test_closing_the_console_window_stops_the_server(install):
    console = _Console(install, free_port())
    record = install.record()
    assert desktop.prove_identity(record["port"], record["secret"])
    console.hang_up()
    console.master = -1
    assert _stopped_within(install, 30), "the server kept running after its console window was closed"
    assert console.wait(30) == 0 and install.record() is None
    assert install.paths.runs_dir.is_dir()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")
def test_ctrl_c_in_the_console_stops_the_server_and_says_so(install):
    console = _Console(install, free_port())
    console.interrupt()
    assert console.wait(30) == 0
    assert "正在停止服务" in console.text() and "服务已停止" in console.text()
    assert _stopped_within(install, 5) and install.record() is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")
def test_the_console_ends_when_the_server_is_stopped_from_the_page(install):
    console = _Console(install, free_port())
    record = install.record()
    assert desktop.request_stop(record["port"], record["secret"])  # what the page's stop control does
    assert console.wait(30) == 0 and "服务已结束" in console.text()
    assert _stopped_within(install, 5)


def test_a_program_on_the_preferred_port_is_left_alone(install):
    bystander = Bystander()
    try:
        result = install.run("launch", "--port", str(bystander.port))
        assert result.returncode == 0, result.stdout + result.stderr
        assert str(bystander.port) in result.stdout and "其他程序占用" in result.stdout
        assert install.record()["port"] != bystander.port
        assert bystander.alive()
    finally:
        install.cleanup()
        assert bystander.alive()
        bystander.close()


def test_a_stale_record_is_replaced_without_signalling_anyone(install):
    bystander = Bystander()
    stale_pid = dead_pid()
    install.paths.desktop_dir.mkdir(parents=True, exist_ok=True)
    desktop.write_record(install.paths.record, {
        "schema": desktop.SCHEMA, "pid": stale_pid, "port": bystander.port, "secret": "z" * 43,
        "url": f"http://127.0.0.1:{bystander.port}/",
    })
    try:
        result = install.run("launch", "--port", str(free_port()))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "过期的实例记录" in result.stdout and str(stale_pid) in result.stdout
        record = install.record()
        assert record["pid"] != stale_pid and record["port"] != bystander.port
        assert bystander.alive()
    finally:
        bystander.close()


def test_a_lock_holder_that_cannot_prove_itself_is_neither_reused_nor_stopped(install):
    bystander = Bystander()
    lock = install.paths.instance_lock
    holder = subprocess.Popen([sys.executable, "-c", (
        "import time; from pathlib import Path; from crystalx_web.desktop import FileLock; "
        f"l = FileLock(Path({str(lock)!r})); assert l.acquire(5); print('held', flush=True); time.sleep(120)")],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        desktop.write_record(install.paths.record, {
            "schema": desktop.SCHEMA, "pid": holder.pid, "port": bystander.port, "secret": "y" * 43,
            "url": f"http://127.0.0.1:{bystander.port}/",
        })
        result = install.run("launch", "--verify-timeout", "2")
        assert result.returncode == desktop.EXIT_UNVERIFIED, result.stdout + result.stderr
        assert "身份核验" in result.stdout
        assert holder.poll() is None and bystander.alive()  # nothing was signalled
        assert install.opened_urls() == []
        assert install.run("stop").returncode == desktop.EXIT_UNVERIFIED and holder.poll() is None
    finally:
        holder.kill()
        holder.wait()
        bystander.close()
        install.paths.record.unlink(missing_ok=True)


def test_a_poisoned_proxy_environment_does_not_break_the_start(install):
    dead = f"http://127.0.0.1:{free_port()}"
    env = finder_env(install.home, HTTP_PROXY=dead, HTTPS_PROXY=dead, ALL_PROXY=dead,
                     http_proxy=dead, https_proxy=dead, all_proxy=dead)
    result = install.run("launch", "--port", str(free_port()), env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert desktop.prove_identity(install.record()["port"], install.record()["secret"])


def test_a_missing_dependency_is_reported_readably_and_nothing_is_left_running(install, tmp_path):
    shadow = tmp_path / "shadow"
    (shadow / "gradio").mkdir(parents=True)
    (shadow / "gradio" / "__init__.py").write_text(
        "raise ImportError('simulated missing dependency: gradio')\n", encoding="utf-8")
    env = finder_env(install.home, PYTHONPATH=str(shadow))
    result = install.run("launch", "--port", str(free_port()), "--startup-timeout", "60", env=env)
    assert result.returncode == desktop.EXIT_FAILED
    assert "启动失败" in result.stdout and "simulated missing dependency: gradio" in result.stdout
    assert str(install.paths.server_log) in result.stdout
    assert not desktop.lock_is_held(install.paths.instance_lock) and install.record() is None


def test_an_unusable_data_folder_is_reported_not_crashed(tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    result = subprocess.run([sys.executable, "-m", "crystalx_web.desktop", "launch", "--data-dir", str(blocker),
                             "--no-browser"], env=finder_env(tmp_path / "home"), cwd="/",
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == desktop.EXIT_FAILED and "启动失败" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_the_code_location_can_be_read_only(install, tmp_path):
    """Like an application bundle: nothing may be written next to the code."""
    import shutil

    bundle = tmp_path / "Read Only 只读 Bundle"
    for package in ("crystalx_core", "crystalx_native_tools", "crystalx_web"):
        source = REPO_WEB.parent / package / "src" / package
        shutil.copytree(source, bundle / package, ignore=shutil.ignore_patterns("__pycache__"))
    before = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*"))
    for path in [bundle, *bundle.rglob("*")]:
        path.chmod(path.stat().st_mode & ~0o222)
    try:
        # The packaged app keeps byte code in the user's cache (-X pycache_prefix); the same policy
        # here. Permission bits alone do not stop root, so the comparison below is the real check.
        env = finder_env(install.home, PYTHONPATH=str(bundle), PYTHONPYCACHEPREFIX=str(tmp_path / "pycache"))
        result = install.run("launch", "--port", str(free_port()), env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert install.record()["python"] == sys.executable
        install.cleanup()
        after = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*"))
        assert after == before  # no __pycache__, logs or records inside
    finally:
        for path in [bundle, *bundle.rglob("*")]:
            path.chmod(path.stat().st_mode | 0o200)


def test_the_double_click_file_delegates_to_the_supported_entry(tmp_path):
    """The .command goes through scripts/crystalx.sh from any folder, whatever the path looks like."""
    checkout = tmp_path / "我的 项目 プロジェクト"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "crystalx_web").mkdir()
    calls = tmp_path / "calls.txt"
    (checkout / "scripts" / "crystalx.sh").write_text(
        f"#!/bin/sh\nprintf '%s|' \"$@\" >> '{calls}'\nprintf '\\n' >> '{calls}'\nexit ${{FAKE_EXIT:-0}}\n",
        encoding="utf-8")
    command = checkout / "crystalx_web" / COMMAND_FILE.name
    command.write_bytes(COMMAND_FILE.read_bytes())
    command.chmod(0o755)
    assert os.access(COMMAND_FILE, os.X_OK), "the delivered .command must be executable for Finder"
    env = finder_env(tmp_path / "home")
    done = subprocess.run([str(command)], env=env, cwd=str(tmp_path / "home"), capture_output=True, text=True,
                          timeout=30)
    assert done.returncode == 0 and calls.read_text().splitlines() == ["web|--desktop|--attach|"]
    failed = subprocess.run([str(command)], env={**env, "FAKE_EXIT": "2"}, cwd="/", input="\n",
                            capture_output=True, text=True, timeout=30)
    assert failed.returncode == 2 and "Press Return" in failed.stdout  # the window stays readable
    (checkout / "scripts" / "crystalx.sh").unlink()
    missing = subprocess.run([str(command)], env=env, cwd="/", input="\n", capture_output=True, text=True,
                             timeout=30)
    assert missing.returncode == 2 and "Cannot find the project start script" in missing.stderr
