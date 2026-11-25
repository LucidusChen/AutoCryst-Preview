"""Helpers for the desktop-start acceptance tests: real processes, synthetic everything else.

Every server here is started through the real launcher (``python -m crystalx_web.desktop``) in a
Finder-like environment: a minimal PATH, no locale, no activated environment, the working directory
``/`` and a data folder whose path contains spaces, Chinese and Japanese. The tool folder is a
tripwire: fake SHELX/PLATON programs that only leave a marker file when something starts them, so
"no proprietary program was launched" is checked, not assumed.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from crystalx_web import desktop

FINDER_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
TOOL_NAMES = ("shelxt", "shelxl", "shelxs", "shelxc", "shelxd", "shelxe", "anode", "platon", "platon_special")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def finder_env(home: Path, **extra: str) -> dict[str, str]:
    """What a Finder double-click gives a process: no shell profile, no locale, no venv, no proxy."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "tmp").mkdir(exist_ok=True)
    env = {"PATH": FINDER_PATH, "HOME": str(home), "TMPDIR": str(home / "tmp"), "USER": "crystalx-test"}
    env.update(extra)
    return env


class Installation:
    """One data folder with an opener that records URLs and a tripwire tool folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data = root / "数据 データ folder"
        self.data.mkdir(parents=True)
        self.home = root / "home"
        self.marker = root / "TRIPWIRE-a-program-was-started.txt"
        self.toolbox = root / "工具 ツール box"
        self.toolbox.mkdir()
        for name in TOOL_NAMES:
            program = self.toolbox / name
            program.write_text(f"#!/bin/sh\necho \"$0 $*\" >> '{self.marker}'\nexit 1\n", encoding="utf-8")
            program.chmod(0o755)
        (self.data / "config.local.json").write_text(json.dumps({
            "toolbox": str(self.toolbox), "include_olex2": False, "include_known_locations": False,
        }), encoding="utf-8")
        self.opened = root / "opened.txt"
        self.opener = root / "open-url.sh"
        self.opener.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$1\" >> '{self.opened}'\n", encoding="utf-8")
        self.opener.chmod(0o755)
        self.paths = desktop.paths_for(data_dir=self.data)

    def run(self, action: str, *args: str, env: dict[str, str] | None = None, timeout: float = 180.0,
            python: str | None = None) -> subprocess.CompletedProcess:
        command = [python or sys.executable, "-m", "crystalx_web.desktop", action, "--data-dir", str(self.data)]
        if action == "launch":
            command += ["--opener", str(self.opener)]
        command += list(args)
        return subprocess.run(command, env=env if env is not None else finder_env(self.home), cwd="/",
                              capture_output=True, text=True, timeout=timeout)

    def record(self) -> dict | None:
        return desktop.read_record(self.paths.record)

    def opened_urls(self) -> list[str]:
        return self.opened.read_text(encoding="utf-8").split() if self.opened.exists() else []

    def cleanup(self) -> None:
        """Stop a server this test started: the authenticated stop first, the test's own child last."""
        record = self.record()
        if record is None:
            return
        desktop.request_stop(record["port"], record["secret"])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and desktop.lock_is_held(self.paths.instance_lock):
            time.sleep(0.2)
        if desktop.lock_is_held(self.paths.instance_lock):  # pragma: no cover - only if the stop failed
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class Bystander:
    """An unrelated program listening on a port. It must still be there, untouched, afterwards."""

    def __init__(self, port: int | None = None, banner: bytes = b"not-crystalx\n") -> None:
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", port or 0))
        self.socket.listen(16)
        self.port = self.socket.getsockname()[1]
        self.banner = banner
        self.connections = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self.socket.settimeout(0.2)
        while not self._stop.is_set():
            try:
                connection, _ = self.socket.accept()
            except OSError:
                continue
            self.connections += 1
            try:
                connection.sendall(self.banner)
            finally:
                connection.close()

    def alive(self) -> bool:
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as client:
            return client.recv(64) == self.banner

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.socket.close()


def dead_pid() -> int:
    """The pid of a process that has already exited (a stale record's pid)."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def listening_addresses(port: int) -> set[str]:
    """Local addresses with a listening socket on ``port`` (Linux /proc only; empty elsewhere)."""
    found: set[str] = set()
    for table, family in (("/proc/net/tcp", "v4"), ("/proc/net/tcp6", "v6")):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            local, state = fields[1], fields[3]
            address, hex_port = local.rsplit(":", 1)
            if state == "0A" and int(hex_port, 16) == port:  # 0A = LISTEN
                if family == "v4":
                    found.add(socket.inet_ntoa(bytes.fromhex(address)[::-1]))
                else:
                    found.add(address)
    return found
