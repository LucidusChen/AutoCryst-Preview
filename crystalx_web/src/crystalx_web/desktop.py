"""Double-click start of the local Web demo: one owned server per data folder, reused while healthy.

Entry points (through ``crystalx-web``, which the project's supported start script runs)::

    crystalx-web --desktop             start or reuse the owned server, then open it in the browser
    crystalx-web --desktop --attach    the same, and keep this terminal as its console
    crystalx-web --desktop-stop        stop the owned server; nothing else is touched
    crystalx-web --desktop-status      report whether it runs, where, and where its logs are

How ownership is established
    The server is a separate, detached process ("serve"). For its whole life it holds an exclusive
    lock on ``<desktop>/instance.lock`` and it writes ``<desktop>/instance.json`` (mode 0600) with
    its pid, port and a random secret. A later start that finds the lock taken sends the listener
    a fresh challenge; only an HMAC of it under that secret counts as "our server". A listener that
    cannot answer is never trusted and never stopped -- the next free port is used instead. A record
    whose lock is free is stale (its process is gone) and is replaced. The only process this module
    ever signals is a server child it started itself, in the same call, that failed to come up.

Networking
    127.0.0.1 only. Loopback is exempted from proxies inside this process and its children
    (``NO_PROXY``/``no_proxy``, existing entries kept); system proxy, DNS and firewall settings are
    not touched. Every wait has a deadline. Nothing here runs pip, downloads anything or changes
    quarantine/Gatekeeper attributes.

This module imports only the standard library at import time; Gradio is imported by the server.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

try:  # POSIX only; the desktop start targets macOS (and Linux for tests)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "APP_ID",
    "APP_NAME",
    "DesktopPaths",
    "paths_for",
    "loopback_proxy_exemption",
    "port_is_free",
    "choose_port",
    "prove_identity",
    "read_record",
    "lock_is_held",
    "launch",
    "stop",
    "status",
    "serve",
    "main",
]

APP_ID = "crystalx-web-desktop"
APP_NAME = "CrystalX Demo"
SCHEMA = "crystalx.desktop-instance/1"
LOOPBACK = "127.0.0.1"
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")
DEFAULT_PORT = 7860
PORT_SPAN = 100
HEALTH_PATH = "/crystalx-desktop/health"
STOP_PATH = "/crystalx-desktop/stop"
DEFAULT_STARTUP_TIMEOUT = 120.0
STOP_TIMEOUT = 60.0
LOG_ROTATE_BYTES = 5 * 1024 * 1024
_NONCE = re.compile(r"[0-9a-f]{32}")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_BUSY = 3
EXIT_UNVERIFIED = 4
EXIT_TIMEOUT = 5

# ------------------------------------------------------------------------------------- messages

_MESSAGES: dict[str, tuple[str, str, str]] = {
    "starting": ("正在启动 CrystalX 本地 Web 演示……", "CrystalX ローカル Web デモを起動しています……",
                 "Starting the CrystalX local Web demo..."),
    "waiting": ("等待服务就绪（最长 {seconds} 秒）", "サービスの準備を待っています（最大 {seconds} 秒）",
                "Waiting for the server (at most {seconds} s)"),
    "ready": ("已就绪：{url}", "準備完了：{url}", "Ready: {url}"),
    "reused": ("已在运行，复用现有服务：{url}", "すでに起動しています。既存のサービスを使います：{url}",
               "Already running; reusing it: {url}"),
    "opened": ("已在浏览器中打开。", "ブラウザで開きました。", "Opened in your browser."),
    "open_failed": ("无法自动打开浏览器，请手动打开上面的地址。", "ブラウザを自動で開けませんでした。上のアドレスを手動で開いてください。",
                    "Could not open the browser automatically; open the address above yourself."),
    "port_taken": ("端口 {port} 已被其他程序占用（未作任何改动），改用其他端口。",
                   "ポート {port} は別のプログラムが使用中です（何も変更していません）。別のポートを使います。",
                   "Port {port} is used by another program (left untouched); using another port."),
    "stale": ("发现过期的实例记录（进程 {pid} 已不存在），已替换。", "古いインスタンス記録を見つけました（プロセス {pid} は存在しません）。置き換えました。",
              "Found a stale instance record (process {pid} no longer exists); replaced it."),
    "unverified": (
        "已有进程持有本服务的实例锁（记录中的进程 {pid}，端口 {port}），但它没有通过身份核验，因此既未复用也未停止。请稍后重试；如果问题持续，查看日志：{log}",
        "このサービスのインスタンスロックを保持しているプロセス（記録上のプロセス {pid}、ポート {port}）が本人確認に応答しないため、再利用も停止もしていません。しばらくして再試行してください。続く場合はログを確認してください：{log}",
        "A process holds this server's instance lock (recorded process {pid}, port {port}) but did not prove its identity, so it was neither reused nor stopped. Try again shortly; if it persists, see the log: {log}",
    ),
    "failed": ("启动失败：{reason}\n日志：{log}", "起動に失敗しました：{reason}\nログ：{log}", "Start failed: {reason}\nLog: {log}"),
    "timeout": ("服务在 {seconds} 秒内没有就绪，已停止本次启动的进程。日志：{log}",
                "サービスが {seconds} 秒以内に準備できなかったため、今回起動したプロセスを停止しました。ログ：{log}",
                "The server was not ready within {seconds} s; the process started for it was stopped. Log: {log}"),
    "exited": ("服务进程提前退出（退出码 {code}）。", "サービスのプロセスが早期に終了しました（終了コード {code}）。",
               "The server process exited early (exit code {code})."),
    "how_to_stop_attach": (
        "此窗口是服务的控制台：关闭此窗口或按 Ctrl-C 即停止服务；也可以在页面右上角点击“停止本地服务”。运行记录保存在：{runs}",
        "このウィンドウがサービスのコンソールです。ウィンドウを閉じるか Ctrl-C で停止します。ページ右上の「ローカルサービスを停止」でも停止できます。実行記録の保存先：{runs}",
        "This window is the server's console: close it or press Ctrl-C to stop the server, or use \"Stop local server\" at the top of the page. Run records are kept in: {runs}",
    ),
    "how_to_stop": (
        "服务在后台运行。停止方法：页面右上角“停止本地服务”。运行记录保存在：{runs}；日志：{log}",
        "サービスはバックグラウンドで動作しています。停止するにはページ右上の「ローカルサービスを停止」を使います。実行記録：{runs}；ログ：{log}",
        "The server runs in the background. To stop it, use \"Stop local server\" at the top of the page. Run records: {runs}; log: {log}",
    ),
    "stopping": ("正在停止服务……", "サービスを停止しています……", "Stopping the server..."),
    "stopped": ("服务已停止。运行记录已保留。", "サービスを停止しました。実行記録は保持されています。", "The server has stopped. Run records are kept."),
    "stop_slow": ("已请求停止，但服务在 {seconds} 秒内仍未退出。日志：{log}",
                  "停止を要求しましたが、{seconds} 秒以内に終了しませんでした。ログ：{log}",
                  "Stop was requested but the server had not exited after {seconds} s. Log: {log}"),
    "not_running": ("没有正在运行的服务。", "起動中のサービスはありません。", "No server is running."),
    "running": ("正在运行：{url}（进程 {pid}，启动于 {started}）", "起動中：{url}（プロセス {pid}、開始 {started}）",
                "Running: {url} (process {pid}, started {started})"),
    "busy": ("另一个启动过程正在进行，请稍后再试。", "別の起動処理が進行中です。しばらくしてから再試行してください。",
             "Another start is in progress; try again shortly."),
    "server_ended": ("服务已结束（可能是在页面上被停止）。", "サービスは終了しました（ページから停止された可能性があります）。",
                     "The server has ended (perhaps stopped from the page)."),
    "unsupported": ("此平台不支持桌面启动（需要 macOS 或 Linux）。", "このプラットフォームではデスクトップ起動を利用できません（macOS または Linux が必要）。",
                    "Desktop start is not supported on this platform (macOS or Linux required)."),
}


def message_lang(environ: dict[str, str] | None = None) -> str:
    """zh/ja/en from the locale environment; Finder starts apps without one, which gives zh."""
    environ = os.environ if environ is None else environ
    for key in ("CRYSTALX_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = (environ.get(key) or "").lower()
        if value.startswith("ja"):
            return "ja"
        if value.startswith("en"):
            return "en"
        if value.startswith("zh"):
            return "zh"
    return "zh"


def msg(key: str, lang: str | None = None, **values: object) -> str:
    index = {"zh": 0, "ja": 1, "en": 2}.get(lang or message_lang(), 0)
    return _MESSAGES[key][index].format(**values)


# ---------------------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class DesktopPaths:
    """Where one desktop installation keeps its runs, its instance record and its logs."""

    profile: str
    """``dev`` (a repository checkout), ``app`` (the packaged application) or ``custom``."""

    base: Path
    runs_dir: Path
    desktop_dir: Path
    log_dir: Path
    config_file: Path
    toolbox: Path | None = None

    @property
    def record(self) -> Path:
        return self.desktop_dir / "instance.json"

    @property
    def instance_lock(self) -> Path:
        return self.desktop_dir / "instance.lock"

    @property
    def launch_lock(self) -> Path:
        return self.desktop_dir / "launch.lock"

    @property
    def server_log(self) -> Path:
        return self.log_dir / "server.log"

    def ensure(self) -> None:
        for directory in (self.runs_dir, self.desktop_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


def paths_for(
    profile: str = "dev",
    data_dir: Path | str | None = None,
    *,
    base: Path | str | None = None,
    home: Path | str | None = None,
    system: str | None = None,
) -> DesktopPaths:
    """The folders for a profile. Nothing is written inside an application bundle.

    * ``dev``: the checkout the start script ran from (``base``, default the working directory),
      using its ignored ``runs/`` exactly like ``crystalx-web``.
    * ``app``: per-user locations -- on macOS ``~/Library/Application Support/CrystalX Demo`` and
      ``~/Library/Logs/CrystalX Demo``.
    * ``data_dir``: everything under one folder (tests, or a user's explicit choice).
    """
    if data_dir is not None:
        root = Path(os.path.abspath(Path(data_dir).expanduser()))
        return DesktopPaths("custom", root, root / "runs", root / "desktop", root / "logs",
                            root / "config.local.json", None)
    system = system or platform.system()
    home_dir = Path(home).expanduser() if home is not None else Path.home()
    if profile == "app":
        if system == "Darwin":
            root = home_dir / "Library" / "Application Support" / APP_NAME
            logs = home_dir / "Library" / "Logs" / APP_NAME
        else:  # the Linux stand-in of the packaged layout: XDG data and state folders
            data = os.environ.get("XDG_DATA_HOME")
            state = os.environ.get("XDG_STATE_HOME")
            root = (Path(data) if data else home_dir / ".local" / "share") / "crystalx-demo"
            logs = (Path(state) if state else home_dir / ".local" / "state") / "crystalx-demo"
        return DesktopPaths("app", root, root / "runs", root / "desktop", logs, root / "config.local.json", None)
    if profile != "dev":
        raise ValueError(f"unknown desktop profile {profile!r}")
    root = Path(os.path.abspath(Path(base).expanduser() if base is not None else Path.cwd()))
    runs = root / "runs"
    desktop = runs / "_web" / "desktop"
    toolbox = root / "toolbox" if (root / "toolbox").is_dir() else None
    return DesktopPaths("dev", root, runs, desktop, desktop / "logs", root / "config.local.json", toolbox)


# ----------------------------------------------------------------------------------- networking


def loopback_proxy_exemption(environ: dict[str, str]) -> dict[str, str]:
    """A copy of ``environ`` whose ``NO_PROXY``/``no_proxy`` also cover loopback.

    Existing exclusions are kept in their order. Proxy variables themselves are left as they are,
    so anything that does go to the network still uses the user's proxy. Only this process and its
    children see the change.
    """
    entries: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        for item in (environ.get(key) or "").split(","):
            item = item.strip()
            if item and item not in entries:
                entries.append(item)
    for name in LOOPBACK_NAMES:
        if name not in entries:
            entries.append(name)
    env = dict(environ)
    env["NO_PROXY"] = env["no_proxy"] = ",".join(entries)
    return env


def apply_loopback_proxy_exemption() -> None:
    os.environ.update({k: v for k, v in loopback_proxy_exemption(dict(os.environ)).items()
                       if k in ("NO_PROXY", "no_proxy")})


#: Talks to 127.0.0.1 only and never through any proxy, environment or system.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request_json(url: str, *, data: dict | None = None, timeout: float = 3.0) -> dict:
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="GET" if data is None else "POST",
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with _DIRECT.open(request, timeout=timeout) as response:
        payload = response.read(65536)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("not a JSON object")
    return value


def _proof(secret: str, purpose: str, nonce: str) -> str:
    return hmac.new(secret.encode("utf-8"), f"{APP_ID}|{purpose}|{nonce}".encode("utf-8"), hashlib.sha256).hexdigest()


def prove_identity(port: int, secret: str, *, timeout: float = 3.0) -> bool:
    """True only if the listener on 127.0.0.1:``port`` answers a fresh challenge with our secret."""
    nonce = secrets.token_hex(16)
    try:
        reply = _request_json(f"http://{LOOPBACK}:{int(port)}{HEALTH_PATH}?nonce={nonce}", timeout=timeout)
    except Exception:  # refused, timed out, not HTTP, not JSON: not proven
        return False
    return reply.get("app") == APP_ID and hmac.compare_digest(str(reply.get("proof", "")),
                                                               _proof(secret, "health", nonce))


def request_stop(port: int, secret: str, *, timeout: float = 5.0) -> bool:
    nonce = secrets.token_hex(16)
    try:
        reply = _request_json(f"http://{LOOPBACK}:{int(port)}{STOP_PATH}",
                              data={"nonce": nonce, "proof": _proof(secret, "stop", nonce)}, timeout=timeout)
    except Exception:
        return False
    return reply.get("stopping") is True


def port_is_free(port: int) -> bool:
    """Nothing answers on 127.0.0.1:``port`` and it can be bound without sharing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((LOOPBACK, port)) == 0:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((LOOPBACK, port))  # deliberately without SO_REUSEADDR: never share a port
        except OSError:
            return False
    return True


def choose_port(preferred: int = DEFAULT_PORT, span: int = PORT_SPAN,
                is_free: Callable[[int], bool] = port_is_free, avoid: Sequence[int] = ()) -> tuple[int, list[int]]:
    """The first free port from ``preferred`` on, and the occupied ports that were skipped."""
    skipped: list[int] = []
    for port in range(preferred, min(preferred + span, 65536)):
        if port in avoid:
            continue
        if is_free(port):
            return port, skipped
        skipped.append(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:  # let the system pick one
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1], skipped


# ------------------------------------------------------------------------------- records, locks


def read_record(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return None
    if not isinstance(value.get("pid"), int) or not isinstance(value.get("port"), int):
        return None
    if not isinstance(value.get("secret"), str) or len(value["secret"]) < 32:
        return None
    return value


def write_record(path: Path, record: dict) -> None:
    """Atomic write, readable by this user only (the secret proves the server's identity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    os.replace(temporary, path)


def remove_record_if(path: Path, predicate: Callable[[dict], bool]) -> None:
    current = read_record(path)
    if current is not None and predicate(current):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class FileLock:
    """An exclusive ``flock`` that the system releases when the holder exits, even on a crash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def acquire(self, timeout: float = 0.0) -> bool:
        if fcntl is None:  # pragma: no cover
            raise RuntimeError(msg("unsupported"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600), "r+b", buffering=0)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    handle.close()
                    return False
                time.sleep(0.1)

    def release(self) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None


def lock_is_held(path: Path) -> bool:
    """Whether another process holds the lock. A missing lock file means nobody does."""
    if not path.exists():
        return False
    probe = FileLock(path)
    if probe.acquire(0.0):
        probe.release()
        return False
    return True


def _pid_alive(pid: int) -> bool:
    """Existence only (signal 0 sends nothing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------------------- server


def _routes(secret: str, stop_event: threading.Event, used_nonces: set[str]) -> list:
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    no_store = {"Cache-Control": "no-store"}

    async def health(request):
        nonce = request.query_params.get("nonce", "")
        if not _NONCE.fullmatch(nonce):
            return JSONResponse({"app": APP_ID, "error": "challenge required"}, status_code=400, headers=no_store)
        return JSONResponse({"app": APP_ID, "pid": os.getpid(), "proof": _proof(secret, "health", nonce)},
                            headers=no_store)

    async def stop_route(request):
        try:
            body = await request.json()
        except Exception:
            body = None
        nonce = body.get("nonce") if isinstance(body, dict) else None
        proof = body.get("proof") if isinstance(body, dict) else None
        if not (isinstance(nonce, str) and _NONCE.fullmatch(nonce) and isinstance(proof, str)) or nonce in used_nonces \
                or not hmac.compare_digest(proof, _proof(secret, "stop", nonce)):
            return JSONResponse({"app": APP_ID, "error": "not authorised"}, status_code=403, headers=no_store)
        used_nonces.add(nonce)
        stop_event.set()
        return JSONResponse({"app": APP_ID, "stopping": True}, headers=no_store)

    return [Route(HEALTH_PATH, health, methods=["GET"]), Route(STOP_PATH, stop_route, methods=["POST"])]


def _log_line(text: str) -> None:
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {text}", flush=True)


def serve(
    paths: DesktopPaths,
    *,
    port: int | None = None,
    toolbox: Path | None = None,
    tools_config: Path | None = None,
    allow_translation: bool = False,
    lock_timeout: float = 2.0,
) -> int:
    """Run the owned server until it is told to stop. Returns an exit code."""
    paths.ensure()
    lock = FileLock(paths.instance_lock)
    if not lock.acquire(lock_timeout):
        _log_line("another CrystalX desktop server holds the instance lock; this process exits")
        return EXIT_BUSY
    apply_loopback_proxy_exemption()
    stop_event = threading.Event()
    secret = secrets.token_urlsafe(32)
    pid = os.getpid()

    def request_shutdown(*_args) -> None:
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_shutdown)

    service = app = None
    try:
        from crystalx_web.launcher import build_service_and_settings, prepare_environment

        prepare_environment(paths.runs_dir)
        import gradio  # noqa: F401 - after the environment above is set

        from crystalx_web.app import build_app, launch_options

        def stop_soon() -> None:  # lets the page receive its answer before the server closes
            threading.Timer(1.0, stop_event.set).start()

        page_lang = os.environ.get("CRYSTALX_LANG", "")
        service, settings, notes = build_service_and_settings(
            paths.runs_dir,
            toolbox=toolbox if toolbox is not None else paths.toolbox,
            tools_config=tools_config if tools_config is not None else paths.config_file,
            allow_translation=allow_translation,
            desktop_stop=stop_soon,
            quick_start=True,
            default_lang=page_lang if page_lang in ("zh", "ja", "en") else None,
        )
        for note in notes:
            _log_line(note)
        app = build_app(service, settings)
        app.queue(default_concurrency_limit=4)
        routes = _routes(secret, stop_event, set())
        chosen = None
        tried: list[int] = []
        for _attempt in range(5):
            candidate, skipped = choose_port(port or DEFAULT_PORT, avoid=tried)
            for occupied in skipped:
                if occupied not in tried:
                    _log_line(f"port {occupied} is in use by another program; left untouched")
            tried.extend(skipped)
            try:
                app.launch(**launch_options(LOOPBACK, candidate), prevent_thread_lock=True, quiet=True,
                           app_kwargs={"routes": routes})
                chosen = int(app.server_port or candidate)
                break
            except OSError as exc:
                _log_line(f"could not bind 127.0.0.1:{candidate} ({exc}); trying another port")
                tried.append(candidate)
        if chosen is None:
            _log_line("no port could be bound on 127.0.0.1")
            return EXIT_FAILED
        url = f"http://{LOOPBACK}:{chosen}/"
        write_record(paths.record, {
            "schema": SCHEMA,
            "app": APP_ID,
            "pid": pid,
            "port": chosen,
            "url": url,
            "secret": secret,
            "profile": paths.profile,
            "runs_dir": str(paths.runs_dir),
            "log": str(paths.server_log),
            "python": sys.executable,
            "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        })
        _log_line(f"serving {url} (pid {pid}); runs in {paths.runs_dir}")
        while not stop_event.wait(0.5):
            pass
        _log_line("stop requested; cancelling active runs and closing the server")
        return EXIT_OK
    except Exception as exc:  # a readable reason in the log; the launcher shows the tail
        _log_line(f"server failed: {type(exc).__name__}: {exc}")
        return EXIT_FAILED
    finally:
        if service is not None:
            try:
                service.shutdown()
            except Exception as exc:  # pragma: no cover - best effort at exit
                _log_line(f"run service shutdown: {exc}")
        if app is not None:
            try:
                app.close()
            except Exception:  # pragma: no cover
                pass
        remove_record_if(paths.record, lambda record: record.get("pid") == pid and record.get("secret") == secret)
        lock.release()
        _log_line("server stopped")


# ------------------------------------------------------------------------------------- launcher


def _rotate(log: Path) -> None:
    try:
        if log.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(log, log.with_name(log.name + ".1"))
    except FileNotFoundError:
        pass


def _log_tail(log: Path, lines: int = 12) -> str:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _server_command(paths: DesktopPaths, *, port: int | None, toolbox: Path | None, tools_config: Path | None,
                    allow_translation: bool, python: str | None = None) -> list[str]:
    executable = python or sys.executable
    command = [executable]
    if sys.flags.isolated:
        command.append("-I")
    prefix = getattr(sys, "pycache_prefix", None)
    if prefix:
        command += ["-X", f"pycache_prefix={prefix}"]
    command += ["-m", "crystalx_web.desktop", "serve", "--profile", paths.profile]
    if paths.profile == "custom":
        command += ["--data-dir", str(paths.base)]
    elif paths.profile == "dev":
        command += ["--base", str(paths.base)]
    if port:
        command += ["--port", str(port)]
    if toolbox is not None:
        command += ["--toolbox", str(toolbox)]
    if tools_config is not None:
        command += ["--tools-config", str(tools_config)]
    if allow_translation:
        command.append("--allow-translation")
    return command


def _terminate_own_child(child: subprocess.Popen, grace: float = 10.0) -> None:
    """Stop a server this call started (its own session/process group) that failed to come up."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        child.wait(timeout=5)


def open_url(url: str, opener: str | None = None) -> bool:
    """Open the page: ``/usr/bin/open`` on macOS, ``xdg-open`` elsewhere, or an explicit program."""
    program = opener or os.environ.get("CRYSTALX_DESKTOP_OPENER")
    if program:
        command = [program, url]
    elif sys.platform == "darwin":
        command = ["/usr/bin/open", url]
    else:
        found = shutil.which("xdg-open")
        if not found:
            return False
        command = [found, url]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@dataclass
class _Existing:
    state: str  # none | stale | healthy | unverified
    record: dict | None = None


def _existing_instance(paths: DesktopPaths, *, verify_for: float = 15.0) -> _Existing:
    record = read_record(paths.record)
    if not lock_is_held(paths.instance_lock):
        return _Existing("stale" if paths.record.exists() else "none", record)
    deadline = time.monotonic() + verify_for
    while True:
        record = read_record(paths.record)
        if record is not None and prove_identity(record["port"], record["secret"]):
            return _Existing("healthy", record)
        if time.monotonic() >= deadline or not lock_is_held(paths.instance_lock):
            break
        time.sleep(0.5)
    if not lock_is_held(paths.instance_lock):
        return _Existing("stale" if paths.record.exists() else "none", read_record(paths.record))
    return _Existing("unverified", record)


def launch(
    paths: DesktopPaths,
    *,
    open_browser: bool = True,
    attach: bool = False,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    opener: str | None = None,
    port: int | None = None,
    toolbox: Path | None = None,
    tools_config: Path | None = None,
    allow_translation: bool = False,
    python: str | None = None,
    say: Callable[[str], None] = print,
    lang: str | None = None,
    verify_timeout: float = 15.0,
) -> int:
    """Start (or reuse) the owned server, wait until it proves itself, open it. Returns an exit code."""
    if fcntl is None:  # pragma: no cover
        say(msg("unsupported", lang))
        return EXIT_USAGE
    apply_loopback_proxy_exemption()
    try:
        paths.ensure()
    except OSError as exc:
        say(msg("failed", lang, reason=f"{exc.strerror or exc}: {exc.filename or paths.base}", log=paths.server_log))
        return EXIT_FAILED
    launch_lock = FileLock(paths.launch_lock)
    if not launch_lock.acquire(startup_timeout + 30.0):
        say(msg("busy", lang))
        return EXIT_TIMEOUT
    child: subprocess.Popen | None = None
    try:
        existing = _existing_instance(paths, verify_for=verify_timeout)
        if existing.state == "healthy":
            url = existing.record["url"]
            say(msg("reused", lang, url=url))
            if open_browser:
                say(msg("opened", lang) if open_url(url, opener) else msg("open_failed", lang))
            return EXIT_OK
        if existing.state == "unverified":
            record = existing.record or {}
            say(msg("unverified", lang, pid=record.get("pid", "?"), port=record.get("port", "?"), log=paths.server_log))
            return EXIT_UNVERIFIED
        if existing.state == "stale":
            say(msg("stale", lang, pid=(existing.record or {}).get("pid", "?")))
            remove_record_if(paths.record, lambda record: True)
            try:
                paths.record.unlink()
            except FileNotFoundError:
                pass
        say(msg("starting", lang))
        preferred = port or DEFAULT_PORT
        if not port_is_free(preferred):
            say(msg("port_taken", lang, port=preferred))
        _rotate(paths.server_log)
        with open(paths.server_log, "ab") as log:
            child = subprocess.Popen(
                _server_command(paths, port=port, toolbox=toolbox, tools_config=tools_config,
                                allow_translation=allow_translation, python=python),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True, cwd=str(paths.base),
                env=loopback_proxy_exemption(dict(os.environ)),
            )
        say(msg("waiting", lang, seconds=int(startup_timeout)))
        deadline = time.monotonic() + startup_timeout
        record = None
        while time.monotonic() < deadline:
            code = child.poll()
            if code is not None:
                say(msg("failed", lang, reason=msg("exited", lang, code=code), log=paths.server_log))
                tail = _log_tail(paths.server_log)
                if tail:
                    say(tail)
                return EXIT_FAILED
            candidate = read_record(paths.record)
            if candidate is not None and candidate.get("pid") == child.pid and \
                    prove_identity(candidate["port"], candidate["secret"], timeout=2.0):
                record = candidate
                break
            time.sleep(0.25)
        if record is None:
            _terminate_own_child(child)
            say(msg("timeout", lang, seconds=int(startup_timeout), log=paths.server_log))
            tail = _log_tail(paths.server_log)
            if tail:
                say(tail)
            return EXIT_TIMEOUT
    finally:
        launch_lock.release()

    url = record["url"]
    say(msg("ready", lang, url=url))
    if open_browser:
        say(msg("opened", lang) if open_url(url, opener) else msg("open_failed", lang))
    if not attach:
        say(msg("how_to_stop", lang, runs=paths.runs_dir, log=paths.server_log))
        return EXIT_OK
    say(msg("how_to_stop_attach", lang, runs=paths.runs_dir))
    return _attach(paths, record, child, say=say, lang=lang)


def _detach_console() -> None:
    """Point stdout and stderr at /dev/null: the console window is gone, and nothing (not even the
    interpreter's final flush) may fail because of that."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:  # pragma: no cover
        return
    for fd in (1, 2):
        try:
            os.dup2(devnull, fd)
        except OSError:  # pragma: no cover
            pass
    os.close(devnull)


def _attach(paths: DesktopPaths, record: dict, child: subprocess.Popen | None, *,
            say: Callable[[str], None], lang: str | None) -> int:
    """Keep the terminal as the server's console: follow its log; Ctrl-C or closing stops it.

    Closing the Terminal window hangs up the console: the launcher gets SIGHUP and every later write
    to the window fails. So the stop request goes out before anything is printed, and a console that
    can no longer be written to counts as closed (stop) instead of ending the launcher with an error
    and leaving the server running without its console."""
    stop_requested = threading.Event()

    def on_signal(*_args) -> None:
        stop_requested.set()

    def console(text: str) -> None:
        try:
            say(text)
        except (OSError, ValueError):  # the window was closed
            _detach_console()
            stop_requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, on_signal)
    try:
        position = paths.server_log.stat().st_size
    except OSError:
        position = 0
    while True:
        if stop_requested.is_set():
            request_stop(record["port"], record["secret"])
            console(msg("stopping", lang))
            return _wait_until_stopped(paths, child, say=console, lang=lang)
        ended = (child.poll() is not None) if child is not None else not lock_is_held(paths.instance_lock)
        position = _follow(paths.server_log, position, console)
        if ended:
            console(msg("server_ended", lang))
            return EXIT_OK
        stop_requested.wait(0.5)


def _follow(log: Path, position: int, say: Callable[[str], None]) -> int:
    try:
        with open(log, "rb") as handle:
            handle.seek(position)
            chunk = handle.read()
    except OSError:
        return position
    if chunk:
        text = chunk.decode("utf-8", errors="replace").rstrip("\n")
        if text:
            say(text)
    return position + len(chunk)


def _wait_until_stopped(paths: DesktopPaths, child: subprocess.Popen | None, *,
                        say: Callable[[str], None], lang: str | None, timeout: float = STOP_TIMEOUT) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gone = (child.poll() is not None) if child is not None else False
        if gone or not lock_is_held(paths.instance_lock):
            say(msg("stopped", lang))
            return EXIT_OK
        time.sleep(0.25)
    say(msg("stop_slow", lang, seconds=int(timeout), log=paths.server_log))
    return EXIT_TIMEOUT


def stop(paths: DesktopPaths, *, say: Callable[[str], None] = print, lang: str | None = None,
         timeout: float = STOP_TIMEOUT) -> int:
    """Stop the owned server after it proves its identity. Never signals any process."""
    if not lock_is_held(paths.instance_lock):
        if paths.record.exists():
            say(msg("stale", lang, pid=(read_record(paths.record) or {}).get("pid", "?")))
            try:
                paths.record.unlink()
            except FileNotFoundError:
                pass
        say(msg("not_running", lang))
        return EXIT_OK
    existing = _existing_instance(paths, verify_for=5.0)
    if existing.state != "healthy":
        record = existing.record or {}
        say(msg("unverified", lang, pid=record.get("pid", "?"), port=record.get("port", "?"), log=paths.server_log))
        return EXIT_UNVERIFIED
    say(msg("stopping", lang))
    if not request_stop(existing.record["port"], existing.record["secret"]):
        say(msg("unverified", lang, pid=existing.record.get("pid"), port=existing.record.get("port"),
                log=paths.server_log))
        return EXIT_UNVERIFIED
    return _wait_until_stopped(paths, None, say=say, lang=lang, timeout=timeout)


def status(paths: DesktopPaths, *, say: Callable[[str], None] = print, lang: str | None = None) -> int:
    existing = _existing_instance(paths, verify_for=3.0)
    if existing.state == "healthy":
        record = existing.record
        say(msg("running", lang, url=record["url"], pid=record["pid"], started=record.get("started_at", "?")))
        say(msg("how_to_stop", lang, runs=paths.runs_dir, log=paths.server_log))
        return EXIT_OK
    if existing.state == "unverified":
        record = existing.record or {}
        say(msg("unverified", lang, pid=record.get("pid", "?"), port=record.get("port", "?"), log=paths.server_log))
        return EXIT_UNVERIFIED
    if existing.state == "stale":
        say(msg("stale", lang, pid=(existing.record or {}).get("pid", "?")))
    say(msg("not_running", lang))
    return 3


# ----------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m crystalx_web.desktop",
                                     description="Desktop start of the CrystalX local Web demo.")
    parser.add_argument("action", choices=("launch", "serve", "stop", "status"))
    parser.add_argument("--profile", choices=("dev", "app", "custom"), default="dev")
    parser.add_argument("--base", type=Path, default=None, help="dev profile: the checkout (default: cwd)")
    parser.add_argument("--data-dir", type=Path, default=None, help="keep runs, records and logs here")
    parser.add_argument("--port", type=int, default=None, help=f"preferred port (default {DEFAULT_PORT})")
    parser.add_argument("--toolbox", type=Path, default=None)
    parser.add_argument("--tools-config", type=Path, default=None)
    parser.add_argument("--allow-translation", action="store_true")
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--opener", default=None, help="program that opens the URL (default: the system's)")
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    parser.add_argument("--verify-timeout", type=float, default=15.0,
                        help="seconds a running server gets to prove its identity (default 15)")
    return parser


def run_action(action: str, args: argparse.Namespace) -> int:
    if args.profile == "custom" and args.data_dir is None:
        print("--profile custom needs --data-dir", file=sys.stderr)
        return EXIT_USAGE
    paths = paths_for("dev" if args.profile == "custom" else args.profile, args.data_dir, base=args.base)
    if action == "serve":
        return serve(paths, port=args.port, toolbox=args.toolbox, tools_config=args.tools_config,
                     allow_translation=args.allow_translation)
    if action == "stop":
        return stop(paths)
    if action == "status":
        return status(paths)
    return launch(paths, open_browser=not args.no_browser, attach=args.attach, startup_timeout=args.startup_timeout,
                  opener=args.opener, port=args.port, toolbox=args.toolbox, tools_config=args.tools_config,
                  allow_translation=args.allow_translation, verify_timeout=args.verify_timeout)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        print("--port must be between 1 and 65535", file=sys.stderr)
        return EXIT_USAGE
    return run_action(args.action, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
