"""Project-local launcher, deliberately runnable with Python -S and stdlib only.

On macOS Python ignores hidden .pth files, even when every package remains installed. An
external operation has repeatedly hidden this project's virtual environment. Its origin is
unknown. Repair import-path flags and validate explicit installed development links so child
imports survive recurrence, without changing source, home, system or user startup hooks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys


class RuntimeProblem(RuntimeError):
    pass


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def repair_hidden_flags(repo: Path) -> tuple[Path, ...]:
    """Repair the fixed .venv's import path, refusing environments outside this project.

    Do not walk package trees or browser caches. Hidden flags on directories do not prevent
    normal traversal; the immediate .pth files are the ones Python explicitly skips.
    """
    repo = repo.resolve()
    venv = repo / ".venv"
    if venv.is_symlink() or not venv.is_dir() or not (venv / "pyvenv.cfg").is_file():
        raise RuntimeProblem(".venv must be a real project-local virtual environment (not a symlink)")
    candidates = [venv, venv / "bin", venv / "lib"]
    for site_packages in sorted((venv / "lib").glob("python*/site-packages")):
        candidates.extend([site_packages.parent, site_packages])
        candidates.extend(sorted(site_packages.glob("*.pth")))
    if len(candidates) == 3:
        raise RuntimeProblem("No project site-packages found; use the documented bootstrap")
    changed = []
    hidden = getattr(stat, "UF_HIDDEN", 0)
    for candidate in candidates:
        try:
            if candidate.is_symlink() or not _within(candidate.resolve(), venv):
                raise RuntimeProblem(f"Refusing to repair a symlink or path outside .venv: {candidate}")
            info = candidate.stat()
            flags = getattr(info, "st_flags", 0)
            if hidden and flags & hidden:
                os.chflags(candidate, flags & ~hidden, follow_symlinks=False)
                changed.append(candidate)
        except (OSError, ValueError) as exc:
            raise RuntimeProblem(f"Cannot inspect/repair {candidate}: {exc}") from exc
    return tuple(changed)


def clean_environment(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["GRADIO_ANALYTICS_ENABLED"] = "False"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(repo / ".venv" / "playwright-browsers"))
    return env


def ensure_project_import_links(repo: Path) -> None:
    """Stabilize installed editable packages without a hidden-sensitive .pth dependency.

    These explicit, project-local development links are discoverable through ordinary
    site-packages imports, including child interpreters and -I. Require the existing editable
    install and its exact source path first; never mask a missing install or replace a wheel.
    No source directory is inserted into sys.path and no sitecustomize is replaced.
    """
    repo = repo.resolve()
    for site_packages in sorted((repo / ".venv/lib").glob("python*/site-packages")):
        for _command, module, _distribution in PACKAGES.values():
            source_dir = repo / module / "src"
            source = source_dir / module
            metadata = tuple(site_packages.glob(f"__editable__.{module}-*.pth"))
            if not metadata:
                # A package not installed here stays missing; self-check supplies the diagnosis.
                continue
            if len(metadata) != 1 or metadata[0].is_symlink():
                raise RuntimeProblem(f"Ambiguous or symlinked editable metadata for {module}")
            content = metadata[0].read_text(encoding="utf-8").strip()
            if content != str(source_dir) or not (source / "__init__.py").is_file():
                raise RuntimeProblem(f"Editable metadata/source for {module} does not match this project")
            if not _within(source.resolve(), repo):
                raise RuntimeProblem(f"Project source for {module} resolves outside the repository")
            target = site_packages / module
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise RuntimeProblem(f"Existing import link for {module} points elsewhere; preserved")
                continue
            if target.exists():
                raise RuntimeProblem(f"Existing installed package {module} is not our development link; preserved")
            try:
                target.symlink_to(source, target_is_directory=True)
            except FileExistsError:
                if not target.is_symlink() or target.resolve() != source.resolve():
                    raise RuntimeProblem(f"Concurrent installed-package change for {module}; preserved")


PACKAGES = {
    "core": ("crystalx", "crystalx_core", "crystalx-core"),
    "web": ("crystalx-web", "crystalx_web", "crystalx-web"),
    "apple": ("crystalx-apple", "crystalx_engine_apple_silicon", "crystalx-engine-apple-silicon"),
    "tools": ("crystalx-tools", "crystalx_native_tools", "crystalx-native-tools"),
}


def _run(repo: Path, argv: list[str], env: dict[str, str]) -> int:
    try:
        return subprocess.call(argv, cwd=repo, env=env)
    except OSError as exc:
        print(f"Cannot start {argv[0]}: {exc}", file=sys.stderr)
        return 2


def _exec(argv: list[str], env: dict[str, str]) -> int:
    """Give long-running applications normal terminal signal handling, including Ctrl-C."""
    try:
        os.execve(argv[0], argv, env)
    except OSError as exc:
        print(f"Cannot start {argv[0]}: {exc}", file=sys.stderr)
        return 2


def self_check(repo: Path, env: dict[str, str]) -> int:
    """Check actual clean imports and installed commands, with no solver or tensor execution."""
    probe = (
        "import importlib, importlib.metadata, json; from pathlib import Path; "
        "items=" + repr(list(PACKAGES.values())) + "; "
        "print(json.dumps([{\"command\": c, \"version\": importlib.metadata.version(d), "
        "\"source\": str(Path(importlib.import_module(m).__file__).resolve())} for c,m,d in items], indent=2))"
    )
    code = _run(repo, [str(repo / ".venv/bin/python"), "-I", "-c", probe], env)
    if code:
        print("Installed imports failed after flag repair. Check installation; no path override was used.",
              file=sys.stderr)
        return code
    for command, _module, _distribution in PACKAGES.values():
        try:
            result = subprocess.run([str(repo / ".venv/bin" / command), "--help"],
                                    cwd=repo, env=env, capture_output=True, text=True)
        except OSError as exc:
            print(f"Installed entry point {command} is missing or unreadable: {exc}", file=sys.stderr)
            return 2
        if result.returncode:
            print(result.stderr or result.stdout, file=sys.stderr)
            return result.returncode
        print(f"{command}: installed entry point OK")
    print("Local startup check passed. No crystallographic program or model was run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reliable project-local CrystalX startup")
    parser.add_argument("command", nargs="?", default="check", choices=["check", "core", "web", "apple", "tools", "python", "test"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        changed = repair_hidden_flags(repo)
        ensure_project_import_links(repo)
    except (RuntimeProblem, OSError) as exc:
        print(f"Startup blocked: {exc}", file=sys.stderr)
        return 2
    if changed:
        print(f"Repaired hidden flags on {len(changed)} project-environment path(s).", file=sys.stderr)
    env = clean_environment(repo)
    if args.command == "check":
        return self_check(repo, env)
    if args.command in PACKAGES:
        os.chdir(repo)
        return _exec([str(repo / ".venv/bin" / PACKAGES[args.command][0]), *args.args], env)
    python = str(repo / ".venv/bin/python")
    if args.command == "python":
        os.chdir(repo)
        return _exec([python, *args.args], env)
    suites = args.args or ["all"]
    if len(suites) != 1 or suites[0] not in ("core", "web", "apple", "tools", "startup", "all"):
        parser.error("test accepts one of: core, web, apple, tools, startup, all")
    test_paths = {
        "startup": "tests/test_local_runtime.py",
        "core": "crystalx_core/tests",
        "tools": "crystalx_native_tools/tests",
        "web": "crystalx_web/tests",
        "apple": "crystalx_engine_apple_silicon/tests",
    }
    selected = list(test_paths) if suites[0] == "all" else suites
    for name in selected:
        test_env = dict(env)
        if name == "apple":
            test_env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        code = _run(repo, [python, "-m", "pytest", "-q", test_paths[name]], test_env)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
