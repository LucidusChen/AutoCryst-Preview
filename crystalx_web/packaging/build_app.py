"""Build ``CrystalX Demo.app``: the relocatable, Web-backed desktop application.

This is a launcher for the existing Web interface, not a native GUI. The build is pure file
assembly and runs on macOS or Linux; nothing inside the bundle is executed while building::

    1. runtime       python-build-standalone CPython (pinned archive, SHA-256 checked)
    2. dependencies  the locked wheels (SHA-256 checked), installed with ``pip --target`` for the
                     target platform -- not copied from any development environment
    3. CrystalX      crystalx-core, crystalx-native-tools, crystalx-web, copied from this checkout
    4. start         a native main executable (``launcher.c``), ``launch.sh`` and ``Info.plist``
    5. notices       every bundled distribution's licence files and an index
    6. checks        Mach-O architecture, signature container, minimum macOS and linkage of every
                     binary; no development environment, engine, weights, tools, data or configs

Typical use on a Mac (the only network access is the explicit ``--download``)::

    python3 crystalx_web/packaging/build_app.py --download --compile-launcher

or fully offline from files already at hand::

    python3 crystalx_web/packaging/build_app.py --python-archive cpython-...tar.gz \\
        --wheelhouse wheels/ --launcher "CrystalX Demo"

The result goes to ``crystalx_web/dist/`` (ignored by Git), with ``BUILD-INFO.json`` beside it.
The application is not notarized and not signed with a Developer ID: macOS may ask for
confirmation the first time a downloaded copy is opened.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as _dt
import hashlib
import json
import os
import plistlib
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGES = ("crystalx_core", "crystalx_native_tools", "crystalx_web")
APP_EXECUTABLE = "CrystalX Demo"
BUNDLE_ID = "local.crystalx.demo"

#: Never inside the application: development environments, the upstream engine and training tree,
#: model weights, proprietary programs, private data, local configuration and run records.
FORBIDDEN_DIRS = {".venv", "crystalx_engine", "crystalx_engine_apple_silicon", "inference_repo",
                  "realdata_examples", "toolbox", "runs", "paper_ref", ".git"}
FORBIDDEN_FILES = {"shelxt", "shelxl", "shelxs", "shelxc", "shelxd", "shelxe", "anode", "ciftab", "shredcif",
                   "platon", "platon_special", "check.def", "config.local.json"}
FORBIDDEN_SUFFIXES = {".pt", ".ckpt", ".safetensors", ".onnx", ".ins", ".hkl", ".res", ".fcf", ".cif", ".lst"}
ALLOWED_DYLIB_PREFIXES = ("/usr/lib/", "/System/Library/", "@rpath/", "@loader_path/", "@executable_path/")
#: Parts of the standard runtime the Web demo never uses (tests, IDLE, Tk); removed to save space.
RUNTIME_TRIM = ("test", "idlelib", "tkinter", "turtledemo", "ensurepip", "turtle.py")


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def say(text: str) -> None:
    print(text, flush=True)


# ------------------------------------------------------------------------------------- inputs


def load_pins(target: str) -> dict:
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    try:
        return pins["targets"][target]
    except KeyError:
        raise BuildError(f"unknown target {target!r}") from None


def read_lock(path: Path) -> dict[str, tuple[str, str]]:
    """``name -> (version, sha256)`` from a pip requirements file with ``--hash`` entries."""
    locked: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s]+)\s+--hash=sha256:([0-9a-f]{64})", line)
        if not match:
            raise BuildError(f"{path.name}: unexpected line {line!r}")
        locked[_normalise(match.group(1))] = (match.group(2), match.group(3))
    return locked


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_name_version(filename: str) -> tuple[str, str]:
    parts = filename[:-4].split("-")
    return _normalise(parts[0]), parts[1]


def write_lock(wheelhouse: Path, out: Path) -> None:
    """Maintenance: a hash-pinned requirements file for the wheels in ``wheelhouse``."""
    lines = ["# CrystalX Demo.app dependency lock: exact files, checked by SHA-256 before use.",
             f"# Generated {_dt.date.today().isoformat()} from {len(list(wheelhouse.glob('*.whl')))} wheels."]
    for wheel in sorted(wheelhouse.glob("*.whl"), key=lambda p: p.name.lower()):
        name, version = _wheel_name_version(wheel.name)
        lines.append(f"{name}=={version} --hash=sha256:{sha256(wheel)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verified_wheels(wheelhouse: Path, lock: dict[str, tuple[str, str]] | None) -> list[Path]:
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise BuildError(f"no wheels in {wheelhouse}")
    if lock is None:
        return wheels
    chosen = {}
    for wheel in wheels:
        name, version = _wheel_name_version(wheel.name)
        if name in lock and lock[name][0] == version and sha256(wheel) == lock[name][1]:
            chosen[name] = wheel
    missing = sorted(set(lock) - set(chosen))
    if missing:
        raise BuildError("wheels missing or not matching the lock: " + ", ".join(missing))
    return [chosen[name] for name in sorted(chosen)]


def download(url: str, destination: Path, expected: str) -> Path:
    say(f"download: {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256(destination)
    if actual != expected:
        destination.unlink()
        raise BuildError(f"{destination.name}: SHA-256 {actual} does not match the pinned {expected}")
    return destination


def download_wheels(pins: dict, lock_file: Path, destination: Path) -> None:
    target = pins["pip_target"]
    command = [sys.executable, "-m", "pip", "download", "--dest", str(destination), "--require-hashes",
               "--no-deps", "--only-binary=:all:", "--python-version", target["python_version"],
               "--implementation", target["implementation"], "-r", str(lock_file)]
    for platform_tag in target["platforms"]:
        command += ["--platform", platform_tag]
    say("download: the locked wheels from the Python Package Index")
    subprocess.run(command, check=True)


# ------------------------------------------------------------------------------------ assembly


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract without letting any member or link escape ``destination``."""
    root = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise BuildError(f"unsafe archive member {member.name!r}")
            if member.issym():
                target = (root / name.parent / member.linkname).resolve()
            elif member.islnk():
                target = (root / member.linkname).resolve()
            else:
                continue
            if target != root and root not in target.parents:
                raise BuildError(f"archive link leaves the bundle: {member.name} -> {member.linkname}")
        try:
            tar.extractall(destination, filter="data")
        except TypeError:  # Python without extraction filters; the members were checked above
            tar.extractall(destination)


def install_runtime(resources: Path, *, archive: Path | None, python_dir: Path | None, version: str) -> Path:
    runtime = resources / "python"
    if archive is not None:
        with tempfile.TemporaryDirectory() as scratch:
            _safe_extract(archive, Path(scratch))
            source = Path(scratch) / "python"
            if not (source / "bin").is_dir():
                raise BuildError(f"{archive.name} does not contain python/bin")
            shutil.move(str(source), str(runtime))
    elif python_dir is not None:
        shutil.copytree(python_dir, runtime, symlinks=True)
    else:
        raise BuildError("give --python-archive, --python-dir or --download")
    stdlib = next(runtime.glob("lib/python3.*"), None)
    if stdlib is None or not (runtime / "bin" / "python3").exists():
        raise BuildError("the runtime has no bin/python3 or lib/python3.X")
    for name in RUNTIME_TRIM:
        target = stdlib / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for pattern in ("lib-dynload/_tkinter*", "site-packages/pip*", "site-packages/setuptools*",
                    "site-packages/_distutils_hack", "site-packages/distutils-precedence.pth"):
        for path in stdlib.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for pattern in ("lib/tcl*", "lib/tk*", "lib/itcl*", "lib/thread*", "lib/libtcl*", "lib/libtk*",
                    "include", "share", "lib/pkgconfig"):
        for path in runtime.glob(pattern):
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    for script in (runtime / "bin").iterdir():  # keep the interpreter, drop pip/idle/2to3 wrappers
        if not script.name.startswith("python"):
            script.unlink()
    return stdlib / "site-packages"


def _project_name_version(package: str) -> tuple[str, str]:
    """Distribution name and version from pyproject.toml, without importing or building anything."""
    pyproject = (REPO / package / "pyproject.toml").read_text(encoding="utf-8")
    name = re.search(r'^name\s*=\s*"([^"]+)"', pyproject, re.M)
    static = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    if name is None:
        raise BuildError(f"{package}/pyproject.toml has no project name")
    if static:
        return name.group(1), static.group(1)
    attr = re.search(r'version\s*=\s*\{\s*attr\s*=\s*"([\w.]+)"', pyproject)
    if attr is None:
        raise BuildError(f"{package}/pyproject.toml: cannot find the version")
    module, _, variable = attr.group(1).rpartition(".")
    base = REPO / package / "src" / Path(*module.split("."))
    source = base / "__init__.py" if base.is_dir() else base.with_suffix(".py")
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return name.group(1), str(ast.literal_eval(node.value))
    raise BuildError(f"{source}: no {variable}")


def install_crystalx_from_source(site_packages: Path) -> list[dict]:
    """Copy the three pure-Python packages from this checkout, with minimal installed metadata.

    No build backend, network or import is involved, so the build also works offline and with an
    old setuptools. ``importlib.metadata`` sees the right names and versions inside the app.
    """
    installed = []
    for package in PACKAGES:
        name, version = _project_name_version(package)
        source = REPO / package / "src" / package
        target = site_packages / package
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                            encoding="utf-8")
        (dist_info / "INSTALLER").write_text("crystalx-app-build\n", encoding="utf-8")
        records, digest = [], hashlib.sha256()
        for file in sorted(p for p in target.rglob("*") if p.is_file()):
            data = file.read_bytes()
            digest.update(file.relative_to(site_packages).as_posix().encode() + b"\0" + data)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append(f"{file.relative_to(site_packages).as_posix()},sha256={encoded},{len(data)}")
        records += [f"{dist_info.name}/METADATA,,", f"{dist_info.name}/INSTALLER,,", f"{dist_info.name}/RECORD,,"]
        (dist_info / "RECORD").write_text("\n".join(records) + "\n", encoding="utf-8")
        installed.append({"distribution": name, "version": version, "source_sha256": digest.hexdigest()})
    return installed


def source_state(repo: Path = REPO) -> dict:
    """Which commit the CrystalX sources came from, and which of their files differ from it.

    Read-only Git queries (``--no-optional-locks``: not even an index refresh is written). The app is
    built from the working tree, so uncommitted changes are listed rather than hidden."""
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args],
                                  capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "note": "not a Git checkout or Git unavailable"}
    status = git("status", "--porcelain", "--untracked-files=all", "--",
                 *(f"{package}/src" for package in PACKAGES)) or ""
    changed = sorted(line[3:] for line in status.splitlines() if len(line) > 3)
    return {"commit": commit.strip(), "uncommitted_files": changed, "clean": not changed}


GRADIO_SEED = "gradio/hash_seed.txt"


def preseed_gradio(site_packages: Path) -> bool:
    """Create the file Gradio would otherwise write into its own package folder on first import.

    Gradio salts the names of its cache files with a random seed that it keeps in
    ``gradio/hash_seed.txt``, next to its code, and creates that file the first time it is imported.
    In the application that folder is inside the bundle, so the build writes the seed (random, per
    build) and the running app only reads it. An existing seed is kept."""
    seed_file = site_packages / GRADIO_SEED
    if not seed_file.parent.is_dir():
        return False
    if not seed_file.exists():
        seed_file.write_text(secrets.token_hex(16), encoding="utf-8")
    return True


def install_wheels(site_packages: Path, wheels: list[Path], pins: dict) -> None:
    target = pins["pip_target"]
    command = [sys.executable, "-m", "pip", "install", "--target", str(site_packages), "--no-deps",
               "--no-compile", "--no-index", "--only-binary=:all:", "--upgrade",
               "--python-version", target["python_version"], "--implementation", target["implementation"]]
    for platform_tag in target["platforms"]:
        command += ["--platform", platform_tag]
    command += [str(w) for w in wheels]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    scripts = site_packages / "bin"  # console scripts written for the build machine: not used
    if scripts.is_dir():
        shutil.rmtree(scripts)


def compile_launcher(target: str, destination: Path, pins: dict) -> str:
    source = HERE / pins["launcher"]["source"]
    if target == "macos-arm64":
        clang = Path("/Library/Developer/CommandLineTools/usr/bin/clang")
        xcode = Path("/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang")
        if sys.platform == "darwin" and (clang.exists() or xcode.exists()):
            compiler = [str(clang if clang.exists() else xcode), *pins["launcher"]["clang_flags"]]
        else:
            try:
                import ziglang  # noqa: F401
            except ImportError:
                raise BuildError("no compiler: install the Command Line Tools, or the 'ziglang' package, "
                                 "or pass a prebuilt --launcher") from None
            compiler = [sys.executable, "-m", "ziglang", "cc", "-target", pins["launcher"]["zig_target"], "-O2"]
    else:
        compiler = [pins["launcher"].get("cc", "cc"), "-O2"]
    subprocess.run([*compiler, "-o", str(destination), str(source)], check=True)
    destination.chmod(0o755)
    return " ".join(Path(compiler[0]).name if i == 0 else part for i, part in enumerate(compiler))


def write_info_plist(contents: Path, *, version: str, min_macos: str | None) -> None:
    info = {
        "CFBundleDevelopmentRegion": "zh_CN",
        "CFBundleDisplayName": "CrystalX Demo",
        "CFBundleExecutable": APP_EXECUTABLE,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "CrystalX Demo",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSUIElement": True,  # no Dock icon: the window is the browser; stop from the page
        "LSArchitecturePriority": ["arm64"],
        "LSRequiresNativeExecution": True,  # never start this application under Rosetta 2
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "Local demonstration build; not notarized.",
    }
    if min_macos:
        info["LSMinimumSystemVersion"] = min_macos
    with open(contents / "Info.plist", "wb") as handle:
        plistlib.dump(info, handle)


def collect_licences(site_packages: Path, runtime: Path, resources: Path) -> list[dict]:
    root = resources / "licenses"
    root.mkdir()
    index = []
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        metadata = (dist_info / "METADATA").read_text(encoding="utf-8", errors="replace")
        name = re.search(r"^Name: (.+)$", metadata, re.M)
        version = re.search(r"^Version: (.+)$", metadata, re.M)
        licence = re.search(r"^License-Expression: (.+)$", metadata, re.M) or re.search(r"^License: (.+)$", metadata, re.M)
        files = [p for p in dist_info.rglob("*") if p.is_file() and re.match(r"(?i)(licen[cs]e|copying|notice|authors)", p.name)]
        folder = root / dist_info.name.removesuffix(".dist-info")
        copied = []
        for file in files:
            relative = file.relative_to(dist_info)
            (folder / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, folder / relative)
            copied.append(str(relative))
        index.append({"distribution": name.group(1).strip() if name else dist_info.name,
                      "version": version.group(1).strip() if version else "?",
                      "licence": (licence.group(1).strip() if licence else "see files")[:200],
                      "files": copied})
    python_licence = next(runtime.glob("lib/python3.*/LICENSE.txt"), None)
    if python_licence is not None:
        shutil.copy2(python_licence, root / "Python-LICENSE.txt")
    lines = ["CrystalX Demo -- third-party notices", "",
             "The application bundles the components below. Their licence files are in this folder.",
             "The Python runtime is a python-build-standalone build of CPython (PSF License); it also",
             "contains libraries such as OpenSSL, SQLite, zlib, bzip2, xz, libffi and expat under their",
             "own licences, documented by the python-build-standalone project for that release.", ""]
    for item in index:
        lines.append(f"{item['distribution']} {item['version']}: {item['licence']}")
    (root / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


# ------------------------------------------------------------------------------------- checks


def verify_bundle(bundle: Path, *, target: str, min_macos: str | None) -> dict:
    """Static checks of the finished bundle. Returns a report; raises nothing, lists violations."""
    sys.path.insert(0, str(REPO / "crystalx_native_tools" / "src"))
    from crystalx_native_tools.macho import read_macho

    violations: list[str] = []
    machos = 0
    biggest_min = (0,)
    files = 0
    size = 0
    for path in sorted(bundle.rglob("*")):
        relative = path.relative_to(bundle).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(bundle.resolve()):
                violations.append(f"symlink leaves the bundle: {relative}")
            continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRS:
                violations.append(f"forbidden folder: {relative}")
            if (path / "pyvenv.cfg").exists():
                violations.append(f"a virtual environment: {relative}")
            if path.name == "gradio" and (path / "__init__.py").is_file() and not (path / "hash_seed.txt").is_file():
                violations.append(f"{relative}/hash_seed.txt is missing: Gradio would write it into the bundle "
                                  "when the app starts")
            continue
        files += 1
        size += path.stat().st_size
        if path.name in FORBIDDEN_FILES:
            violations.append(f"forbidden file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"data or weights file: {relative}")
        if path.suffix == ".pth" and path.stat().st_size > 1_000_000:
            violations.append(f"large .pth (weights?): {relative}")
        if target != "macos-arm64":
            continue
        with open(path, "rb") as handle:
            magic = handle.read(4)
        if magic not in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
            continue
        macho = read_macho(path)
        if macho is None:
            continue
        machos += 1
        slice_ = macho.slice_for("arm64")
        if slice_ is None:
            violations.append(f"no arm64 code: {relative} ({', '.join(macho.arches)})")
            continue
        problems = list(macho.problems) + list(slice_.problems)
        if problems:
            violations.append(f"malformed: {relative}: {'; '.join(problems)}")
        sig = slice_.code_signature
        if not sig.present or sig.note:
            violations.append(f"arm64 code without a readable signature container: {relative}")
        if slice_.min_os:
            version = tuple(int(x) for x in slice_.min_os.split("."))
            biggest_min = max(biggest_min, version)
            if min_macos and version > tuple(int(x) for x in min_macos.split(".")):
                violations.append(f"needs macOS {slice_.min_os} > {min_macos}: {relative}")
        for library in slice_.libraries:
            if not library.name.startswith(ALLOWED_DYLIB_PREFIXES):
                violations.append(f"links outside the bundle and the system: {relative} -> {library.name}")
    return {"files": files, "bytes": size, "macho_files": machos,
            "highest_minimum_macos": ".".join(map(str, biggest_min)) if machos else None,
            "violations": violations}


# --------------------------------------------------------------------------------------- build


def build(args: argparse.Namespace) -> int:
    pins = load_pins(args.target)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / pins["bundle_name"]
    if bundle.exists():
        shutil.rmtree(bundle)
    contents = bundle / "Contents"
    resources = contents / "Resources"
    (contents / "MacOS").mkdir(parents=True)
    resources.mkdir()
    work = Path(tempfile.mkdtemp(prefix="crystalx-app-build-"))
    info: dict = {"schema": "crystalx.app-build/1", "target": args.target,
                  "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                  "built_on": f"{sys.platform} / Python {sys.version.split()[0]}", "notarized": False}
    try:
        archive = Path(args.python_archive).resolve() if args.python_archive else None
        if args.download and archive is None and "url" in pins["python"]:
            archive = download(pins["python"]["url"], work / pins["python"]["archive"], pins["python"]["sha256"])
        if archive is not None and "sha256" in pins["python"]:
            actual = sha256(archive)
            if actual != pins["python"]["sha256"]:
                raise BuildError(f"{archive.name}: SHA-256 {actual} is not the pinned {pins['python']['sha256']}")
        info["python"] = {**pins["python"], "used": archive.name if archive else str(args.python_dir)}
        site_packages = install_runtime(resources, archive=archive,
                                        python_dir=Path(args.python_dir).resolve() if args.python_dir else None,
                                        version=pins["python"]["version"])
        say(f"runtime: {pins['python']['distribution']}")

        lock_file = Path(args.lock).resolve() if args.lock else (HERE / pins["lock"] if pins["lock"] else None)
        wheelhouse = Path(args.wheelhouse).resolve() if args.wheelhouse else work / "wheels"
        if args.download and not args.wheelhouse:
            if lock_file is None:
                raise BuildError("this target has no lock; give --wheelhouse")
            wheelhouse.mkdir()
            download_wheels(pins, lock_file, wheelhouse)
        lock = read_lock(lock_file) if lock_file else None
        dependencies = verified_wheels(wheelhouse, lock)
        install_wheels(site_packages, dependencies, pins)
        preseed_gradio(site_packages)
        crystalx = install_crystalx_from_source(site_packages)
        info["dependencies"] = [{"wheel": w.name, "sha256": sha256(w)} for w in dependencies]
        info["crystalx"] = crystalx
        info["crystalx_source"] = source_state()
        say(f"dependencies: {len(dependencies)} locked wheels; CrystalX: "
            + ", ".join(f"{c['distribution']} {c['version']}" for c in crystalx))
        state = info["crystalx_source"]
        if state.get("commit"):
            say(f"CrystalX sources: commit {state['commit'][:12]}"
                + ("" if state["clean"] else f" + {len(state['uncommitted_files'])} uncommitted file(s)"))

        executable = contents / "MacOS" / APP_EXECUTABLE
        if args.launcher:
            shutil.copy2(args.launcher, executable)
            executable.chmod(0o755)
            info["launcher"] = {"prebuilt": Path(args.launcher).name}
        elif args.compile_launcher:
            info["launcher"] = {"compiled_with": compile_launcher(args.target, executable, pins)}
        else:
            raise BuildError("give --launcher FILE or --compile-launcher")
        info["launcher"].update({"sha256": sha256(executable), "source_sha256": sha256(HERE / "launcher.c")})
        shutil.copy2(HERE / "launch.sh", resources / "launch.sh")
        (resources / "launch.sh").chmod(0o755)
        web_version = next(c["version"] for c in crystalx if c["distribution"] == "crystalx-web")
        write_info_plist(contents, version=web_version, min_macos=pins["min_macos"])
        info["licences"] = collect_licences(site_packages, resources / "python", resources)
        report = verify_bundle(bundle, target=args.target, min_macos=pins["min_macos"])
        info["verification"] = report
        (resources / "BUILD-INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        (output / f"{pins['bundle_name']}.BUILD-INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        for path in [bundle, *bundle.rglob("*")]:  # readable by everyone, writable by the builder only
            if not path.is_symlink():
                mode = path.stat().st_mode
                path.chmod((mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH) & ~(stat.S_IWGRP | stat.S_IWOTH))
        say(f"bundle: {bundle} ({report['files']} files, {report['bytes'] / 1e6:.0f} MB, "
            f"{report['macho_files']} Mach-O files)")
        if report["violations"]:
            for violation in report["violations"][:50]:
                say(f"  VIOLATION: {violation}")
            return 3
        say("checks passed. Not notarized; no Developer ID signature.")
        return 0
    except (BuildError, subprocess.CalledProcessError, OSError) as exc:
        say(f"build failed: {exc}")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", choices=("macos-arm64", "linux-x86_64"), default="macos-arm64")
    parser.add_argument("--python-archive", help="python-build-standalone install_only archive (pinned digest)")
    parser.add_argument("--python-dir", help="an extracted standalone runtime (Linux stand-in only)")
    parser.add_argument("--wheelhouse", help="folder with the locked wheels")
    parser.add_argument("--lock", help="hash-pinned requirements file (default: the target's lock)")
    parser.add_argument("--download", action="store_true",
                        help="download the pinned runtime and the locked wheels (explicit network use)")
    parser.add_argument("--launcher", help="prebuilt native main executable")
    parser.add_argument("--compile-launcher", action="store_true")
    parser.add_argument("--output", default=str(REPO / "crystalx_web" / "dist"))
    parser.add_argument("--write-lock", nargs=2, metavar=("WHEELHOUSE", "OUT"),
                        help="maintenance: write a hash-pinned lock for a wheel folder and exit")
    args = parser.parse_args(argv)
    if args.write_lock:
        write_lock(Path(args.write_lock[0]), Path(args.write_lock[1]))
        return 0
    if args.target == "macos-arm64" and args.python_dir:
        parser.error("the macOS application needs the pinned --python-archive (or --download)")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
