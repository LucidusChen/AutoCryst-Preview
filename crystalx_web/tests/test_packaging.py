"""The application build tool: locks, safe extraction, bundle checks and the start script.

No network and no real runtime: tiny synthetic inputs only. The full build and its launch are
acceptance steps recorded separately (they need the pinned runtime and the locked wheels).
"""

from __future__ import annotations

import importlib.util
import io
import os
import plistlib
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
_spec = importlib.util.spec_from_file_location("build_app", PACKAGING / "build_app.py")
build_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_app)

_mb_spec = importlib.util.spec_from_file_location(
    "machobuild_for_packaging", PACKAGING.parents[1] / "crystalx_native_tools" / "tests" / "machobuild.py")
mb = importlib.util.module_from_spec(_mb_spec)
_mb_spec.loader.exec_module(mb)


def _wheel(folder: Path, name: str, version: str) -> Path:
    path = folder / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}/__init__.py", "")
    return path


def test_the_lock_round_trips_and_rejects_a_tampered_wheel(tmp_path: Path):
    house = tmp_path / "wheels"
    house.mkdir()
    a = _wheel(house, "alpha_pkg", "1.0")
    _wheel(house, "Beta", "2.0")
    lock_file = tmp_path / "lock.txt"
    build_app.write_lock(house, lock_file)
    lock = build_app.read_lock(lock_file)
    assert set(lock) == {"alpha-pkg", "beta"} and lock["beta"][0] == "2.0"
    assert {w.name for w in build_app.verified_wheels(house, lock)} == {w.name for w in house.iterdir()}
    a.write_bytes(a.read_bytes() + b"tampered")
    with pytest.raises(build_app.BuildError, match="alpha-pkg"):
        build_app.verified_wheels(house, lock)


def test_the_pins_are_complete_for_the_macos_application():
    pins = build_app.load_pins("macos-arm64")
    assert len(pins["python"]["sha256"]) == 64 and pins["python"]["url"].startswith("https://")
    assert pins["min_macos"] == "14.0" and (PACKAGING / pins["lock"]).is_file()
    lock = build_app.read_lock(PACKAGING / pins["lock"])
    assert "gradio" in lock and lock["gradio"][0] == "6.30.0"
    assert not any(name.startswith("crystalx") for name in lock)  # CrystalX is built from this checkout


def test_archive_members_cannot_escape(tmp_path: Path):
    for member in ("../evil.txt", "/abs/evil.txt"):
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            data = b"x"
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        with pytest.raises(build_app.BuildError):
            build_app._safe_extract(archive, tmp_path / "out")
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("python/bin/escape")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../../etc/passwd"
        tar.addfile(info)
    with pytest.raises(build_app.BuildError, match="leaves the bundle"):
        build_app._safe_extract(archive, tmp_path / "out2")


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "CrystalX Demo.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    executable = bundle / "Contents" / "MacOS" / "CrystalX Demo"
    executable.write_bytes(mb.thin(mb.CPU_ARM64, minos=(13, 0, 0)))
    (bundle / "Contents" / "Resources").mkdir()
    return bundle


def test_bundle_checks_pass_for_a_clean_bundle_and_catch_every_forbidden_kind(tmp_path: Path):
    bundle = _bundle(tmp_path)
    clean = build_app.verify_bundle(bundle, target="macos-arm64", min_macos="14.0")
    assert clean["violations"] == [] and clean["macho_files"] == 1
    resources = bundle / "Contents" / "Resources"
    (resources / ".venv").mkdir()
    (resources / "crystalx_engine").mkdir()
    (resources / "shelxl").write_bytes(b"x")
    (resources / "case.hkl").write_text("1 0 0 10 1\n")
    (resources / "model.safetensors").write_bytes(b"0")
    (resources / "config.local.json").write_text("{}")
    (resources / "intel.so").write_bytes(mb.thin(mb.CPU_X86_64))
    (resources / "brew.so").write_bytes(mb.thin(mb.CPU_ARM64, libraries=("/opt/homebrew/lib/libfoo.dylib",)))
    (resources / "future.so").write_bytes(mb.thin(mb.CPU_ARM64, minos=(30, 0, 0)))
    (resources / "unsigned.so").write_bytes(mb.thin(mb.CPU_ARM64, signed=False))
    (resources / "outside").symlink_to("/etc")
    found = " | ".join(build_app.verify_bundle(bundle, target="macos-arm64", min_macos="14.0")["violations"])
    for fragment in (".venv", "crystalx_engine", "forbidden file: Contents/Resources/shelxl", "case.hkl",
                     "model.safetensors", "config.local.json", "no arm64 code", "/opt/homebrew", "needs macOS 30.0",
                     "unsigned.so", "symlink leaves the bundle"):
        assert fragment in found, fragment


def test_gradio_finds_its_seed_in_the_bundle_instead_of_writing_one(tmp_path: Path):
    """Gradio writes gradio/hash_seed.txt next to its code on first import; inside the app that
    would modify the bundle (found in the real double-click acceptance on a writable copy)."""
    bundle = _bundle(tmp_path)
    site_packages = bundle / "Contents" / "Resources" / "python" / "lib" / "python3.12" / "site-packages"
    (site_packages / "gradio").mkdir(parents=True)
    (site_packages / "gradio" / "__init__.py").write_text("")
    found = " | ".join(build_app.verify_bundle(bundle, target="macos-arm64", min_macos="14.0")["violations"])
    assert "gradio/hash_seed.txt is missing" in found
    assert build_app.preseed_gradio(site_packages) is True
    seed = (site_packages / "gradio" / "hash_seed.txt").read_text()
    assert len(seed) == 32 and int(seed, 16) >= 0  # the format Gradio itself writes (uuid4().hex)
    assert build_app.verify_bundle(bundle, target="macos-arm64", min_macos="14.0")["violations"] == []
    build_app.preseed_gradio(site_packages)
    assert (site_packages / "gradio" / "hash_seed.txt").read_text() == seed  # an existing seed is kept
    assert build_app.preseed_gradio(tmp_path / "no-gradio-here") is False


def test_the_info_plist_never_allows_rosetta_and_keeps_no_dock_icon(tmp_path: Path):
    contents = tmp_path / "Contents"
    contents.mkdir()
    build_app.write_info_plist(contents, version="0.1.1", min_macos="14.0")
    info = plistlib.loads((contents / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "CrystalX Demo" and info["CFBundlePackageType"] == "APPL"
    assert info["LSRequiresNativeExecution"] is True and info["LSArchitecturePriority"] == ["arm64"]
    assert info["LSUIElement"] is True and info["LSMinimumSystemVersion"] == "14.0"
    assert info["CFBundleShortVersionString"] == "0.1.1"


def test_the_build_records_its_commit_and_any_uncommitted_sources(tmp_path: Path):
    if subprocess.run(["sh", "-c", "command -v git"], capture_output=True).returncode != 0:
        pytest.skip("git is not installed")
    repo = tmp_path / "checkout"
    source = repo / "crystalx_web" / "src" / "crystalx_web"
    source.mkdir(parents=True)
    (source / "desktop.py").write_text("A = 1\n")
    (repo / "notes.txt").write_text("outside the packages\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    for command in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
        subprocess.run(["git", "-C", str(repo), *command], check=True, env=env, capture_output=True)
    clean = build_app.source_state(repo)
    assert len(clean["commit"]) == 40 and clean["clean"] is True and clean["uncommitted_files"] == []
    (source / "desktop.py").write_text("A = 2\n")
    (source / "new_module.py").write_text("B = 1\n")
    (repo / "notes.txt").write_text("changed, but not part of the app\n")
    dirty = build_app.source_state(repo)
    assert dirty["clean"] is False
    assert dirty["uncommitted_files"] == ["crystalx_web/src/crystalx_web/desktop.py",
                                          "crystalx_web/src/crystalx_web/new_module.py"]
    assert not (repo / ".git" / "index.lock").exists()
    assert build_app.source_state(tmp_path / "not-a-checkout")["commit"] is None


def test_the_start_script_is_valid_and_writes_nothing_into_the_bundle():
    script = (PACKAGING / "launch.sh").read_text(encoding="utf-8")
    assert subprocess.run(["sh", "-n", str(PACKAGING / "launch.sh")]).returncode == 0
    assert '-I -X "pycache_prefix=$cache/pycache"' in script
    assert "--profile app" in script and "unset PYTHONPATH PYTHONHOME" in script
    for forbidden in ("pip ", "curl", "xattr", "spctl", "networksetup", "softwareupdate"):
        assert forbidden not in script, forbidden


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX")
def test_the_launcher_source_builds_and_finds_its_bundle(tmp_path: Path):
    cc = next((c for c in ("cc", "clang", "gcc") if subprocess.run(["sh", "-c", f"command -v {c}"],
                                                                       capture_output=True).returncode == 0), None)
    if cc is None or sys.platform == "darwin":
        pytest.skip("needs a C compiler for this host (the macOS build is checked in the acceptance step)")
    bundle = tmp_path / "Some 应用 App" / "Contents"
    (bundle / "MacOS").mkdir(parents=True)
    (bundle / "Resources").mkdir()
    (bundle / "Resources" / "launch.sh").write_text("#!/bin/sh\necho \"launched from $0\"\n")
    executable = bundle / "MacOS" / "CrystalX Demo"
    subprocess.run([cc, "-O2", "-o", str(executable), str(PACKAGING / "launcher.c")], check=True)
    done = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert done.returncode == 0 and f"launched from {bundle / 'Resources' / 'launch.sh'}" in done.stdout
