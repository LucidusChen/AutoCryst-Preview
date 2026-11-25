# CrystalX Demo.app: the relocatable desktop launcher of the Web demo

`CrystalX Demo.app` starts the existing Web interface on `127.0.0.1` and opens it in the
browser. It is **not** a native macOS GUI (the Swift/Metal GUI is deferred). Unlike
`crystalx_web/CrystalX Web Demo (developer).command`, it does not use this checkout or its `.venv`.
It carries its own Python runtime and libraries, so a copy can be moved anywhere, including to
another Mac. It needs macOS 14 or later on Apple Silicon.

## What is inside

```
CrystalX Demo.app/Contents/
  Info.plist           LSUIElement (no Dock icon), LSRequiresNativeExecution, arm64 only
  MacOS/CrystalX Demo  native arm64 starter compiled from launcher.c (ad-hoc linker signature)
  Resources/launch.sh  start logic: bundled Python, user folders, readable errors
  Resources/python/    CPython 3.12.14 from python-build-standalone (tests/Tk/IDLE/pip removed)
                       + the locked wheels and crystalx-core, crystalx-native-tools, crystalx-web
  Resources/licenses/  licence files of every bundled distribution, THIRD_PARTY_NOTICES.txt
  Resources/BUILD-INFO.json  inputs with SHA-256 digests, versions, check results, and the Git
                       commit of the CrystalX sources with every uncommitted source file listed
```

The CrystalX packages are copied from the working tree as it is at build time, including changes
that are not committed yet. `BUILD-INFO.json` names them (`crystalx_source.uncommitted_files`);
rebuild after committing for a bundle that matches a commit exactly.

The app never contains any of the following, and the build checks for them:

* a development environment;
* the upstream engine or training tree;
* model weights;
* SHELX/PLATON programs or sources;
* private data, configuration or run records.

AI inference is not part of the app.

## Building

Inputs are pinned in `pins.json` (runtime archive, SHA-256 from the pbs-installer package) and
`requirements-macos-arm64.txt` (52 wheels, each with its SHA-256; versions as tested). The build is
file assembly only: it never runs anything from the bundle and works on macOS or Linux.

```sh
# downloads exactly the pinned runtime and the locked wheels (explicit network use)
python3 crystalx_web/packaging/build_app.py --download --compile-launcher
# or offline, from files at hand
python3 crystalx_web/packaging/build_app.py --python-archive cpython-3.12.14+20260929-aarch64-apple-darwin-install_only_stripped.tar.gz \
    --wheelhouse wheels/ --launcher "CrystalX Demo"
```

* **Starter executable.** `--compile-launcher` uses the Command Line Tools' clang on a Mac, or the
  `ziglang` package elsewhere (`zig cc -target aarch64-macos.13.0`).
* **Output.** The result is `crystalx_web/dist/CrystalX Demo.app`, which Git ignores, with
  `CrystalX Demo.app.BUILD-INFO.json` beside it.
* **Checks.** Every Mach-O file must contain arm64 code, have a readable signature container and
  require no newer macOS than 14.0. It may link only system libraries or files inside the bundle.
  The build fails if any forbidden content is found.
* **Linux stand-in.** `--target linux-x86_64 --python-dir <standalone python>` builds the same
  layout on Linux. It is used only to test the start logic, not as a product.

## Running and stopping

Double-click the app:

* it starts the server, or reuses a healthy one after proving its identity, and opens the page;
* the first-use route is the labelled synthetic demo;
* to stop, use "停止本地服务" at the top of the page.

Data stays in the user's folders, never in the app:

* `~/Library/Application Support/CrystalX Demo/` holds runs and records;
* `~/Library/Logs/CrystalX Demo/` holds `launcher.log` and `server.log`;
* `~/Library/Caches/CrystalX Demo/` holds Python byte code.

On failure the app shows an alert with the reason and the log path. It does not run pip, download
anything, install Rosetta 2, or change proxy, DNS, quarantine or Gatekeeper settings.

## Trust prompts

The app is **not notarized** and has no Developer ID signature. A copy created locally by this
build is not quarantined and opens directly. A copy downloaded or received from elsewhere carries
macOS's quarantine attribute, and Gatekeeper will ask the user to confirm it (right-click → Open, or
System Settings → Privacy & Security). Removing that attribute is the user's decision; the app never
does it.
