# crystalx-web

Local Web interface for the CrystalX/AutoCryst workflow core. It runs one **preliminary** solve:

upload INS/HKL → diagnosis and plan → SHELXT initial solution → bounded isotropic SHELXL
refinement → PLATON diagnostics → interactive 3-D model and downloads.

The result is a preliminary, unvalidated model. It can be inspected and exported; it is never
ranked, accepted or called validated, because the chemical-validity and coordination gates have no
calculator yet and stay `NOT_EVALUATED`.

## Install and launch

From the repository root:

```sh
sh crystalx_web/scripts/bootstrap_web.sh   # creates/repairs .venv, installs core, native tools and web, runs --check
.venv/bin/crystalx-web                     # the one launch command
```

Open the address it prints (`http://127.0.0.1:7860`, or the next free port). Options:

| option | meaning |
|---|---|
| `--runs-dir DIR` | where run records go (default `./runs`, ignored by git) |
| `--toolbox DIR` | a directory holding your own SHELX/PLATON executables (default `./toolbox` if present) |
| `--tools-config FILE` | tool locations: explicit paths, search directories, `check.def` (default `./config.local.json`, git-ignored) |
| `--allow-translation` | start with strict native mode off (x86_64 programs may run through Rosetta 2) |
| `--port N` | a fixed port |
| `--check` | verify the installation and print static tool status; starts no server |

## Double-click start (macOS)

Two ways, for two kinds of user:

| | double-click | needs | what it is |
|---|---|---|---|
| developer | `crystalx_web/CrystalX Web Demo (developer).command` | this checkout's `.venv` (run `bootstrap_web.sh` once) | uses the project environment in place, through `scripts/crystalx.sh` |
| everyone else | `CrystalX Demo.app` (built into `crystalx_web/dist/`, see `packaging/README.md`) | macOS 14 or later on Apple Silicon | relocatable: its own Python and libraries inside; no project, no `.venv`, no network |

Either way the start does the same things:

* **One server per data folder.** It is reused when it is healthy. A repeated double click proves the
  running server's identity, using an HMAC challenge over a secret in a private record file, and
  then just opens the page again.
* **Ports.** A program on port 7860 that cannot prove it is ours is left alone, and the next free
  port is used. Nothing is ever killed.
* **Stale records.** A record left by a crashed server is replaced.
* **Networking.** Listens on `127.0.0.1` only. Loopback is exempted from proxies inside its own
  processes (`NO_PROXY`), so an inherited or broken proxy setting cannot stop the start. System
  proxy, DNS and firewall settings are not touched.
* **Bounded waits.** Startup waits at most 120 s, with a readable error and the log's tail if the
  server does not come up. A server it started that never answered is stopped.
* **First use: the synthetic demo.** The page opens with a labelled "▶ 一键运行合成演示（MOCK）".
  It needs no SHELX, PLATON, XQuartz or model weights. Real data is refused by the mock tools.
* **Real mode is unchanged.** It keeps the strict native checks and is never relaxed to look ready.

Stopping:

* **"停止本地服务"** at the top of the page. It confirms first, cancels an active run and keeps
  the records.
* **The developer launcher's Terminal window.** It is the server's console: close it or press
  Ctrl-C.
* **From a terminal:** `crystalx-web --desktop-stop` (or `--desktop-status`). Closing a browser tab
  does not stop the server.

Where things go:

| | developer launcher | `CrystalX Demo.app` |
|---|---|---|
| runs, records | `runs/` of this checkout | `~/Library/Application Support/CrystalX Demo/` |
| logs | `runs/_web/desktop/logs/server.log` | `~/Library/Logs/CrystalX Demo/` |

Nothing is ever written into the application. Nothing runs pip, downloads weights, installs Rosetta
2 or changes quarantine or Gatekeeper settings. The app is not notarized: a copy downloaded from the
Internet may need right-click → Open the first time.

## Native tools and what the page says about them

The **native tools** panel (open at the top) is a static inventory from `crystalx-native-tools`.
Files are read and nothing is run. For each program the pipeline uses (SHELXT, SHELXL,
`platon_special`, then `platon`) it shows:

- the file a run would use;
- its architecture;
- four separate facts: static check, launch, validated, and blockers with the next step.

**Strict native mode** is on by default. It refuses x86_64-only programs on Apple Silicon instead of
running them through Rosetta 2, and a real run is then BLOCKED before anything starts. It never
substitutes mock results. "Launch" shows only what the owner established with
`crystalx-tools probe <program> --authorize-launch`; the page never starts a probe.

The **Atom assignment** tab says where element labels come from: SHELXT peak heights as a starting
guess, kept by the bounded SHELXL refinement. The experimental CrystalX Apple backend is shown as
**not available** until trained weights, crystallographic preprocessing and integration acceptance
exist. The page shows no AI assignment and no atom confidence.

There is deliberately no option to listen on anything but the loopback interface.

## What stays on the machine

* The server binds to `127.0.0.1` only; sharing is off; Gradio telemetry is switched off before
  Gradio is imported; the theme uses system fonts, so no font CDN is contacted.
* Uploads, Gradio's cache, download copies and export bundles live under the runs directory
  (`runs/_gradio_tmp`, `runs/_web`), which git ignores.
* The 3-D viewer is a self-contained document in a sandboxed iframe whose Content-Security-Policy
  forbids every network request. Structure data cannot leave the page.
* Downloads are restricted to the files the run itself recorded.

## Demo mode

“载入内置合成演示数据” loads a generated, fictional structure and switches to the mock toolchain:
stand-ins for SHELXT, SHELXL and PLATON that write files in the shapes the real programs write,
marked `MOCK` everywhere. They refuse to run on anything except the built-in demo case, so a demo
result can never be produced for real data.

## Tests

```sh
.venv/bin/python -m pytest crystalx_web/tests                 # unit + browser tests
.venv/bin/python -m pytest crystalx_web/tests -m "not e2e"    # without a browser
```

Browser tests need `playwright` (`pip install -e crystalx_web[e2e]`) and an installed Chromium; they
are skipped otherwise and never download anything. No test runs a crystallographic program.
