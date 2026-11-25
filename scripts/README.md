# Supported local startup

From the repository root:

```sh
sh scripts/crystalx.sh check
sh scripts/crystalx.sh web
```

The first command checks all three installed packages and command entry points.
The second starts the existing local Web interface. Keep its terminal open;
Ctrl-C stops it. Use its labelled synthetic demo for a software smoke test.
These commands do not launch proprietary programs merely to check startup.

```sh
sh scripts/crystalx.sh core --version
sh scripts/crystalx.sh apple doctor
sh scripts/crystalx.sh test all
```

`test core`, `test web`, `test apple` and `test startup` run individual suites.
`test tools` runs the native-tools suite; `tools inventory` performs read-only static inspection
and returns exit 3 when prerequisites are blocked. All four installed package commands are
covered by the startup check and recurrence protection.
The existing local Chromium location is supplied for browser tests. Apple tests
remove implicit per-operator MPS fallback from their child environment; a normal
Mac terminal is needed to run actual GPU and browser tests. Sandboxed runtimes may
skip GPU/browser tests or refuse OS process/resource operations.

The built-in Web/core mock demo does not need upstream model sources. Apple
inference and its synthetic GPU smoke test require separately authorized local
upstream sources. The private-only upstream copy is described in NOTICE; weights remain excluded. Supply your own path
to the audited `inference_repo` directory. No upstream source or weights are
downloaded automatically; missing or mismatched sources block execution. This
snapshot supplies no upstream redistribution license.

Strict synthetic GPU smoke:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK sh scripts/crystalx.sh apple self-test \
  --source-root /path/to/licensed-upstream/inference_repo --device mps --no-cpu-fallback
```

Expect synthetic weights, `device_used=mps`, `fell_back=false` and
`software_smoke_passed=true`. This is not a trained-checkpoint or scientific pass.

## Why use the local entry point

Python 3.12 skips hidden `.pth` files. An unknown external operation has repeatedly
applied macOS UF_HIDDEN to this project's environment, including during a test
suite. The launcher runs before Python site processing, inspects only the fixed
project `.venv`, and clears UF_HIDDEN only on its import directories/immediate
`.pth` files. No recursive home/repository/system flag change occurs.

It also validates the existing editable-install metadata and creates explicit
package links in that environment's ordinary site-packages for core, Web, Apple and native-tools.
These point to this project's source, so subsequent child interpreters import
normally even if the `.pth` files become hidden again. It does not add PYTHONPATH,
reinstall packages, create a new environment, or replace user/system startup hooks.
Unexpected package directories, conflicting links or mismatched metadata are
preserved and reported instead of overwritten. A missing install remains an
error. The helper deliberately manages only `.venv` inside this repository.

When intentionally replacing an editable package with a wheel in the same
environment, remove its exact development link first so an installer cannot
write through that link into source. Ordinary editable reinstallation remains
supported. Existing core/Web bootstrap scripts still perform initial installation;
the launcher is the ongoing startup path, not a replacement package installer.

The flag-reapplication origin remains unknown. Successful recurrence tests prove
this import mechanism survives observed hidden-flag changes; they do not claim
protection against arbitrary concurrent environment replacement.
