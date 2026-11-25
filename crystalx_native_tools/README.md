# crystalx-native-tools

Native tool management for the CrystalX/AutoCryst workflow: the owner's own, separately licensed
SHELX and PLATON executables. Standard library plus the dependency-free `crystalx-core`.

## What it does

- **Discovery.** It looks in `config.local.json` (explicit paths), the project `toolbox/`, extra
  search directories, `PATH` (plus `/usr/local/bin`), Olex2 installations (including versioned
  `Olex2-*.app` bundles) and conventional locations. The order is the same as a run uses, because
  selection is delegated to `crystalx_core`'s resolver with the same configuration and policy.
- **Static inspection.** For each file it reads the Mach-O header: architectures (thin or
  universal, including `arm64e`), linked libraries resolved as dyld would resolve them (`@rpath`,
  `@loader_path`, the dyld shared cache), the code signature (ad-hoc, linker-signed, identifier,
  team), the minimum macOS version and the quarantine flag. It records SHA-256, size and mtime as
  provenance. Nothing is executed.
- **Strict native mode.** This is the default. On Apple Silicon an x86_64-only program is refused
  and the search continues to arm64 or universal builds. The report says what to do instead. A
  wrapper cannot turn x86_64 code into arm64 code.
- **PLATON resources.** It reports `check.def` candidates and their (heuristic) `yymmdd` version
  against dates found in the executable. It also flags X11/XQuartz and gfortran runtime libraries.
  Three facts stay separate:
  - the file **exists** (static);
  - the program **can start** (the opt-in launch probe only);
  - a validation **succeeded** (an authorised real run with a parsed `.chk` only).
- **Official availability evidence.** For each tool: whether a native arm64 build is known
  (`unknown` for SHELX), whether source is available, and the sources consulted (see
  `availability.py`).
- **Build plans, never builds.** It plans an isolated user-local arm64 build from source the owner
  supplies and may legally build: recipe, steps, toolchain requirements and blockers. A plan
  always has `executed: false`. Compiler presence is judged without running anything, including
  Apple's `xcode-select` shims.

## Command line

```sh
crystalx-tools inventory                       # pipeline programs; exit 3 when blocked
crystalx-tools inventory --all --json          # every SHELX/PLATON program, as JSON provenance
crystalx-tools inventory --allow-translation   # judge x86_64 builds as Rosetta-runnable
crystalx-tools toolchain                       # Command Line Tools, gfortran (arm64?), XQuartz
crystalx-tools build-plan platon_special SRC --licence-ack "terms allow a local build"
crystalx-tools probe platon_special --authorize-launch   # RUNS the program once, isolated
```

The launch probe is the only command that runs a crystallographic program. It requires
`--authorize-launch`. It runs the program with no arguments in an empty temporary directory, with
closed standard input, `DISPLAY`/`CHECKDEF` unset, a 10-second limit and process-group cleanup.
Its record (`runs/_tools/launch_probes.json`, git-ignored) is tied to the file's SHA-256: a changed
file has no launch evidence. The Web interface never starts a probe.

## Boundaries

Use `sh scripts/crystalx.sh tools inventory` from the project root for the supported local entry.
The panel follows the same core PLATON preference as a real run. If deeper inspection blocks
the selected preferred build, the pipeline is reported blocked; it does not announce an unused
fallback. Deeper library/signature/minimum-OS findings do not yet steer the core's selection or
guarantee a prelaunch stop. Resolve those blockers before a real computation.

No download, installation, compilation, redistribution or system change. Proprietary programs,
source and build products never enter Git or a distribution. `config.local.json` and `runs/` are
git-ignored.
