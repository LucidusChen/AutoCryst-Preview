# AutoCryst (preview)

AutoCryst is a local, AI-assisted workflow for small-molecule crystal structure solution and
refinement on Apple silicon Macs. It combines the pretrained [CrystalX](https://doi.org/10.1021/jacs.5c21832)
element-assignment model with the established SHELX programs and PLATON, and keeps every step
auditable: inputs are never modified, each run is recorded, and the final scientific judgement
stays with the crystallographer.

This repository is a **developer preview**. It is a source snapshot of a private development
repository, without its internal records. Expect rough edges and manual setup.

## Status (November 2026)

What works:

* A local Web GUI (Chinese, Japanese, English) runs the whole chain:
  1. initial solution (SHELXT; SHELXS/SHELXD; a protein SHELXC/D/E route);
  2. CrystalX element assignment;
  3. SHELXL refinement rounds;
  4. sealed candidate files;
  5. a 3-D view;
  6. a separate local checkCIF/PLATON step.
* The CrystalX heavy-atom (non-hydrogen) stage, selectable as "initial heavy stage", reproduces the
  upstream CrystalX predictions. In our tests, the upstream code run unchanged and this
  implementation chose the same element at every site. In development tests on 10 public
  organic structures it assigned 250 of 250 non-hydrogen elements correctly.
* CrystalX inference runs on the CPU or on the Apple GPU (PyTorch MPS).
* A synthetic demo mode runs without any external program or model weight.

Not yet done or still experimental:

* The CrystalX hydrogen stage is not integrated yet.
* The iterative growth loop and the metal-anchored growth mode are experimental. They can add
  wrong sites, and large or disordered structures are not solved reliably.
* SHELX programs for macOS are x86_64 builds and need Rosetta 2. A native backup engine based on
  cctbx/smtbx is in development.
* There is no installer; setup is manual (below).
* A native SwiftUI/Metal frontend is planned; only its Python bridge exists (`crystalx_bridge`).

## Repository layout

| Directory | Contents |
| --- | --- |
| `crystalx_core/` | Workflow engine: initial-solution routes, CrystalX assignment boundary, SHELXL rounds, stop codes, sealed run workspaces, comparison tools |
| `crystalx_web/` | Gradio Web GUI and the macOS double-click launcher |
| `crystalx_engine_apple_silicon/` | Runs the CrystalX model on Apple silicon (CPU or MPS) from a separately supplied upstream checkout |
| `crystalx_native_tools/` | Finds and inspects SHELX/PLATON installations, reports native or translated (Rosetta) execution |
| `crystalx_bridge/` | JSON-lines helper for a future native frontend |
| `scripts/` | Local startup and test entry point (`scripts/crystalx.sh`) |
| `toolbox/` | Empty placeholder for your own SHELX/PLATON executables |

## What you must supply

None of the following is included or redistributed here.

* **SHELX** (SHELXT, SHELXL, and optionally SHELXS/SHELXD/SHELXC/SHELXE): from the official
  SHELX site, after registration.
* **PLATON** (for local checkCIF).
* **CrystalX upstream inference code and the official weights**: from
  [InternScience/CrystalX](https://github.com/InternScience/CrystalX) and the official CrystalX
  model repository. Expected layout: `crystalx_engine/inference_repo/` with
  `weights/crystalx-heavy.pth`. The upstream model files and the heavy checkpoint are verified by
  SHA-256 against the revision this code was tested with; a different revision is refused rather
  than used silently.
* **Python 3.10 or newer** (3.12 recommended), from which the bootstrap script below builds the
  project environment.

## Developer setup (macOS, Apple silicon)

The `python3` that comes with the macOS command line tools is too old. Install Python 3.12 from
python.org or with Homebrew (`brew install python@3.12`), then:

```sh
git clone https://github.com/LucidusChen/AutoCryst-Preview.git
cd AutoCryst-Preview
PYTHON=python3.12 sh crystalx_web/scripts/bootstrap_web.sh   # creates .venv: core, native tools, CrystalX Apple engine (PyTorch), Web GUI
sh scripts/crystalx.sh check                                  # checks the four installed packages
sh scripts/crystalx.sh web                                    # starts the Web GUI on http://127.0.0.1:7860/ (Ctrl-C stops it)
```

The bootstrap downloads ordinary Python packages (Gradio, PyTorch and their dependencies). It never
downloads CrystalX code or weights and runs no crystallographic program.

### First run: the synthetic demo

The demo needs no SHELX, PLATON or model weights. On the start page:

1. click **Load the built-in synthetic demo** (载入内置合成演示数据);
2. keep **Mock demo** selected and click **Start run** (开始运行).

All numbers in this mode are placeholders from stand-in programs, not results.

### Real data

* **SHELX/PLATON.** Put the executables in `toolbox/`, or name them in a local, git-ignored
  `config.local.json`. The macOS SHELX builds are x86_64 and run through Rosetta 2, so allow
  translation:

  ```json
  { "toolbox": "toolbox", "strict_native": false }
  ```

* **CrystalX.** Place the upstream inference code at `crystalx_engine/inference_repo/` and the
  heavy checkpoint at `crystalx_engine/inference_repo/weights/crystalx-heavy.pth`; the folder is
  git-ignored. Four upstream model files are checked by SHA-256 before use, and a different
  revision is refused with a message that names the file.

See `crystalx_web/README.md` and `scripts/README.md` for the desktop launcher and the test suites
(`sh scripts/crystalx.sh test all`).

## Scientific scope

A numerical stop, or a low R1, is not a validated structure. Candidates are labelled as such until
a crystallographer has checked them. The tool never uses an expert or reference structure as a
solving input.

## Attribution

AutoCryst builds on CrystalX: K. Zheng, W. Huang, W. Ouyang, H.-S. Zhong, Y. Li, "CrystalX:
High-Accuracy Crystal Structure Analysis Using Deep Learning", *J. Am. Chem. Soc.* 2026, 148,
20412–20421, doi:[10.1021/jacs.5c21832](https://doi.org/10.1021/jacs.5c21832). The upstream code
is at [InternScience/CrystalX](https://github.com/InternScience/CrystalX). This project is not
affiliated with or endorsed by the CrystalX authors. See [NOTICE](NOTICE).

SHELX is by G. M. Sheldrick, PLATON by A. L. Spek, and cctbx/smtbx by their respective authors;
they are separate programs under their own terms.

## License

This project's own code is released under the [MIT License](LICENSE). The license does not cover
CrystalX upstream code or weights, SHELX, PLATON or any other third-party program; they keep their
own terms.
