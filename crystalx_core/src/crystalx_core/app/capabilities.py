"""What this build can do, and what this machine can run -- stated separately.

Two different questions are answered here and never merged:

* **Code readiness** -- whether a component is implemented, plan-only, or not integrated. This is a
  property of the software, the same on every machine.
* **Tool availability** -- what static inspection of the local executables says. That is header
  bytes, the executable bit, the quarantine attribute and the host architecture; it is **not** a
  launch test, and the report says so. The first real invocation is the evidence that a program
  starts.

Nothing in this module imports a heavy dependency. Whether the CrystalX inference package and
PyTorch are installed is checked with ``importlib.util.find_spec``, which locates a package without
importing it.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform

from crystalx_core.app.toolchain import PIPELINE_TOOLS, Toolchain
from crystalx_core.domain.tools import ToolKind
from crystalx_core.version import CORE_VERSION

__all__ = ["capability_report", "COMPONENTS"]

#: Code readiness, one row per component the user interface shows.
COMPONENTS: tuple[dict[str, str], ...] = (
    {
        "id": "shelxt_solution",
        "name": "SHELXT initial solution",
        "readiness": "production",
        "note": (
            "Runs SHELXT on an unmodified copy of the instruction file, so SHELXT searches every "
            "space group in the Laue group given by LATT/SYMM. Its .res is parsed; its .lxt table "
            "is read tolerantly and marked unverified."
        ),
    },
    {
        "id": "shelxl_preliminary",
        "name": "SHELXL preliminary refinement (bounded, isotropic)",
        "readiness": "implemented",
        "note": (
            "A fixed, small number of isotropic cycles with LIST 4 and ACTA. Listing patterns are "
            "not yet verified against a SHELXL run on this project's machines; values that do not "
            "parse are shown as unknown."
        ),
    },
    {
        "id": "platon_diagnostics",
        "name": "PLATON diagnostics (platon -u)",
        "readiness": "implemented",
        "note": (
            "Requires the CIF and the FCF from the refinement; without an FCF PLATON would start "
            "its own SHELXL job, so the step is blocked instead. The .chk format is unverified."
        ),
    },
    {
        "id": "heavy_atom_routes",
        "name": "Heavy-atom routes (SHELXS Patterson, SHELXC/SHELXD/SHELXE, ANODE)",
        "readiness": "plan_only",
        "note": "Command shapes are built and shown; no output parser exists, so they never run.",
    },
    {
        "id": "crystalx_inference",
        "name": "CrystalX element assignment (machine learning)",
        "readiness": "not_integrated",
        "note": (
            "Not part of this workflow in this release. Element labels come from SHELXT/SHELXL "
            "peak heights and are shown as unverified."
        ),
    },
    {
        "id": "eligibility_gates",
        "name": "Chemical and coordination eligibility gates",
        "readiness": "not_evaluated",
        "note": (
            "No geometry or charge-balance calculator exists yet, so these gates stay "
            "NOT_EVALUATED. A model can be displayed and exported, never ranked or called "
            "validated."
        ),
    },
)


def _distribution_version(name: str) -> str | None:
    """Installed distribution version from its metadata; the package itself is not imported."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - broken metadata
        return None


def _optional_package(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installations
        return False


def capability_report(toolchain: Toolchain) -> dict[str, object]:
    """A JSON-ready description of the build and of the tools this machine appears to have."""
    report = toolchain.availability()
    host = report.host
    tools: dict[str, dict[str, object]] = {}
    for kind in PIPELINE_TOOLS:
        resolution = report.get(kind)
        if resolution is None:
            continue
        identity = resolution.identity
        tools[kind.value] = {
            "usable_by_static_inspection": resolution.is_usable,
            "diagnostic": resolution.diagnostic.kind.value,
            "message": resolution.diagnostic.message,
            "remedy": resolution.diagnostic.remedy,
            "path": str(identity.path) if identity else None,
            "arch": identity.arch.value if identity else None,
            "source": identity.source.value if identity else None,
            "sha256": identity.sha256 if identity else None,
            "used_by_pipeline": kind
            in (ToolKind.SHELXT, ToolKind.SHELXL, ToolKind.PLATON, ToolKind.PLATON_SPECIAL),
        }
    platon = toolchain.platon()
    return {
        "core_version": CORE_VERSION,
        "mode": toolchain.mode.value,
        "host": {
            "system": host.system,
            "arch": host.arch.value,
            "rosetta_marker_present": host.rosetta_available,
            "python": platform.python_version(),
        },
        "tools": tools,
        "platon_for_pipeline": platon.kind.value if platon is not None and platon.is_usable else None,
        "components": [dict(row) for row in COMPONENTS],
        "translation_allowed": getattr(toolchain, "allow_translation", None),
        "crystalx_inference": {
            "integrated": False,
            "apple_engine_installed": _optional_package("crystalx_engine_apple_silicon"),
            "apple_engine_version": _distribution_version("crystalx-engine-apple-silicon"),
            "torch_installed": _optional_package("torch"),
            "note": (
                "Inference is not used by this workflow in this release, whether or not a backend "
                "is installed. The experimental Apple backend has been verified only with synthetic "
                "weights. Trained-model parity and crystallographic preprocessing are not validated "
                "or integrated into this workflow, so no element assignment or atom confidence comes from it. Model weights are "
                "never bundled."
            ),
        },
        "static_inspection_note": (
            "Tool availability comes from static inspection of each executable (header bytes, "
            "executable bit, quarantine flag, host architecture). It is not a launch test: the "
            "first real invocation is the evidence that a program starts."
        ),
    }
