"""crystalx_core: the platform-neutral workflow core for AutoCryst / CrystalX GUI.

This package owns the crystallographic workflow, the provenance model, the external-tool adapters
and the candidate lineage. It never imports PyTorch and contains no CUDA or Apple-specific code, so
that a CUDA engine and an Apple Silicon engine can be swapped behind the same contract.
"""

from __future__ import annotations

from crystalx_core.version import CORE_VERSION, RECORD_SCHEMA_VERSION

__all__ = ["CORE_VERSION", "RECORD_SCHEMA_VERSION", "__version__"]

__version__ = CORE_VERSION
