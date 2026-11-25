"""Local Gradio Web interface for the CrystalX/AutoCryst workflow core.

The interface is a client of :class:`crystalx_core.app.RunService`; it contains no workflow logic.
Gradio is this package's dependency, never the core's.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.1"
