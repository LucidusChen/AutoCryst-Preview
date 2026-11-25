"""Native tool management for the CrystalX/AutoCryst workflow: the owner's SHELX and PLATON.

Discovers the user's own, separately licensed crystallographic executables (configuration file,
project toolbox, system directories, Olex2 installations), inspects them statically (architecture,
linked libraries, code signature, resources), applies the strict native (no-Rosetta) policy,
records hashed provenance, and plans -- never performs -- isolated local builds from source the
owner supplies. Nothing is downloaded, installed, compiled or redistributed, and no program is run
except by the explicitly authorised launch probe.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
