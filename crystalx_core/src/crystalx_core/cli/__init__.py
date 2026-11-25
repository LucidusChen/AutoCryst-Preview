"""Command-line interface for crystalx_core.

Note: this package re-exports the ``main`` *function*, which shadows the ``crystalx_core.cli.main``
*module* on the package object. Import the module with
``importlib.import_module("crystalx_core.cli.main")`` if you need it directly.
"""

from __future__ import annotations

from crystalx_core.cli.main import build_parser, main

__all__ = ["build_parser", "main"]
