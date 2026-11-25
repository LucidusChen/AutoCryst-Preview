"""Entry point for ``python -m crystalx_core``.

The console script installed by ``pip`` is ``crystalx``. This module exists so that the package is
also runnable without installing it -- ``PYTHONPATH=src python -m crystalx_core doctor`` -- which is
how it gets exercised on a machine that only has the source tree.

Running ``python -m crystalx_core.cli.main`` also works, but emits a ``RuntimeWarning`` because
``crystalx_core.cli.__init__`` has already imported that module by the time ``runpy`` executes it.
Prefer this entry point.
"""

from __future__ import annotations

import sys

from crystalx_core.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
