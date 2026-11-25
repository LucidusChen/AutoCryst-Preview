"""Adapters for optional third-party capabilities.

Nothing in this package is imported by ``crystalx_core`` itself. Importing
``crystalx_core.providers.gemmi_provider`` is what pulls gemmi in, and that import only happens when
an application asks for it:

    pip install -e '.[geo]'

    from crystalx_core.providers import gemmi_provider
    gemmi_provider.register()

The zero-dependency test imports every *other* subpackage and asserts no third-party module ends up
in ``sys.modules``. This one is excluded from that sweep by construction, not by exception.
"""

from __future__ import annotations

__all__: list[str] = []
