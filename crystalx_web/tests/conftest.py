"""Test environment for the Web interface.

Gradio reads some settings at import time, so telemetry is switched off and its upload directory
is pointed at a throwaway location *before* any test module imports it.
"""

from __future__ import annotations

import os
import tempfile

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("GRADIO_TEMP_DIR", tempfile.mkdtemp(prefix="crystalx-web-tests-"))
