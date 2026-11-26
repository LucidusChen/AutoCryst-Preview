from pathlib import Path

import pytest


@pytest.fixture
def source_root():
    upstream = Path(__file__).resolve().parents[2] / "crystalx_engine"
    if not upstream.exists() and not upstream.is_symlink():
        pytest.skip("The public preview omits the separately supplied upstream snapshot")
    return upstream / "inference_repo"
