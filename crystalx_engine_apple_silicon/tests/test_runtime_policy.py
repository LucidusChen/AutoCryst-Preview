"""The adapter must not change a calling application's global tensor policy."""

import pytest
import torch

from test_backend import SOURCE, request_sites, weights  # noqa: F401
from crystalx_engine_apple_silicon.backend import AppleSiliconBackend


def test_non_float32_default_is_a_clear_failure_without_global_mutation(weights, request_sites):
    engine = AppleSiliconBackend(SOURCE, weights)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="default dtype.*float32"):
            engine.predict_elements(request_sites, device="cpu")
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original)
