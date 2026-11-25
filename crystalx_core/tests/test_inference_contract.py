import pytest

from crystalx_core.domain.inference import ElementInferenceRequest


def test_request_freezes_sequences_and_retains_context():
    request = ElementInferenceRequest(["a", "b"], [6, 8], [[0., 0., 0.], [1., 0., 0.]], [6, 8], 1)
    assert request.site_ids == ("a", "b")
    assert request.positions_angstrom == ((0., 0., 0.), (1., 0., 0.))
    assert request.main_site_count == 1


@pytest.mark.parametrize("kwargs", [
    {"site_ids": ()}, {"site_ids": ("a", "a")}, {"atomic_numbers": (0, 8)},
    {"atomic_numbers": (True, 8)}, {"positions_angstrom": ((float("nan"), 0., 0.), (1., 0., 0.))},
    {"positions_angstrom": ((0., 0.), (1., 0., 0.))}, {"candidate_atomic_numbers": (6,)},
    {"candidate_atomic_numbers": (6, 6)}, {"main_site_count": 0}, {"main_site_count": 3},
])
def test_bad_requests_fail(kwargs):
    values = dict(site_ids=("a", "b"), atomic_numbers=(6, 8), positions_angstrom=((0., 0., 0.), (1., 0., 0.)))
    values.update(kwargs)
    with pytest.raises(ValueError):
        ElementInferenceRequest(**values)
