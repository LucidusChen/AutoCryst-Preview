"""A native post-refinement zero-active evaluation is not a zero-parameter model."""
from crystalx_core.io.shelxl_lst import parse_lst_text


def test_final_zero_active_evaluation_preserves_last_refined_parameter_count():
    listing = ('wR2 = .1385 before cycle 16 for 13027 data and 145 / 145 parameters\n'
               'wR2 = .1385 before cycle 17 for 13027 data and 0 / 145 parameters\n')
    assert parse_lst_text(listing).parameters_refined == 145


def test_zero_only_evaluation_does_not_invent_completed_refinement():
    assert parse_lst_text('wR2 = .2 before cycle 1 for 100 data and 0 / 145 parameters\n').parameters_refined == 0
