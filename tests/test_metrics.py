from __future__ import annotations

import numpy as np
import pytest

from nem_battery.metrics import hit_rates, rank_correlation, scores, spread

ACTUAL = np.array([10.0, 50, 20, 80, 30, 100])


def test_a_shifted_forecast_has_perfect_shape_but_a_large_error():
    s = scores(ACTUAL + 40, ACTUAL, k=2)
    assert s["mae"] == pytest.approx(40)
    assert s["rank_corr"] == pytest.approx(1)
    assert s["trough_hits"] == s["peak_hits"] == 1
    assert s["spread_error"] == pytest.approx(0)


def test_a_reversed_forecast_misses_every_extreme():
    reversed_ = ACTUAL.max() + ACTUAL.min() - ACTUAL
    assert rank_correlation(reversed_, ACTUAL) == pytest.approx(-1)
    assert hit_rates(reversed_, ACTUAL, k=2) == (0, 0)


def test_spread_is_dearest_minus_cheapest():
    assert spread(ACTUAL, k=2) == pytest.approx((100 + 80) / 2 - (10 + 20) / 2)


def test_flat_forecast_has_no_rank_correlation():
    assert np.isnan(rank_correlation(np.full(6, 50.0), ACTUAL))
