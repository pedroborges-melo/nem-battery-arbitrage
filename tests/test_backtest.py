from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nem_battery.backtest import evaluate, forecast, load_forecasts, summarise
from nem_battery.battery import Battery
from nem_battery.forecasters import PERFECT, naive_last_week, naive_yesterday, perfect_foresight, profile_7d_median

# The models need more history than the synthetic data has, so the backtest is tested with the naive methods.
NAIVE = {
    PERFECT: perfect_foresight,
    "naive_yesterday": naive_yesterday,
    "naive_last_week": naive_last_week,
    "profile_7d_median": profile_7d_median,
}
DAYS = pd.date_range("2024-01-10", "2024-01-12")


def test_forecast_keeps_one_row_per_day_and_method(market):
    forecasts = forecast(market["rrp"], DAYS, NAIVE)
    assert forecasts.shape == (len(DAYS) * len(NAIVE), 48)
    assert forecasts.index.names == ["day", "forecaster"]


def test_days_with_a_missing_half_hour_are_skipped(market):
    prices = market["rrp"].copy()
    prices["2024-01-11 12:00"] = np.nan  # the grid keeps the slot; the value is missing
    assert set(forecast(prices, DAYS, NAIVE).index.get_level_values("day")) == {DAYS[0], DAYS[2]}


def test_perfect_foresight_scores_perfectly_and_earns_the_most(market):
    results = evaluate(forecast(market["rrp"], DAYS, NAIVE), market["rrp"], Battery())
    perfect = results[results["forecaster"] == PERFECT]
    assert (perfect["mae"] == 0).all() and (perfect["rank_corr"] == 1).all() and (perfect["extreme_hits"] == 1).all()
    summary = summarise(results)
    assert summary.loc[PERFECT, "capture_rate"] == pytest.approx(1)
    assert (summary["capture_rate"] <= 1 + 1e-9).all()


def test_saved_forecasts_read_back_exactly(market, tmp_path):
    forecasts = forecast(market["rrp"], DAYS, NAIVE)
    forecasts.to_csv(tmp_path / "forecasts.csv")
    pd.testing.assert_frame_equal(load_forecasts(tmp_path / "forecasts.csv"), forecasts, check_exact=True)
