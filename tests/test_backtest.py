from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nem_battery.backtest import FORECASTERS, _day, naive_yesterday, run, summarise


def test_day_takes_intervals_ending_after_midnight_through_the_next_midnight(market):
    day = pd.Timestamp("2024-01-10")
    intervals = _day(market["rrp"], day).index
    assert len(intervals) == 48
    assert intervals[0] == day + pd.Timedelta("30min") and intervals[-1] == day + pd.Timedelta(days=1)


def test_naive_yesterday_is_the_previous_day(market):
    day = pd.Timestamp("2024-01-10")
    expected = _day(market["rrp"], day - pd.Timedelta(days=1)).to_numpy()
    np.testing.assert_array_equal(naive_yesterday(market["rrp"], day, 48), expected)


def test_perfect_foresight_beats_every_forecast(market):
    days = pd.date_range("2024-01-10", "2024-01-16")
    summary = summarise(run(market["rrp"], days))
    assert summary.loc["perfect_foresight", "capture_rate"] == pytest.approx(1)
    assert (summary["capture_rate"] <= 1 + 1e-9).all()
    assert set(summary.index) == set(FORECASTERS)


def test_incomplete_days_are_skipped(market):
    prices = market["rrp"].drop(market.index[(market.index > "2024-01-12") & (market.index <= "2024-01-12 06:00")])
    days = pd.date_range("2024-01-11", "2024-01-13")
    assert set(run(prices, days, {"yesterday": naive_yesterday})["day"]) == {pd.Timestamp("2024-01-11")}
