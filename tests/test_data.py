from __future__ import annotations

import pandas as pd
import pytest

from nem_battery import data


@pytest.fixture
def fake_month(tmp_path, monkeypatch):
    """One AEMO-style 5-minute file for January 2024, served instead of the download."""
    ends = pd.date_range("2024-01-01 00:05", "2024-02-01 00:00", freq="5min")
    raw = pd.DataFrame(
        {
            "REGION": "SA1",
            "SETTLEMENTDATE": ends.strftime("%Y/%m/%d %H:%M:%S"),
            "TOTALDEMAND": 1000.0,
            "RRP": [float(i % 6) for i in range(len(ends))],  # 0..5 within each half-hour
            "PERIODTYPE": "TRADE",
        }
    )
    path = tmp_path / "PRICE_AND_DEMAND_202401_SA1.csv"
    raw.to_csv(path, index=False)
    monkeypatch.setattr(data, "download_month", lambda region, year, month: path)


def test_load_prices_averages_to_half_hours_labelled_by_interval_end(fake_month):
    prices = data.load_prices("SA1", "2024-01-01", "2024-01-02")
    assert len(prices) == 48
    assert prices.index[0] == pd.Timestamp("2024-01-01 00:30") and prices.index[-1] == pd.Timestamp("2024-01-02")
    assert (prices["rrp"] == 2.5).all() and (prices["demand"] == 1000).all()


def test_unknown_region_is_rejected():
    with pytest.raises(ValueError):
        data.download_month("WA1", 2024, 1)
