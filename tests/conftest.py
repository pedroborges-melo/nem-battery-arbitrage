"""Synthetic half-hourly prices and demand, labelled by interval end like AEMO's data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def half_hours(start: str, days: int) -> pd.DatetimeIndex:
    """Interval-end labels for `days` whole trading days starting at `start`."""
    return pd.date_range(pd.Timestamp(start) + pd.Timedelta("30min"), periods=days * 48, freq="30min")


@pytest.fixture
def market() -> pd.DataFrame:
    """Ninety days with a cheap midday and an expensive evening, plus noise."""
    index = half_hours("2024-01-01", 90)
    hour = (index - pd.Timedelta("1min")).hour
    rng = np.random.default_rng(0)
    rrp = 60 + 80 * np.isin(hour, [18, 19, 20]) - 50 * np.isin(hour, [11, 12, 13]) + rng.normal(0, 5, len(index))
    demand = 1500 + 300 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 20, len(index))
    return pd.DataFrame({"rrp": rrp, "demand": demand}, index=index)
