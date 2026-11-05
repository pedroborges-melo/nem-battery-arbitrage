from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nem_battery.agreement import agreement, oriented


def results(days: int = 40) -> pd.DataFrame:
    """Four methods whose revenue follows their shape score exactly and their MAE in reverse."""
    rng = np.random.default_rng(3)
    rows = []
    for day in pd.date_range("2024-01-01", periods=days):
        revenue = rng.permutation(4) * 1000.0
        noise = rng.permutation(4)
        for i, name in enumerate(["a", "b", "c", "d"]):
            rows.append({
                "day": day, "forecaster": name, "revenue_aud": revenue[i],
                "rank_corr": revenue[i] / 4000,  # perfectly aligned with revenue
                "mae": 100 - revenue[i] / 100,  # lower error earns more: aligned once flipped
                "spread_error": float(noise[i]),  # unrelated to revenue
                "trough_hits": 0.5, "peak_hits": 0.5, "extreme_hits": 0.5,  # tied every day: no correlation
            })  # fmt: skip
        rows.append({"day": day, "forecaster": "perfect_foresight", "revenue_aud": 1e6, "rank_corr": 1.0,
                     "mae": 0.0, "spread_error": 0.0, "trough_hits": 1.0, "peak_hits": 1.0,
                     "extreme_hits": 1.0})  # fmt: skip
    return pd.DataFrame(rows)


def test_aligned_scores_agree_fully_and_unrelated_ones_do_not():
    table = agreement(results())
    assert table.loc["rank_corr", "daily_agreement"] == pytest.approx(1)
    assert table.loc["mae", "daily_agreement"] == pytest.approx(1)  # flipped: lower error, higher revenue
    assert abs(table.loc["spread_error", "daily_agreement"]) < 0.3
    assert table.loc["rank_corr", "gap_to_mae"] == pytest.approx(0)


def test_tied_days_are_left_out_and_counted():
    table = agreement(results(days=10))
    assert table.loc["rank_corr", "days"] == 10
    assert table.loc["trough_hits", "days"] == 0 and np.isnan(table.loc["trough_hits", "daily_agreement"])


def test_an_aligned_score_picks_the_top_earner_every_month():
    table = agreement(results(days=60))  # January and February
    assert table.loc["rank_corr", "months_top_earner"] == 2


def test_perfect_foresight_is_left_out_and_lower_is_better_scores_are_flipped():
    scored = oriented(results())
    assert set(scored["forecaster"]) == {"a", "b", "c", "d"}
    assert (scored["mae"] <= 0).all()
