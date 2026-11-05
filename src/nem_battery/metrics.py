"""How good a day's forecast is, measured by error and by what the battery needs from it.

MAE scores every half-hour alike. A battery only acts in a few of them: it charges in
the cheapest and discharges in the dearest, so what it needs from a forecast is the
day's shape and the timing of its extremes, not the level of each interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(forecast: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(forecast - actual).mean())


def rank_correlation(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Spearman correlation within the day: does the forecast order the half-hours right?"""
    if np.ptp(forecast) == 0 or np.ptp(actual) == 0:
        return float("nan")  # undefined for a flat day
    return float(pd.Series(forecast).rank().corr(pd.Series(actual).rank()))  # Pearson on ranks


def extremes(prices: np.ndarray, k: int) -> tuple[set[int], set[int]]:
    """Indices of the k cheapest and the k dearest intervals."""
    order = np.argsort(prices, kind="stable")
    return set(order[:k].tolist()), set(order[-k:].tolist())


def hit_rates(forecast: np.ndarray, actual: np.ndarray, k: int) -> tuple[float, float]:
    """Share of the actual k cheapest and k dearest intervals the forecast also picks."""
    (fc_low, fc_high), (low, high) = extremes(forecast, k), extremes(actual, k)
    return len(fc_low & low) / k, len(fc_high & high) / k


def spread(prices: np.ndarray, k: int) -> float:
    """Mean of the k dearest intervals minus mean of the k cheapest."""
    ordered = np.sort(prices)
    return float(ordered[-k:].mean() - ordered[:k].mean())


def scores(forecast: np.ndarray, actual: np.ndarray, k: int) -> dict[str, float]:
    """All of a day's forecast scores, keyed by column name. `k` is the battery's duration in intervals."""
    trough_hits, peak_hits = hit_rates(forecast, actual, k)
    return {
        "mae": mae(forecast, actual),
        "rank_corr": rank_correlation(forecast, actual),
        "trough_hits": trough_hits,
        "peak_hits": peak_hits,
        "extreme_hits": (trough_hits + peak_hits) / 2,
        "spread_error": abs(spread(forecast, k) - spread(actual, k)),
    }
