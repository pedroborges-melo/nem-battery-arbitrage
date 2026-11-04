"""The README's figures, each in a light and a dark version.

- capture_vs_mae: capture rate vs forecast error, one dot per method and month;
- score_agreement: how well each forecast score ranks the methods' daily revenue.

Reads the reports written by run_backtest.py and compare_metrics.py, so run those first.
Writes PNGs to figures/.

Usage: uv run python scripts/make_figures.py --region SA1 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS, FIGURES = ROOT / "reports", ROOT / "figures"
SOURCE = "Source: AEMO 5-minute price and demand data, averaged to half-hours. 100 MW / 200 MWh battery."

# Validated pair (blue, orange), a recessive grey for the other methods, and chart ink.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "other": "#b4b2a9", "blue": "#2a78d6", "orange": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "other": "#5c5b56", "blue": "#3987e5", "orange": "#d95926"},
}  # fmt: skip
HIGHLIGHT = {"lightgbm": ("LightGBM", "blue"), "profile_7d_median": ("7-day median profile", "orange")}
MARKER = 7
SCORE_LABELS = {
    "rank_corr": "Shape (rank correlation)",
    "extreme_hits": "Peak and trough hits",
    "peak_hits": "Peak hits",
    "mae": "Mean absolute error",
    "trough_hits": "Trough hits",
    "spread_error": "Spread error",
}
# One month where the less accurate forecast earned more, labelled on the chart.
EXAMPLE_MONTH = pd.Period("2024-09", "M")
EXAMPLES = [
    ("lightgbm", "September, LightGBM:\nerror {mae:.0f}, captured {capture:.0%}", (-130, 22)),
    ("naive_yesterday", "September, yesterday's prices:\nerror {mae:.0f}, captured {capture:.0%}", (14, 8)),
]


def monthly(results: pd.DataFrame) -> pd.DataFrame:
    """MAE and capture rate per forecaster and month."""
    results = results.assign(month=results["day"].dt.to_period("M"))
    out = results.groupby(["month", "forecaster"], as_index=False).agg(
        mae=("mae", "mean"), revenue=("revenue_aud", "sum")
    )
    best = out[out["forecaster"] == "perfect_foresight"].set_index("month")["revenue"]
    out["capture"] = out["revenue"] / out["month"].map(best)
    return out[out["forecaster"] != "perfect_foresight"]


def rc(t: dict) -> dict:
    return {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["secondary"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False, "axes.grid": True, "grid.color": t["grid"],
        "grid.linewidth": 0.75, "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "font.size": 9.5, "legend.frameon": False,
    }  # fmt: skip


def frame(t: dict, title: str, subtitle: str, height: float, left: float = 0.09, top: float = 0.80):
    """A figure with its title, subtitle and source note placed; returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=(8, height))
    fig.subplots_adjust(left=left, right=0.97, top=top, bottom=0.85 / height)
    fig.text(0.09, 1 - 0.22 / height, title, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.09, 1 - 0.47 / height, subtitle, fontsize=9.5, color=t["secondary"], va="top")
    fig.text(0.09, 0.12 / height, SOURCE, fontsize=8, color=t["muted"])
    return fig, ax


def capture_vs_mae(points: pd.DataFrame, t: dict, path: Path) -> None:
    fig, ax = frame(t, "A lower forecast error does not reliably earn more",
                    "Capture rate vs forecast error in South Australia, one dot per method and month of 2024",
                    height=5)  # fmt: skip

    ring = {"markersize": MARKER, "markeredgecolor": t["surface"], "markeredgewidth": 1.5, "linestyle": "none"}
    other = points[~points["forecaster"].isin(HIGHLIGHT)]
    ax.plot(other["mae"], other["capture"], "o", color=t["other"], zorder=2, **ring)
    for name, (_, color) in HIGHLIGHT.items():
        mine = points[points["forecaster"] == name]
        ax.plot(mine["mae"], mine["capture"], "o", color=t[color], zorder=3, **ring)

    for name, text, offset in EXAMPLES:
        row = points[(points["forecaster"] == name) & (points["month"] == EXAMPLE_MONTH)].iloc[0]
        ax.annotate(text.format(mae=row["mae"], capture=row["capture"]), (row["mae"], row["capture"]),
                    xytext=offset, textcoords="offset points", fontsize=8.5, color=t["secondary"],
                    arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.75})  # fmt: skip

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([40, 60, 100, 150, 200, 300]))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}")
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(-0.15, 0.88)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Mean absolute error of the day-ahead forecast (AUD/MWh, log scale)")
    ax.set_ylabel("Capture rate")
    ax.axhline(0, color=t["muted"], linewidth=0.75)

    handles = [Line2D([], [], marker="o", color=t[c], label=label, **ring) for label, c in HIGHLIGHT.values()]
    handles.append(Line2D([], [], marker="o", color=t["other"], label="Naive and MSTL baselines", **ring))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9,
              labelcolor=t["secondary"], borderaxespad=0.3, handletextpad=0.3, columnspacing=1.6)  # fmt: skip
    fig.savefig(path, dpi=200)
    plt.close(fig)


def score_agreement(table: pd.DataFrame, t: dict, path: Path) -> None:
    table = table.sort_values("daily_agreement")
    fig, ax = frame(t, "The forecast's shape tracks revenue better than its error",
                    "Rank correlation between each score and revenue across the five methods, "
                    "mean of the days of 2024", height=4.2, left=0.30, top=0.78)  # fmt: skip
    y = range(len(table))
    is_mae = table.index == "mae"
    ax.errorbar(table["daily_agreement"], y, xerr=table["ci95"], fmt="none", ecolor=t["muted"], elinewidth=1.5,
                capsize=0, zorder=2)  # fmt: skip
    for mask, color in ((~is_mae, t["blue"]), (is_mae, t["ink"])):
        ax.plot(table["daily_agreement"][mask], [i for i, m in zip(y, mask, strict=True) if m], "o", color=color,
                markersize=MARKER, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)  # fmt: skip
    ax.axvline(table.loc["mae", "daily_agreement"], color=t["muted"], linewidth=0.75, linestyle=(0, (3, 3)))
    ax.set_yticks(list(y), [SCORE_LABELS[s] for s in table.index])
    ax.tick_params(axis="y", colors=t["secondary"], labelsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.05, 0.65)
    ax.set_xlabel("Mean daily rank correlation with revenue (bars: 95% interval)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    tag = f"{args.region}_{args.start}_{args.end}"
    points = monthly(pd.read_csv(REPORTS / f"backtest_{tag}.csv", parse_dates=["day"]))
    agreement = pd.read_csv(REPORTS / f"score_agreement_{tag}.csv", index_col="score")
    FIGURES.mkdir(exist_ok=True)
    for mode, theme in THEMES.items():
        with plt.rc_context(rc(theme)):
            capture_vs_mae(points, theme, FIGURES / f"capture_vs_mae-{mode}.png")
            score_agreement(agreement, theme, FIGURES / f"score_agreement-{mode}.png")


if __name__ == "__main__":
    main()
