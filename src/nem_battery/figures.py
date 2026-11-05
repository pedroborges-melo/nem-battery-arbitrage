"""The README's figures, each in a light and a dark version, drawn from the reports.

- cover: the preview image for the article and the repository (light only, 1280 x 640);
- battery_day: one day's prices and the plan of a battery that knew them in advance;
- capture_bars: the share of the perfect-foresight revenue each method earned;
- two_forecasts: a day on which the less accurate forecast earned more;
- capture_vs_mae: capture rate vs forecast error, one dot per method and month;
- score_agreement: how well each forecast score ranks the methods' daily revenue;
- sensitivity: the shape's lead over MAE for other batteries and regions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, PercentFormatter

from .backtest import load_forecasts
from .battery import Battery, schedule
from .data import FIGURES_DIR, INTERVAL_HOURS, INTERVALS, load_prices, prices_on
from .forecasters import PERFECT
from .pipeline import BASE_SCENARIO, agreement_path, capture_path, daily_path, forecasts_path, year_days

SOURCE = "Source: AEMO 5-minute price and demand data, averaged to half-hours."
BASE_NOTE = f"{SOURCE} 100 MW / 200 MWh battery."
SENSITIVITY_NOTE = "Source: AEMO price data. SA: South Australia. 100 MW batteries; base: 2 hours, AUD 10/MWh wear."

# Validated pair (blue, orange), a recessive grey for the other methods, and chart ink.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "other": "#b4b2a9", "blue": "#2a78d6", "orange": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "other": "#5c5b56", "blue": "#3987e5", "orange": "#d95926"},
}  # fmt: skip
HIGHLIGHT = {"lightgbm": ("LightGBM", "blue"), "profile_7d_median": ("7-day median profile", "orange")}
MARKER = 7
METHOD_LABELS = {
    PERFECT: "Perfect foresight",
    "lightgbm": "LightGBM (machine learning)",
    "profile_7d_median": "Median of the past week",
    "naive_last_week": "Same day last week",
    "naive_yesterday": "Yesterday's prices",
    "mstl": "MSTL (statistical)",
}
# Days picked from the 2024 data to explain the idea: one whose perfect-foresight plan is
# simple (cheap midday, dear evening), and one on which a less accurate forecast earned
# more because LightGBM's plan bought into a price spike (labelled at that half-hour).
BATTERY_DAY = "12-25"
FORECAST_DAY, SIMPLE_METHOD, FORECAST_NOTE = "06-24", "naive_last_week", (14.75, "LightGBM's plan buys into this spike")
PLAN_NAMES = {"naive_yesterday": "yesterday's-prices", "naive_last_week": "same-day-last-week", "lightgbm": "LightGBM"}
# Each half-hour's value holds until its end, so a step line starts at midnight.
STEP_HOURS = np.arange(INTERVALS + 1) * INTERVAL_HOURS
SCORE_LABELS = {
    "rank_corr": "Shape (rank correlation)",
    "extreme_hits": "Peak and trough hits",
    "peak_hits": "Peak hits",
    "mae": "Mean absolute error",
    "trough_hits": "Trough hits",
    "spread_error": "Spread error",
}
# A month in which the less accurate forecast earned more (in 2024), labelled on the chart.
EXAMPLE_MONTH = 9
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
    best = out[out["forecaster"] == PERFECT].set_index("month")["revenue"]
    out["capture"] = out["revenue"] / out["month"].map(best)
    return out[out["forecaster"] != PERFECT]


def rc(t: dict) -> dict:
    return {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["secondary"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False, "axes.grid": True, "grid.color": t["grid"],
        "grid.linewidth": 0.75, "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "font.size": 9.5, "legend.frameon": False,
    }  # fmt: skip


def frame(t: dict, title: str, subtitle: str, height: float, left: float = 0.09, top: float = 0.80,
          note: str = BASE_NOTE, rows: int = 1, height_ratios: list[float] | None = None):  # fmt: skip
    """A figure with its title, subtitle and source note placed; returns (fig, ax), or (fig, axes) for rows > 1."""
    fig = Figure(figsize=(8, height))
    ax = fig.subplots(rows, sharex=True, gridspec_kw={"height_ratios": height_ratios})
    fig.subplots_adjust(left=left, right=0.97, top=top, bottom=0.85 / height)
    fig.text(0.09, 1 - 0.22 / height, title, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.09, 1 - 0.47 / height, subtitle, fontsize=9.5, color=t["secondary"], va="top")
    fig.text(0.09, 0.12 / height, note, fontsize=8, color=t["muted"])
    return fig, ax


def step(values: np.ndarray) -> np.ndarray:
    """A day's half-hourly values ready for a step line over STEP_HOURS."""
    return np.r_[values[0], values]


def hour_axis(ax) -> None:
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6), [f"{h:02d}:00" for h in range(0, 25, 6)])


def cover(prices: pd.Series, day: pd.Timestamp, t: dict, path: Path) -> None:
    """A title over the battery day, stripped to the price line and shaded buy and sell hours.

    1280 x 640 px, GitHub's social-preview size; the margins survive Medium's wider crop.
    """
    price = prices_on(prices, day)
    plan = schedule(price, Battery(), INTERVAL_HOURS)
    fig = Figure(figsize=(6.4, 3.2))
    fig.text(0.07, 0.83, "How much is a price forecast worth?", fontsize=17, fontweight="bold", va="center")
    fig.text(0.07, 0.71, "A year of battery trading in Australia's electricity market", fontsize=10.5,
             color=t["secondary"], va="center")  # fmt: skip
    ax = fig.add_axes((0.07, 0.12, 0.86, 0.48))
    for column, color in (("charge_mw", "blue"), ("discharge_mw", "orange")):
        for end in STEP_HOURS[1:][plan[column].to_numpy() > 1]:
            ax.axvspan(end - INTERVAL_HOURS, end, color=t[color], alpha=0.28, linewidth=0)
    ax.plot(STEP_HOURS, step(price), color=t["ink"], linewidth=2.2, drawstyle="steps-pre")
    ax.axhline(0, color=t["muted"], linewidth=0.75)
    low, high = int(price.argmin()), int(price.argmax())
    ax.text(STEP_HOURS[low], price.min() - 28, "buy cheap", ha="center", va="top", fontsize=10, color=t["ink"])
    ax.text(STEP_HOURS[high], price.max() + 18, "sell dear", ha="center", va="bottom", fontsize=10, color=t["ink"])
    ax.set_xlim(0, 24)
    ax.set_ylim(price.min() - 75, price.max() + 60)
    ax.axis("off")
    fig.savefig(path, dpi=200)


def battery_day(prices: pd.Series, day: pd.Timestamp, t: dict, path: Path) -> None:
    """The day's prices on top; below, the plan a battery with perfect foresight would follow."""
    price = prices_on(prices, day)
    plan = schedule(price, Battery(), INTERVAL_HOURS)
    fig, (top, bottom) = frame(t, "A battery buys cheap and sells dear",
                               f"South Australia, {day:%-d %B %Y}: the price, and the plan of a battery that knew it "
                               "in advance", height=5.4, top=0.82, rows=2)  # fmt: skip
    top.plot(STEP_HOURS, step(price), color=t["ink"], linewidth=1.5, drawstyle="steps-pre")
    top.axhline(0, color=t["muted"], linewidth=0.75)
    top.set_ylabel("Price (AUD/MWh)")

    power = (plan["discharge_mw"] - plan["charge_mw"]).to_numpy()
    centres = STEP_HOURS[1:] - INTERVAL_HOURS / 2
    for sign, color in ((1, "orange"), (-1, "blue")):
        bottom.bar(centres, np.where(power * sign > 0, power, 0), width=INTERVAL_HOURS * 0.85, color=t[color])
    bottom.axhline(0, color=t["muted"], linewidth=0.75)
    bottom.set_ylabel("Battery (MW)")
    bottom.legend(handles=[Patch(color=t["blue"], label="Charging: buys energy"),
                           Patch(color=t["orange"], label="Discharging: sells energy")],
                  loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, labelcolor=t["secondary"],
                  borderaxespad=0.2)  # fmt: skip
    hour_axis(bottom)
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(path, dpi=200)


def capture_bars(capture: pd.DataFrame, year: int, t: dict, path: Path) -> None:
    """Each method's capture rate in the base scenario, with perfect foresight as the 100% reference."""
    base = capture[capture["scenario"] == BASE_SCENARIO].sort_values("capture_rate")
    fig, ax = frame(t, "The best forecast captured less than half of what was possible",
                    f"Share of the perfect-foresight revenue each forecast earned, South Australia, {year}",
                    height=3.9, left=0.30, top=0.78)  # fmt: skip
    y = np.arange(len(base))
    colors = [t["other"] if name == PERFECT else t["blue"] for name in base["forecaster"]]
    ax.barh(y, base["capture_rate"], height=0.62, color=colors)
    for i, (name, rate, revenue) in enumerate(
        base[["forecaster", "capture_rate", "revenue_aud"]].itertuples(index=False)
    ):
        if name == PERFECT:  # labelled inside its bar, which runs to the edge
            ax.text(rate - 0.015, i, f"100% = AUD {revenue / 1e6:.1f} million", va="center", ha="right", fontsize=9,
                    color=t["ink"])  # fmt: skip
        else:
            ax.text(rate + 0.012, i, f"{rate:.0%}", va="center", fontsize=9, color=t["secondary"])
    ax.set_yticks(y, [METHOD_LABELS[name] for name in base["forecaster"]])
    ax.tick_params(axis="y", colors=t["secondary"], labelsize=9.5)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([])
    ax.grid(visible=False)
    fig.savefig(path, dpi=200)


def two_forecasts(prices: pd.Series, forecasts: pd.DataFrame, daily: pd.DataFrame, day: pd.Timestamp, simple: str,
                  note: tuple[float, str], t: dict, path: Path) -> None:  # fmt: skip
    """The actual prices with a simple forecast and LightGBM's; below, what each forecast told the battery to do."""
    results = daily[daily["day"] == day].set_index("forecaster")
    actual = prices_on(prices, day)
    fig, (top, *plans) = frame(t, "Closer on average is not the same as right on time",
                               f"South Australia, {day:%-d %B %Y}: two day-ahead forecasts and the plan each led to",
                               height=6.6, top=0.74, rows=3,
                               height_ratios=[3, 1, 1])  # fmt: skip
    top.plot(STEP_HOURS, step(actual), color=t["ink"], linewidth=2.25, drawstyle="steps-pre", label="Actual price",
             zorder=3)  # fmt: skip
    centres = STEP_HOURS[1:] - INTERVAL_HOURS / 2
    for ax, (name, color) in zip(plans, ((simple, "orange"), ("lightgbm", "blue")), strict=True):
        mae, revenue = results.loc[name, ["mae", "revenue_aud"]]
        outcome = f"earned AUD {round(revenue, -2):,.0f}" if revenue >= 0 else f"lost AUD {round(-revenue, -2):,.0f}"
        values = forecasts.loc[(day, name)].to_numpy()
        top.plot(STEP_HOURS, step(values), color=t[color], linewidth=1.5, drawstyle="steps-pre",
                 label=f"{METHOD_LABELS[name]}: missed by {mae:.0f} on average, {outcome}")  # fmt: skip
        plan = schedule(values, Battery(), INTERVAL_HOURS)
        ax.bar(centres, (plan["discharge_mw"] - plan["charge_mw"]).to_numpy(), width=INTERVAL_HOURS * 0.85,
               color=t[color])  # fmt: skip
        ax.axhline(0, color=t["muted"], linewidth=0.75)
        ax.set_ylim(-115, 115)
        ax.set_yticks([-100, 100], ["buy", "sell"])
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Battery plan using the {PLAN_NAMES[name]} forecast", loc="left", fontsize=9,
                     color=t["secondary"], pad=3)  # fmt: skip
    hour, text = note
    top.annotate(text, (hour, actual[int(hour / INTERVAL_HOURS)]), xytext=(-170, 6), textcoords="offset points",
                 fontsize=8.5, color=t["secondary"], va="center",
                 arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.75})  # fmt: skip
    top.axhline(0, color=t["muted"], linewidth=0.75)
    top.set_ylabel("Price (AUD/MWh)")
    top.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=9, labelcolor=t["secondary"],
               borderaxespad=0.3, handlelength=2.2)  # fmt: skip
    hour_axis(plans[-1])
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(path, dpi=200)


def capture_vs_mae(points: pd.DataFrame, year: int, t: dict, path: Path) -> None:
    fig, ax = frame(t, "A lower forecast error does not reliably earn more",
                    f"Capture rate vs forecast error in South Australia, one dot per method and month of {year}",
                    height=5)  # fmt: skip

    ring = {"markersize": MARKER, "markeredgecolor": t["surface"], "markeredgewidth": 1.5, "linestyle": "none"}
    other = points[~points["forecaster"].isin(HIGHLIGHT)]
    ax.plot(other["mae"], other["capture"], "o", color=t["other"], zorder=2, **ring)
    for name, (_, color) in HIGHLIGHT.items():
        mine = points[points["forecaster"] == name]
        ax.plot(mine["mae"], mine["capture"], "o", color=t[color], zorder=3, **ring)

    for name, text, offset in EXAMPLES:
        row = points[(points["forecaster"] == name) & (points["month"].dt.month == EXAMPLE_MONTH)].iloc[0]
        ax.annotate(text.format(mae=row["mae"], capture=row["capture"]), (row["mae"], row["capture"]),
                    xytext=offset, textcoords="offset points", fontsize=8.5, color=t["secondary"],
                    arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.75})  # fmt: skip

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([40, 60, 100, 150, 200, 300]))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}")
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(-0.15, 0.88)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Average error of the day-ahead forecast (AUD/MWh, log scale)")
    ax.set_ylabel("Capture rate")
    ax.axhline(0, color=t["muted"], linewidth=0.75)

    handles = [Line2D([], [], marker="o", color=t[c], label=label, **ring) for label, c in HIGHLIGHT.values()]
    handles.append(Line2D([], [], marker="o", color=t["other"], label="Naive and MSTL baselines", **ring))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9,
              labelcolor=t["secondary"], borderaxespad=0.3, handletextpad=0.3, columnspacing=1.6)  # fmt: skip
    fig.savefig(path, dpi=200)


def score_agreement(table: pd.DataFrame, year: int, t: dict, path: Path) -> None:
    """Every score's agreement with revenue in the base scenario."""
    table = table[table["scenario"] == BASE_SCENARIO].set_index("score").sort_values("daily_agreement")
    fig, ax = frame(t, "The forecast's shape tracks revenue better than its error",
                    "Rank correlation between each score and revenue across the five methods, "
                    f"mean of the days of {year}", height=4.2, left=0.30, top=0.78)  # fmt: skip
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


def sensitivity(table: pd.DataFrame, year: int, t: dict, path: Path) -> None:
    """The shape's lead over MAE in each scenario, top to bottom in the scenarios' order."""
    gaps = table[table["score"] == "rank_corr"].iloc[::-1]
    fig, ax = frame(t, "The shape beats the error for every battery and region",
                    f"How much better the shape ranks the methods' daily revenue than MAE does, {year}",
                    height=4.6, left=0.33, top=0.80,
                    note=SENSITIVITY_NOTE)  # fmt: skip
    y = list(range(len(gaps)))
    ax.errorbar(gaps["gap_to_mae"], y, xerr=gaps["gap_ci95"], fmt="none", ecolor=t["muted"], elinewidth=1.5,
                capsize=0, zorder=2)  # fmt: skip
    ax.plot(gaps["gap_to_mae"], y, "o", color=t["blue"], markersize=MARKER, markeredgecolor=t["surface"],
            markeredgewidth=1.5, linestyle="none", zorder=3)  # fmt: skip
    ax.axvline(0, color=t["muted"], linewidth=0.75)
    ax.text(0.003, len(y) - 0.4, "MAE", fontsize=8.5, color=t["muted"], va="bottom")
    ax.set_yticks(y, gaps["scenario"])
    ax.set_ylim(-0.6, len(y) - 0.2)
    ax.tick_params(axis="y", colors=t["secondary"], labelsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.05, 0.35)
    ax.set_xlabel("Shape's daily agreement with revenue minus MAE's (bars: 95% interval)")
    fig.savefig(path, dpi=200)


def draw_figures(year: int) -> None:
    """Draw every figure in both themes into figures/."""
    daily = pd.read_csv(daily_path("SA1", year), parse_dates=["day"])
    points = monthly(daily)
    agreement = pd.read_csv(agreement_path(year))
    capture = pd.read_csv(capture_path(year))
    days = year_days(year)
    prices = load_prices("SA1", days[0], days[-1] + pd.Timedelta(days=1))["rrp"]
    forecasts = load_forecasts(forecasts_path("SA1", year))
    battery, forecast = pd.Timestamp(f"{year}-{BATTERY_DAY}"), pd.Timestamp(f"{year}-{FORECAST_DAY}")
    FIGURES_DIR.mkdir(exist_ok=True)
    with mpl.rc_context(rc(THEMES["light"])):
        cover(prices, battery, THEMES["light"], FIGURES_DIR / "cover.png")
    for mode, theme in THEMES.items():
        with mpl.rc_context(rc(theme)):
            battery_day(prices, battery, theme, FIGURES_DIR / f"battery_day-{mode}.png")
            capture_bars(capture, year, theme, FIGURES_DIR / f"capture_bars-{mode}.png")
            two_forecasts(
                prices,
                forecasts,
                daily,
                forecast,
                SIMPLE_METHOD,
                FORECAST_NOTE,
                theme,
                FIGURES_DIR / f"two_forecasts-{mode}.png",
            )
            capture_vs_mae(points, year, theme, FIGURES_DIR / f"capture_vs_mae-{mode}.png")
            score_agreement(agreement, year, theme, FIGURES_DIR / f"score_agreement-{mode}.png")
            sensitivity(agreement, year, theme, FIGURES_DIR / f"sensitivity-{mode}.png")
