# How much is a price forecast worth?

*Battery arbitrage in Australia's National Electricity Market (NEM). Work in progress.*

A grid battery earns money by charging when electricity is cheap and discharging when it is expensive, so every decision rests on a price forecast. This project asks whether a forecast that is more accurate by the usual metrics (MAE) actually earns more. It backtests a year of day-ahead scheduling on public AEMO data, optimizing each day with a mixed-integer linear program.

> **Main figure (to come):** capture rate vs. forecast MAE, one point per method and month.

## Data

AEMO's public 5-minute regional reference price (RRP, AUD/MWh) and demand, averaged to 30-minute intervals. Downloaded on demand by `src/nem_battery/data.py`.

## Model

For each day, choose charge $c_t$ and discharge $d_t$ (MW) in each of 48 half-hours to maximize

$$\sum_t p_t\,(d_t - c_t)\,\Delta t \;-\; \kappa \sum_t d_t\,\Delta t$$

subject to state-of-charge dynamics $s_t = s_{t-1} + \eta_c c_t \Delta t - d_t \Delta t / \eta_d$, capacity and power limits, and $s_{48} = s_0$. Here $p_t$ is the price, $\Delta t$ the interval length, $\eta_c, \eta_d$ the charge and discharge efficiencies and $\kappa$ a degradation cost per MWh.

A binary mode variable stops the battery from charging and discharging in the same interval. Without it, a linear program would do exactly that whenever prices go negative, which they often do in South Australia: it would burn energy through conversion losses to be paid for consuming. The battery is 100 MW / 200 MWh with ~85% round-trip efficiency.

## Backtest

For each day $D$:
1. forecast its 48 prices using data up to $D-1$;
2. optimize on the forecast;
3. settle the plan at the prices that actually cleared.

The yardstick is the revenue under perfect foresight, so the headline metric is the **capture rate**: realized revenue ÷ perfect-foresight revenue.

## Preliminary results (SA1, 2024, naive baselines only)

| Forecast | MAE (AUD/MWh) | Revenue (AUD m) | Capture rate |
|---|---|---|---|
| Perfect foresight | 0 | 33.5 | 100% |
| 7-day median profile | 80 | 14.6 | 44% |
| Same day last week | 123 | 11.0 | 33% |
| Yesterday | 106 | 9.7 | 29% |

Over the year, lower MAE goes with higher revenue. In June 2024 the order flips: the least accurate baseline earned the most. Finding out when and why that happens is the point of the next steps.

## Limitations

- **Day-ahead plan:** the plan is fixed a day ahead. Real batteries rebid every five minutes, so these figures are a floor.
- **No ancillary services:** frequency-control (FCAS) revenue is excluded.
- **Price taker:** the battery does not move the price.

## Reproduce

```bash
uv sync
uv run python scripts/run_baseline.py --region SA1 --start 2024-01-01 --end 2025-01-01
```
