"""Day-ahead battery scheduling as a mixed-integer linear program (PuLP + CBC).

The binary mode variable stops the battery from charging and discharging in the
same interval. Without it, an LP would do exactly that when prices are negative,
burning energy through conversion losses to get paid for consumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp


@dataclass(frozen=True)
class Battery:
    power_mw: float = 100.0  # charge and discharge limit
    energy_mwh: float = 200.0  # usable capacity (2-hour battery)
    eta_charge: float = 0.92  # round-trip efficiency ~0.85
    eta_discharge: float = 0.92
    soc_min_frac: float = 0.10
    soc_init_frac: float = 0.50  # start and end every day here, so days are independent
    degradation_aud_per_mwh: float = 10.0  # cost per MWh discharged, discourages marginal cycles


def schedule(prices: np.ndarray, battery: Battery, dt_hours: float = 0.5) -> pd.DataFrame:
    """Return the revenue-maximising charge/discharge plan for one price vector."""
    n = len(prices)
    b = battery
    soc_min, soc_max = b.soc_min_frac * b.energy_mwh, b.energy_mwh
    soc0 = b.soc_init_frac * b.energy_mwh

    prob = pulp.LpProblem("battery", pulp.LpMaximize)
    c = pulp.LpVariable.dicts("charge", range(n), 0, b.power_mw)
    d = pulp.LpVariable.dicts("discharge", range(n), 0, b.power_mw)
    soc = pulp.LpVariable.dicts("soc", range(n), soc_min, soc_max)
    mode = pulp.LpVariable.dicts("charging", range(n), cat="Binary")

    prob += pulp.lpSum(
        float(prices[t]) * (d[t] - c[t]) * dt_hours - b.degradation_aud_per_mwh * d[t] * dt_hours for t in range(n)
    )
    for t in range(n):
        prev = soc0 if t == 0 else soc[t - 1]
        prob += soc[t] == prev + b.eta_charge * c[t] * dt_hours - d[t] * dt_hours / b.eta_discharge
        prob += c[t] <= b.power_mw * mode[t]
        prob += d[t] <= b.power_mw * (1 - mode[t])
    prob += soc[n - 1] == soc0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver status: {pulp.LpStatus[status]}")
    return pd.DataFrame(
        {
            "charge_mw": [c[t].value() for t in range(n)],
            "discharge_mw": [d[t].value() for t in range(n)],
            "soc_mwh": [soc[t].value() for t in range(n)],
        }
    )


def settle(plan: pd.DataFrame, actual_prices: np.ndarray, battery: Battery, dt_hours: float = 0.5) -> float:
    """Revenue (AUD) of executing `plan` at the prices that actually cleared."""
    net_mw = plan["discharge_mw"].to_numpy() - plan["charge_mw"].to_numpy()
    energy = float((actual_prices * net_mw).sum() * dt_hours)
    wear = battery.degradation_aud_per_mwh * float(plan["discharge_mw"].sum() * dt_hours)
    return energy - wear
