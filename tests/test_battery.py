from __future__ import annotations

import numpy as np
import pytest

from nem_battery.battery import Battery, schedule, settle

BATTERY = Battery()


def test_flat_prices_do_not_trade():
    plan = schedule(np.full(48, 50.0), BATTERY)
    assert plan[["charge_mw", "discharge_mw"]].to_numpy().sum() == pytest.approx(0)


def test_charges_cheap_and_discharges_dear():
    prices = np.r_[np.full(24, 20.0), np.full(24, 300.0)]
    plan = schedule(prices, BATTERY)
    assert plan["charge_mw"][:24].sum() > 0 and plan["charge_mw"][24:].sum() == pytest.approx(0)
    assert plan["discharge_mw"][24:].sum() > 0 and plan["discharge_mw"][:24].sum() == pytest.approx(0)
    assert settle(plan, prices, BATTERY) > 0


def test_negative_prices_never_charge_and_discharge_at_once():
    prices = np.r_[np.full(24, -200.0), np.full(24, 100.0)]
    plan = schedule(prices, BATTERY)
    assert (np.minimum(plan["charge_mw"], plan["discharge_mw"]) < 1e-6).all()


def test_state_of_charge_stays_in_bounds_and_returns_to_start():
    prices = np.random.default_rng(1).normal(80, 60, 48)
    soc = schedule(prices, BATTERY)["soc_mwh"]
    assert soc.min() >= BATTERY.soc_min_frac * BATTERY.energy_mwh - 1e-6
    assert soc.max() <= BATTERY.energy_mwh + 1e-6
    assert soc.iloc[-1] == pytest.approx(BATTERY.soc_init_frac * BATTERY.energy_mwh)


def test_settle_charges_degradation_on_discharge():
    prices = np.r_[np.full(24, 20.0), np.full(24, 300.0)]
    plan = schedule(prices, BATTERY)
    discharged = plan["discharge_mw"].sum() * 0.5
    net = (plan["discharge_mw"] - plan["charge_mw"]).to_numpy()
    expected = (prices * net).sum() * 0.5 - BATTERY.degradation_aud_per_mwh * discharged
    assert settle(plan, prices, BATTERY) == pytest.approx(expected)
