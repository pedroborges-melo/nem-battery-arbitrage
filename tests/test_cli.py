from __future__ import annotations

import pytest

from nem_battery.cli import main


def help_text(args: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as exit_:
        main([*args, "--help"])
    assert exit_.value.code == 0
    return capsys.readouterr().out


def test_help_lists_the_three_steps(capsys):
    out = help_text([], capsys)
    assert all(step in out for step in ("forecast", "evaluate", "figures"))


@pytest.mark.parametrize("command", ["forecast", "evaluate", "figures"])
def test_every_step_takes_a_year(command, capsys):
    assert "--year" in help_text([command], capsys)


def test_unknown_region_is_rejected():
    with pytest.raises(SystemExit):
        main(["forecast", "--region", "WA1"])
