import pandas as pd
import pytest

from tradingagents.backtesting.baselines import (
    buy_and_hold_baseline,
    equal_weight_baseline,
    low_volatility_baseline,
    random_entry_baseline,
    simple_momentum_baseline,
)


def _prices(values):
    return pd.DataFrame(
        [
            {"Date": day.date().isoformat(), "Open": v, "High": v, "Low": v, "Close": v}
            for day, v in zip(pd.bdate_range("2026-01-01", periods=len(values)), values)
        ]
    )


@pytest.mark.unit
class TestBaselines:
    def test_buy_and_hold_return(self):
        result = buy_and_hold_baseline("nifty", _prices([100, 105, 110]))
        assert result.total_return == pytest.approx(0.10)

    def test_equal_weight_average(self):
        result = equal_weight_baseline({"a": _prices([100, 110]), "b": _prices([100, 90])})
        assert result.returns == pytest.approx([0.0])

    def test_momentum_selects_best_forward_return(self):
        result = simple_momentum_baseline(
            {"a": _prices([100, 110, 111, 120]), "b": _prices([100, 90, 91, 92])},
            lookback_sessions=1,
            hold_sessions=1,
            top_n=1,
        )
        assert len(result.returns) == 1
        assert result.returns[0] > 0

    def test_random_entry_is_seeded(self):
        prices = {"a": _prices([100, 101, 102, 103, 104])}
        first = random_entry_baseline(prices, hold_sessions=1, seed=7)
        second = random_entry_baseline(prices, hold_sessions=1, seed=7)
        assert first.returns == second.returns

    def test_low_volatility_selects_lowest_vol_names(self):
        calm = [100 + i for i in range(90)]
        choppy = [100, 120] * 45
        result = low_volatility_baseline(
            {"calm": _prices(calm), "choppy": _prices(choppy)},
            lookback_sessions=20,
            hold_sessions=5,
            top_n=1,
        )
        assert result.name == "low_volatility"
        assert len(result.returns) == 1
