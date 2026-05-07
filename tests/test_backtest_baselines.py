import pandas as pd
import pytest

from tradingagents.backtesting.baselines import (
    buy_and_hold_baseline,
    equal_weight_baseline,
    random_entry_baseline,
    simple_momentum_baseline,
)


def _prices(values):
    return pd.DataFrame(
        [
            {"Date": f"2026-01-{i + 1:02d}", "Open": v, "High": v, "Low": v, "Close": v}
            for i, v in enumerate(values)
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
