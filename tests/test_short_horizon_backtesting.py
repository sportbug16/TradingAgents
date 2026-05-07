import json

import pandas as pd
import pytest

from tradingagents.backtesting.experiments import ExperimentRegistry
from tradingagents.backtesting.rules import RatingPolicy, Signal, rating_to_target_weight
from tradingagents.backtesting.simulator import ShortHorizonSimulator


def _prices():
    return pd.DataFrame(
        [
            {"Date": "2026-01-05", "Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Date": "2026-01-06", "Open": 102, "High": 104, "Low": 101, "Close": 103},
            {"Date": "2026-01-07", "Open": 105, "High": 106, "Low": 104, "Close": 105},
            {"Date": "2026-01-08", "Open": 107, "High": 108, "Low": 106, "Close": 107},
            {"Date": "2026-01-09", "Open": 110, "High": 111, "Low": 109, "Close": 110},
        ]
    )


@pytest.mark.unit
class TestRatingPolicy:
    def test_maps_5_tier_rating_to_long_only_weights(self):
        assert rating_to_target_weight("Buy") == 1.0
        assert rating_to_target_weight("Overweight") == 0.5
        assert rating_to_target_weight("Hold") == 0.0
        assert rating_to_target_weight("Underweight") == 0.0
        assert rating_to_target_weight("Sell") == 0.0

    def test_custom_cap_applies(self):
        policy = RatingPolicy(max_position_weight=0.25)
        assert rating_to_target_weight("Buy", policy) == 0.25


@pytest.mark.unit
class TestShortHorizonSimulator:
    def test_enters_next_session_open_and_exits_after_horizon(self):
        sim = ShortHorizonSimulator(holding_sessions=3, transaction_cost_bps=0, slippage_bps=0)
        result = sim.simulate(
            {"RELIANCE.NS": _prices()},
            [Signal(ticker="RELIANCE.NS", decision_date="2026-01-05", rating="Buy")],
        )
        trade = result.trades[0]
        assert trade.entry_date == "2026-01-06"
        assert trade.exit_date == "2026-01-09"
        assert trade.entry_price == 102
        assert trade.exit_price == 110
        assert trade.net_return == pytest.approx((110 - 102) / 102)

    def test_hold_signal_creates_no_trade(self):
        sim = ShortHorizonSimulator(holding_sessions=1)
        result = sim.simulate(
            {"RELIANCE.NS": _prices()},
            [Signal(ticker="RELIANCE.NS", decision_date="2026-01-05", rating="Hold")],
        )
        assert result.trades == []
        assert result.metrics.trade_count == 0

    def test_stop_loss_exits_early(self):
        prices = _prices()
        prices.loc[1, "Low"] = 95
        sim = ShortHorizonSimulator(
            holding_sessions=3,
            transaction_cost_bps=0,
            slippage_bps=0,
            stop_loss_pct=0.05,
        )
        result = sim.simulate(
            {"RELIANCE.NS": prices},
            [Signal(ticker="RELIANCE.NS", decision_date="2026-01-05", rating="Buy")],
        )
        trade = result.trades[0]
        assert trade.exit_date == "2026-01-06"
        assert trade.exit_price == pytest.approx(102 * 0.95)


@pytest.mark.unit
class TestExperimentRegistry:
    def test_jsonl_registry_round_trip(self, tmp_path):
        registry = ExperimentRegistry(tmp_path)
        registry.record({"run_id": "r1", "ticker": "RELIANCE.NS", "rating": "Buy"})
        assert registry.load() == [{"run_id": "r1", "ticker": "RELIANCE.NS", "rating": "Buy"}]
