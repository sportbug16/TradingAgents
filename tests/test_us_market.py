from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingagents.backtesting.investment import InvestmentHorizonSimulator
from tradingagents.backtesting.rules import Signal
from tradingagents.us.market import USInstrumentRegistry, US_TIMEZONE, is_us_market_open, normalize_us_ticker


def _prices():
    return pd.DataFrame(
        [
            {"Date": "2026-01-02", "Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Date": "2026-01-05", "Open": 101, "High": 103, "Low": 100, "Close": 102},
            {"Date": "2026-01-06", "Open": 103, "High": 104, "Low": 102, "Close": 103},
            {"Date": "2026-01-07", "Open": 104, "High": 105, "Low": 103, "Close": 104},
            {"Date": "2026-01-08", "Open": 105, "High": 106, "Low": 104, "Close": 105},
            {"Date": "2026-01-09", "Open": 106, "High": 107, "Low": 105, "Close": 106},
        ]
    )


@pytest.mark.unit
class TestUSMarket:
    def test_normalizes_ticker_for_yahoo_style(self):
        assert normalize_us_ticker(" brk.b ") == "BRK-B"

    def test_registry_accepts_seed_and_rejects_unknown(self):
        registry = USInstrumentRegistry()
        assert registry.resolve("AAPL").exchange == "NASDAQ"
        assert registry.resolve("TSLA").exchange == "NASDAQ"
        assert registry.resolve("BRK.B").symbol == "BRK-B"
        with pytest.raises(ValueError):
            registry.resolve("XYZ")

    def test_market_hours_use_eastern_time(self):
        open_time = datetime(2026, 5, 6, 10, 0, tzinfo=ZoneInfo(US_TIMEZONE))
        closed_time = datetime(2026, 5, 6, 17, 0, tzinfo=ZoneInfo(US_TIMEZONE))
        assert is_us_market_open(open_time)
        assert not is_us_market_open(closed_time)


@pytest.mark.unit
class TestInvestmentHorizonSimulator:
    def test_accepts_week_scale_horizon(self):
        sim = InvestmentHorizonSimulator(holding_sessions=3, transaction_cost_bps=0, slippage_bps=0)
        result = sim.simulate(
            {"AAPL": _prices()},
            [Signal(ticker="AAPL", decision_date="2026-01-02", rating="Buy")],
        )
        trade = result.trades[0]
        assert trade.entry_date == "2026-01-05"
        assert trade.exit_date == "2026-01-08"
        assert trade.net_return == pytest.approx((105 - 101) / 101)

    def test_accepts_month_scale_horizon_config(self):
        sim = InvestmentHorizonSimulator(holding_sessions=60)
        assert sim.holding_sessions == 60
