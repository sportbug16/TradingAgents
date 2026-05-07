import pandas as pd
import pytest

from tradingagents.backtesting.portfolio import CorrelationRiskModel, PortfolioBacktestSimulator
from tradingagents.backtesting.rules import Signal


def _prices(start=100, step=1):
    rows = []
    price = start
    for day in pd.bdate_range("2026-01-01", periods=35):
        rows.append(
            {
                "Date": day.date().isoformat(),
                "Open": price,
                "High": price + 1,
                "Low": price - 1,
                "Close": price + step,
                "Volume": 1000,
            }
        )
        price += step
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_correlation_risk_model_detects_correlated_names():
    prices = {"AAA": _prices(100, 1), "BBB": _prices(50, 1), "CCC": _prices(200, -1)}
    risk = CorrelationRiskModel(lookback_sessions=20, threshold=0.75)
    weight = risk.cluster_weight("BBB", {"AAA": 0.25, "CCC": 0.25}, prices, "2026-02-01")
    assert weight >= 0.25


@pytest.mark.unit
def test_portfolio_simulator_skips_cluster_over_cap():
    prices = {"AAA": _prices(100, 1), "BBB": _prices(50, 1)}
    sim = PortfolioBacktestSimulator(
        holding_sessions=5,
        max_single_name_exposure=0.25,
        max_correlation_cluster_exposure=0.25,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    result = sim.simulate(
        prices,
        [
            Signal(ticker="AAA", decision_date="2026-01-05", rating="Buy"),
            Signal(ticker="BBB", decision_date="2026-01-05", rating="Buy"),
        ],
    )
    assert len(result.trades) == 1
    assert result.skipped_signals[0]["reason"] == "max_correlation_cluster_exposure"
    assert result.equity_curve
