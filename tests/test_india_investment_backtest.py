import json

import pandas as pd
import pytest

import scripts.india_investment_backtest as india_investment_backtest
from tradingagents.india.data import DataSnapshot, IndianDataFrame


def _prices(start=100, step=1, periods=180):
    rows = []
    price = start
    for day in pd.bdate_range("2025-01-01", periods=periods):
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


class _Provider:
    name = "fixture"

    def get_ohlcv(self, instrument, start, end):
        return IndianDataFrame(
            _prices(),
            DataSnapshot(
                provider="fixture",
                endpoint="bars",
                params={"ticker": instrument.data_symbol},
                adjusted=False,
                fetched_at="2026-01-01T00:00:00Z",
            ),
        )

    def get_benchmark_ohlcv(self, benchmark, start, end):
        return IndianDataFrame(
            _prices(300, 1),
            DataSnapshot(
                provider="fixture",
                endpoint="benchmark",
                params={"ticker": benchmark},
                adjusted=False,
                fetched_at="2026-01-01T00:00:00Z",
            ),
        )


@pytest.mark.unit
def test_india_investment_horizons_support_monthly_and_half_year():
    prices = {"RELIANCE.NS": _prices()}
    signals = india_investment_backtest._investment_signals(prices, lookback=20, max_horizon=126)
    assert signals
    assert india_investment_backtest._horizons("20,60,126") == [20, 60, 126]


@pytest.mark.unit
def test_india_investment_backtest_writes_expected_sections(monkeypatch, tmp_path):
    monkeypatch.setattr(india_investment_backtest, "create_indian_provider", lambda *args, **kwargs: _Provider())
    output = tmp_path / "india_investment.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "india_investment_backtest.py",
            "--tickers",
            "RELIANCE.NS",
            "--horizons",
            "5,20,60",
            "--lookback",
            "20",
            "--output",
            str(output),
        ],
    )

    india_investment_backtest.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["study_type"] == "india_investment_swing"
    assert payload["horizons"] == [5, 20, 60]
    assert "low_volatility" in payload["baselines"]
    assert "rating_bucket_forward_returns" in payload["horizon_results"]["20"]
