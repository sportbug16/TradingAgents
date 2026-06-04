import pandas as pd
import pytest

import scripts.india_paper_live as india_paper_live
from tradingagents.broker import BrokerPosition, PaperBroker
from tradingagents.india.data import DataSnapshot, IndianDataFrame
from tradingagents.india.market import IndianInstrumentRegistry


class _Provider:
    name = "fixture"

    def get_ohlcv(self, instrument, start, end):
        rows = []
        for idx, day in enumerate(pd.bdate_range("2026-01-01", periods=20)):
            rows.append(
                {
                    "Date": day.date().isoformat(),
                    "Open": 100 + idx,
                    "High": 100 + idx,
                    "Low": 100 + idx,
                    "Close": 100 + idx,
                    "Volume": 1000,
                }
            )
        return IndianDataFrame(
            pd.DataFrame(rows),
            DataSnapshot(
                provider="fixture",
                endpoint="bars",
                params={"ticker": instrument.data_symbol},
                adjusted=False,
                fetched_at="2026-01-01T00:00:00Z",
            ),
        )


@pytest.mark.unit
def test_india_paper_live_blocks_orders_when_market_closed():
    broker = PaperBroker(cash=100_000)
    events = india_paper_live._poll_once(
        ["RELIANCE.NS"],
        IndianInstrumentRegistry(),
        _Provider(),
        broker,
        order_value=5_000,
        lookback_days=10,
        allow_trading=False,
    )

    assert events[0]["signal"] == "BUY"
    assert events[0]["order"] is None
    assert events[0]["blocked_reason"] == "market_closed"
    assert events[0]["data_snapshot"]["provider"] == "fixture"
    assert broker.positions() == []


@pytest.mark.unit
def test_india_paper_live_prevents_duplicate_position():
    broker = PaperBroker(
        cash=100_000,
        positions=[BrokerPosition(symbol="RELIANCE.NS", quantity=1, average_price=100, last_price=100)],
    )
    events = india_paper_live._poll_once(
        ["RELIANCE.NS"],
        IndianInstrumentRegistry(),
        _Provider(),
        broker,
        order_value=5_000,
        lookback_days=10,
        allow_trading=True,
    )

    assert events[0]["signal"] == "BUY"
    assert events[0]["order"] is None
    assert events[0]["blocked_reason"] == "existing_position"


@pytest.mark.unit
def test_india_paper_live_records_provider_errors():
    class BadProvider:
        name = "bad"

        def get_ohlcv(self, *args, **kwargs):
            raise RuntimeError("provider down")

    events = india_paper_live._poll_once(
        ["RELIANCE.NS"],
        IndianInstrumentRegistry(),
        BadProvider(),
        PaperBroker(cash=100_000),
        order_value=5_000,
        lookback_days=10,
        allow_trading=True,
    )

    assert events[0]["error"] == "provider down"
    assert events[0]["data_provider"] == "bad"


@pytest.mark.unit
def test_india_paper_live_loads_json_watchlist(monkeypatch, tmp_path):
    watchlist = tmp_path / "watchlist.json"
    watchlist.write_text('{"tickers": ["reliance.ns", "TCS.NS"]}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["india_paper_live.py"])
    args = india_paper_live._parse_args()
    args.tickers_file = str(watchlist)

    assert india_paper_live._load_tickers(args) == ["RELIANCE.NS", "TCS.NS"]


@pytest.mark.unit
def test_india_paper_live_does_not_import_full_graph():
    assert not hasattr(india_paper_live, "TradingAgentsGraph")
