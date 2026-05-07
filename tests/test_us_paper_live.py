import pandas as pd
import pytest

import scripts.us_paper_live as us_paper_live
from tradingagents.broker import PaperBroker
from tradingagents.us.data import USDataFrame, USDataSnapshot


@pytest.mark.unit
def test_us_paper_live_blocks_orders_when_market_closed(monkeypatch):
    prices = pd.DataFrame(
        [
            {"Date": f"2026-01-{i + 1:02d}", "Open": 100 + i, "High": 100 + i, "Low": 100 + i, "Close": 100 + i}
            for i in range(70)
        ]
    ).set_index("Date")

    class Provider:
        def get_ohlcv(self, *args, **kwargs):
            return USDataFrame(
                prices.reset_index(),
                USDataSnapshot(
                    provider="fixture",
                    endpoint="fixture",
                    params={},
                    adjusted=True,
                    fetched_at="2026-01-01T00:00:00Z",
                ),
            )

    broker = PaperBroker(cash=100_000)
    events = us_paper_live._poll_once(
        ["AAPL"],
        broker,
        Provider(),
        order_value=5_000,
        fast_window=20,
        slow_window=60,
        adjusted=True,
        allow_trading=False,
    )
    assert events[0]["signal"] == "BUY"
    assert events[0]["order"] is None
    assert events[0]["blocked_reason"] == "market_closed"
    assert broker.positions() == []
