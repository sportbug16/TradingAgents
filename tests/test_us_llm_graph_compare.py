import json

import pandas as pd
import pytest

import scripts.us_llm_graph_compare as compare
from tradingagents.us.data import USDataFrame, USDataSnapshot


@pytest.mark.unit
def test_missing_provider_keys_are_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(compare, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    prices = pd.DataFrame(
        [
            {"Date": "2026-01-02", "Open": 100, "High": 101, "Low": 99, "Close": 100},
            {"Date": "2026-01-05", "Open": 101, "High": 102, "Low": 100, "Close": 101},
            {"Date": "2026-01-06", "Open": 102, "High": 103, "Low": 101, "Close": 102},
        ]
    )
    snapshot = USDataSnapshot(
        provider="fixture",
        endpoint="fixture",
        params={},
        adjusted=True,
        fetched_at="2026-01-01T00:00:00Z",
    )

    class Provider:
        name = "fixture"

        def get_ohlcv(self, *args, **kwargs):
            return USDataFrame(prices, snapshot)

        def get_benchmark_ohlcv(self, *args, **kwargs):
            return USDataFrame(prices, snapshot)

    monkeypatch.setattr(compare, "create_us_data_provider", lambda *args, **kwargs: Provider())

    out = tmp_path / "compare.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "us_llm_graph_compare.py",
            "--tickers",
            "AAPL",
            "--dates",
            "2026-01-02",
            "--providers",
            "anthropic,google",
            "--output",
            str(out),
        ],
    )

    compare.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["providers"]["anthropic"]["status"] == "skipped"
    assert data["providers"]["google"]["status"] == "skipped"
