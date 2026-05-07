import json
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.us.data import (
    MassiveUSDataProvider,
    YFinanceUSDataProvider,
    create_us_data_provider,
    validate_bar_coverage,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return _Response(self.payload)


@pytest.mark.unit
def test_massive_ohlcv_maps_aggregate_rows_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "massive-key")
    session = _Session(
        {
            "adjusted": True,
            "results": [
                {"t": 1767312000000, "o": 100, "h": 110, "l": 90, "c": 105, "v": 1000, "vw": 102, "n": 12}
            ],
        }
    )
    provider = MassiveUSDataProvider(cache_dir=tmp_path, session=session)
    result = provider.get_ohlcv("aapl", "2026-01-02", "2026-01-03")

    assert list(result.data.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"]
    assert result.data.iloc[0]["Close"] == 105
    assert result.snapshot.provider == "massive"
    assert result.snapshot.adjusted is True
    assert "apiKey=massive-key" in session.urls[0]

    provider.get_ohlcv("AAPL", "2026-01-02", "2026-01-03")
    assert len(session.urls) == 1


@pytest.mark.unit
def test_yfinance_snapshot_is_labeled_unofficial(monkeypatch):
    frame = pd.DataFrame(
        [{"Date": "2026-01-02", "Open": 1, "High": 2, "Low": 1, "Close": 2, "Volume": 10}]
    ).set_index("Date")
    monkeypatch.setattr("tradingagents.us.data.yf.download", lambda *args, **kwargs: frame)
    result = YFinanceUSDataProvider().get_ohlcv("AAPL", "2026-01-01", "2026-01-03")
    assert result.snapshot.fallback_unofficial is True
    assert result.snapshot.provider == "yfinance"


@pytest.mark.unit
def test_provider_factory_falls_back_to_yfinance_without_massive_key(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = create_us_data_provider(
        {"data_cache_dir": "/tmp/tradingagents-test-cache", "us": {"data_provider": "massive", "fallback_provider": "yfinance"}}
    )
    assert provider.name == "yfinance"


@pytest.mark.unit
def test_validate_bar_coverage_rejects_sparse_daily_data():
    frame = pd.DataFrame([{"Date": "2026-01-02", "Open": 1, "High": 1, "Low": 1, "Close": 1}])
    with pytest.raises(RuntimeError):
        validate_bar_coverage(frame, "2026-01-01", "2026-01-10", max_missing_ratio=0.01)
