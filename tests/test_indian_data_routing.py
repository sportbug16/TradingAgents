import pytest
import requests

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.india.data import (
    DhanHQProvider,
    IndianDataFrame,
    IndianDataProviderUnavailable,
    DataSnapshot,
    create_indian_provider,
    dhan_intraday_chunks,
)
from tradingagents.india.market import IndianInstrumentRegistry


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
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response(self.payload)


class _RateLimitResponse:
    status_code = 429
    headers = {}

    def raise_for_status(self):
        raise requests.HTTPError("429 Client Error", response=self)


class _RetrySession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if len(self.calls) == 1:
            return _RateLimitResponse()
        return _Response(self.payload)


@pytest.mark.unit
def test_dhan_stock_data_falls_back_to_yfinance_when_unavailable(monkeypatch):
    calls = []

    def unavailable(*args, **kwargs):
        from tradingagents.india.data import IndianDataProviderUnavailable

        calls.append("dhan")
        raise IndianDataProviderUnavailable("missing credentials")

    def fallback(*args, **kwargs):
        calls.append("yfinance")
        return "fallback csv"

    monkeypatch.setattr(
        "tradingagents.dataflows.interface.VENDOR_METHODS",
        {
            "get_stock_data": {
                "dhan": unavailable,
                "yfinance": fallback,
            }
        },
    )
    monkeypatch.setattr(
        "tradingagents.dataflows.interface.get_vendor",
        lambda category, method=None: "dhan,yfinance",
    )

    assert route_to_vendor("get_stock_data", "RELIANCE.NS", "2026-01-01", "2026-01-05") == "fallback csv"
    assert calls == ["dhan", "yfinance"]


@pytest.mark.unit
def test_dhan_ohlcv_maps_payload_and_uses_cache(tmp_path):
    session = _Session(
        {
            "timestamp": [1767312000],
            "open": [100],
            "high": [110],
            "low": [95],
            "close": [105],
            "volume": [1000],
        }
    )
    instrument = IndianInstrumentRegistry().resolve("RELIANCE.NS")
    provider = DhanHQProvider(access_token="token", cache_dir=tmp_path, session=session)

    first = provider.get_ohlcv(instrument, "2026-01-01", "2026-01-05")
    second = provider.get_ohlcv(instrument, "2026-01-01", "2026-01-05")

    assert first.data.iloc[0]["Close"] == 105
    assert first.snapshot.provider == "dhanhq"
    assert first.snapshot.endpoint == "/charts/historical"
    assert first.snapshot.fallback_unofficial is False
    assert session.calls[0]["json"]["securityId"] == "2885"
    assert session.calls[0]["json"]["exchangeSegment"] == "NSE_EQ"
    assert len(session.calls) == 1
    assert second.snapshot.cache_key == first.snapshot.cache_key


@pytest.mark.unit
def test_dhan_ohlcv_retries_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.india.data.time.sleep", lambda *_: None)
    session = _RetrySession(
        {
            "timestamp": [1767312000],
            "open": [100],
            "high": [110],
            "low": [95],
            "close": [105],
            "volume": [1000],
        }
    )
    provider = DhanHQProvider(
        access_token="token",
        cache_dir=tmp_path,
        session=session,
        max_retries=1,
        retry_base_seconds=0.0,
    )

    result = provider.get_ohlcv(IndianInstrumentRegistry().resolve("RELIANCE.NS"), "2026-01-01", "2026-01-05")

    assert result.data.iloc[0]["Close"] == 105
    assert len(session.calls) == 2


@pytest.mark.unit
def test_dhan_benchmark_uses_index_segment(tmp_path):
    session = _Session(
        {
            "timestamp": [1767312000],
            "open": [24000],
            "high": [24100],
            "low": [23900],
            "close": [24050],
            "volume": [0],
        }
    )
    provider = DhanHQProvider(access_token="token", cache_dir=tmp_path, session=session)

    result = provider.get_benchmark_ohlcv("^NSEI", "2026-01-01", "2026-01-05")

    assert result.snapshot.provider == "dhanhq"
    assert result.snapshot.fallback_unofficial is False
    assert session.calls[0]["json"]["securityId"] == "13"
    assert session.calls[0]["json"]["exchangeSegment"] == "IDX_I"
    assert session.calls[0]["json"]["instrument"] == "INDEX"


@pytest.mark.unit
def test_dhan_requires_security_id(tmp_path):
    provider = DhanHQProvider(access_token="token", cache_dir=tmp_path)
    instrument = IndianInstrumentRegistry().resolve("RELIANCE.BO")
    with pytest.raises(IndianDataProviderUnavailable, match="missing Dhan securityId"):
        provider.get_ohlcv(instrument, "2026-01-01", "2026-01-05")


@pytest.mark.unit
def test_create_indian_provider_is_dhan_only(monkeypatch):
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    with pytest.raises(IndianDataProviderUnavailable):
        create_indian_provider("dhan", allow_fallback=False)
    with pytest.raises(IndianDataProviderUnavailable):
        create_indian_provider("dhan", allow_fallback=True)
    with pytest.raises(ValueError, match="yfinance is not supported"):
        create_indian_provider("yfinance", allow_fallback=True)


@pytest.mark.unit
def test_dhan_intraday_chunks_are_limited_to_90_days():
    chunks = dhan_intraday_chunks("2026-01-01", "2026-04-15")
    assert chunks == [
        ("2026-01-01", "2026-03-31"),
        ("2026-04-01", "2026-04-15"),
    ]
