"""US market data provider interfaces and adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests
import yfinance as yf

from tradingagents.default_config import DEFAULT_CONFIG

from .market import DEFAULT_US_BENCHMARK, USInstrument, normalize_us_ticker


MASSIVE_API_BASE_URL = "https://api.massive.com"


@dataclass(frozen=True)
class USDataSnapshot:
    provider: str
    endpoint: str
    params: dict[str, Any]
    adjusted: bool | None
    fetched_at: str
    cache_key: str | None = None
    fallback_unofficial: bool = False


@dataclass(frozen=True)
class USDataFrame:
    data: pd.DataFrame
    snapshot: USDataSnapshot


class USMarketDataProvider(Protocol):
    name: str

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        multiplier: int = 1,
        timespan: str = "day",
        adjusted: bool = True,
    ) -> USDataFrame:
        ...

    def get_benchmark_ohlcv(self, benchmark: str, start: str, end: str, *, adjusted: bool = True) -> USDataFrame:
        ...

    def get_instrument(self, ticker: str, as_of: str | None = None) -> dict[str, Any]:
        ...

    def get_corporate_actions(self, ticker: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        ...

    def get_market_holidays(self) -> list[dict[str, Any]]:
        ...

    def is_market_open(self) -> bool:
        ...


class MassiveUSDataProvider:
    """Massive/Polygon REST adapter for US stocks and benchmarks."""

    name = "massive"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = MASSIVE_API_BASE_URL,
        cache_dir: str | Path | None = None,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY is required for MassiveUSDataProvider")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir or DEFAULT_CONFIG["data_cache_dir"]) / "massive"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        multiplier: int = 1,
        timespan: str = "day",
        adjusted: bool = True,
    ) -> USDataFrame:
        symbol = normalize_us_ticker(ticker)
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}
        payload, cache_key = self._get_json(endpoint, params)
        rows = [
            {
                "Date": pd.to_datetime(item["t"], unit="ms").date().isoformat(),
                "Open": item.get("o"),
                "High": item.get("h"),
                "Low": item.get("l"),
                "Close": item.get("c"),
                "Volume": item.get("v"),
                "VWAP": item.get("vw"),
                "Transactions": item.get("n"),
            }
            for item in payload.get("results", [])
        ]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame[["Date", "Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"]]
        snapshot = USDataSnapshot(
            provider=self.name,
            endpoint=endpoint,
            params=params,
            adjusted=payload.get("adjusted", adjusted),
            fetched_at=_utc_now(),
            cache_key=cache_key,
        )
        return USDataFrame(data=frame, snapshot=snapshot)

    def get_benchmark_ohlcv(self, benchmark: str, start: str, end: str, *, adjusted: bool = True) -> USDataFrame:
        return self.get_ohlcv(benchmark, start, end, adjusted=adjusted)

    def get_instrument(self, ticker: str, as_of: str | None = None) -> dict[str, Any]:
        endpoint = f"/v3/reference/tickers/{normalize_us_ticker(ticker)}"
        params = {"date": as_of} if as_of else {}
        payload, _ = self._get_json(endpoint, params)
        return payload.get("results", {})

    def get_corporate_actions(self, ticker: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        symbol = normalize_us_ticker(ticker)
        params = {"ticker": symbol, "limit": 1000}
        if start:
            params["ex_dividend_date.gte"] = start
            params["execution_date.gte"] = start
        if end:
            params["ex_dividend_date.lte"] = end
            params["execution_date.lte"] = end
        dividends, _ = self._get_json("/v3/reference/dividends", params)
        splits, _ = self._get_json("/v3/reference/splits", params)
        return {
            "provider": self.name,
            "ticker": symbol,
            "dividends": dividends.get("results", []),
            "splits": splits.get("results", []),
        }

    def get_market_holidays(self) -> list[dict[str, Any]]:
        payload, _ = self._get_json("/v1/marketstatus/upcoming", {})
        return payload if isinstance(payload, list) else []

    def is_market_open(self) -> bool:
        payload, _ = self._get_json("/v1/marketstatus/now", {})
        market = payload.get("market")
        return market == "open"

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> tuple[Any, str]:
        params = {k: v for k, v in params.items() if v is not None}
        cache_key = _cache_key(endpoint, params)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), cache_key
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        query = dict(params)
        query["apiKey"] = self.api_key
        response = self.session.get(f"{url}?{urlencode(query)}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload, cache_key


class YFinanceUSDataProvider:
    """Unofficial yfinance fallback for low-cost local experimentation."""

    name = "yfinance"

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        multiplier: int = 1,
        timespan: str = "day",
        adjusted: bool = True,
    ) -> USDataFrame:
        interval = _yfinance_interval(multiplier, timespan)
        data = yf.download(
            normalize_us_ticker(ticker),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=adjusted,
            multi_level_index=False,
            progress=False,
        )
        if data.empty:
            raise RuntimeError(f"no data downloaded for {ticker}")
        frame = data.reset_index()
        date_col = "Datetime" if "Datetime" in frame.columns else "Date"
        frame["Date"] = pd.to_datetime(frame[date_col]).dt.date.astype(str)
        columns = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
        snapshot = USDataSnapshot(
            provider=self.name,
            endpoint="yf.download",
            params={
                "ticker": normalize_us_ticker(ticker),
                "start": start,
                "end": end,
                "interval": interval,
                "auto_adjust": adjusted,
            },
            adjusted=adjusted,
            fetched_at=_utc_now(),
            fallback_unofficial=True,
        )
        return USDataFrame(data=frame[columns].copy(), snapshot=snapshot)

    def get_benchmark_ohlcv(self, benchmark: str, start: str, end: str, *, adjusted: bool = True) -> USDataFrame:
        return self.get_ohlcv(benchmark, start, end, adjusted=adjusted)

    def get_instrument(self, ticker: str, as_of: str | None = None) -> dict[str, Any]:
        symbol = normalize_us_ticker(ticker)
        return {"ticker": symbol, "active": True, "source": "yfinance", "fallback_unofficial": True}

    def get_corporate_actions(self, ticker: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        return {"provider": self.name, "ticker": normalize_us_ticker(ticker), "dividends": [], "splits": [], "fallback_unofficial": True}

    def get_market_holidays(self) -> list[dict[str, Any]]:
        return []

    def is_market_open(self) -> bool:
        from .market import is_us_market_open

        return is_us_market_open()


class FallbackUSDataProvider:
    """Primary provider with explicit unofficial fallback labeling."""

    def __init__(self, primary: USMarketDataProvider, fallback: USMarketDataProvider):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name

    def get_ohlcv(self, *args, **kwargs) -> USDataFrame:
        try:
            return self.primary.get_ohlcv(*args, **kwargs)
        except Exception:
            return self.fallback.get_ohlcv(*args, **kwargs)

    def get_benchmark_ohlcv(self, *args, **kwargs) -> USDataFrame:
        try:
            return self.primary.get_benchmark_ohlcv(*args, **kwargs)
        except Exception:
            return self.fallback.get_benchmark_ohlcv(*args, **kwargs)

    def get_instrument(self, *args, **kwargs) -> dict[str, Any]:
        try:
            return self.primary.get_instrument(*args, **kwargs)
        except Exception:
            return self.fallback.get_instrument(*args, **kwargs)

    def get_corporate_actions(self, *args, **kwargs) -> dict[str, Any]:
        try:
            return self.primary.get_corporate_actions(*args, **kwargs)
        except Exception:
            return self.fallback.get_corporate_actions(*args, **kwargs)

    def get_market_holidays(self) -> list[dict[str, Any]]:
        try:
            return self.primary.get_market_holidays()
        except Exception:
            return self.fallback.get_market_holidays()

    def is_market_open(self) -> bool:
        try:
            return self.primary.is_market_open()
        except Exception:
            return self.fallback.is_market_open()


def create_us_data_provider(config: dict[str, Any] | None = None) -> USMarketDataProvider:
    config = config or DEFAULT_CONFIG
    us_config = config.get("us", {})
    provider_name = (us_config.get("data_provider") or "massive").lower()
    fallback_name = (us_config.get("fallback_provider") or "yfinance").lower()
    cache_dir = config.get("data_cache_dir") or DEFAULT_CONFIG["data_cache_dir"]
    if fallback_name and fallback_name != provider_name:
        try:
            primary = _create_provider(provider_name, cache_dir)
        except Exception:
            return _create_provider(fallback_name, cache_dir)
        return FallbackUSDataProvider(primary, _create_provider(fallback_name, cache_dir))
    primary = _create_provider(provider_name, cache_dir)
    return primary


def snapshot_dict(snapshot: USDataSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def validate_bar_coverage(frame: pd.DataFrame, start: str, end: str, *, max_missing_ratio: float = 0.01) -> None:
    expected = _expected_weekdays(start, end)
    if not expected:
        return
    actual = {str(pd.Timestamp(value).date()) for value in frame.get("Date", [])}
    missing = [day for day in expected if day not in actual]
    if len(missing) / len(expected) > max_missing_ratio:
        raise RuntimeError(
            f"missing {len(missing)} of {len(expected)} expected weekday bars "
            f"({len(missing) / len(expected):.2%}); first missing dates: {missing[:5]}"
        )


def _create_provider(name: str, cache_dir: str | Path) -> USMarketDataProvider:
    if name in {"massive", "polygon"}:
        return MassiveUSDataProvider(cache_dir=cache_dir)
    if name == "yfinance":
        return YFinanceUSDataProvider()
    raise ValueError(f"unsupported US data provider: {name}")


def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
    raw = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _yfinance_interval(multiplier: int, timespan: str) -> str:
    if timespan == "day" and multiplier == 1:
        return "1d"
    if timespan == "minute":
        return f"{multiplier}m"
    if timespan == "hour":
        return f"{multiplier}h"
    return "1d"


def _expected_weekdays(start: str, end: str) -> list[str]:
    current = datetime.strptime(start, "%Y-%m-%d").date()
    stop = datetime.strptime(end, "%Y-%m-%d").date()
    days: list[str] = []
    while current < stop:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days
