"""Indian market data provider interfaces and adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol

import pandas as pd
import requests

from tradingagents.default_config import DEFAULT_CONFIG

from .market import IndianInstrument, IndianInstrumentRegistry


class IndianDataProviderUnavailable(RuntimeError):
    """Raised when a preferred Indian data provider cannot serve a request."""


@dataclass(frozen=True)
class DataSnapshot:
    provider: str
    endpoint: str
    params: dict[str, Any]
    adjusted: bool | None
    fetched_at: str
    cache_key: str | None = None
    fallback_unofficial: bool = False
    version: str | None = None


@dataclass(frozen=True)
class IndianDataFrame:
    data: pd.DataFrame
    snapshot: DataSnapshot


class IndianMarketDataProvider(Protocol):
    name: str

    def get_ohlcv(self, instrument: IndianInstrument, start_date: str, end_date: str) -> IndianDataFrame:
        ...

    def get_intraday_ohlcv(
        self,
        instrument: IndianInstrument,
        start_datetime: str,
        end_datetime: str,
        interval_minutes: int = 5,
    ) -> IndianDataFrame:
        ...

    def get_benchmark_ohlcv(self, benchmark: str, start_date: str, end_date: str) -> IndianDataFrame:
        ...


class DhanHQProvider:
    """DhanHQ v2 historical candle adapter for NSE/BSE cash equities."""

    name = "dhanhq"
    base_url = "https://api.dhan.co/v2"

    def __init__(
        self,
        access_token: str | None = None,
        timeout: float = 30.0,
        cache_dir: str | Path | None = None,
        session: requests.Session | None = None,
        max_retries: int | None = None,
        retry_base_seconds: float | None = None,
    ):
        self.access_token = access_token or os.getenv("DHAN_ACCESS_TOKEN")
        self.timeout = timeout
        if not self.access_token:
            raise IndianDataProviderUnavailable("DHAN_ACCESS_TOKEN is required for DhanHQProvider")
        self.cache_dir = Path(cache_dir or DEFAULT_CONFIG["data_cache_dir"]) / "dhanhq"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("DHAN_MAX_RETRIES", "4"))
        self.retry_base_seconds = (
            retry_base_seconds
            if retry_base_seconds is not None
            else float(os.getenv("DHAN_RETRY_BASE_SECONDS", "2.0"))
        )

    def get_ohlcv(self, instrument: IndianInstrument, start_date: str, end_date: str) -> IndianDataFrame:
        payload = self._base_payload(instrument) | {
            "fromDate": start_date,
            "toDate": end_date,
        }
        return self._post_chart("/charts/historical", payload, adjusted=False)

    def get_intraday_ohlcv(
        self,
        instrument: IndianInstrument,
        start_datetime: str,
        end_datetime: str,
        interval_minutes: int = 5,
    ) -> IndianDataFrame:
        if interval_minutes not in {1, 5, 15, 25, 60}:
            raise ValueError("Dhan intraday interval must be one of 1, 5, 15, 25, 60")

        frames: list[pd.DataFrame] = []
        snapshots: list[DataSnapshot] = []
        for chunk_start, chunk_end in dhan_intraday_chunks(start_datetime, end_datetime):
            payload = self._base_payload(instrument) | {
                "interval": str(interval_minutes),
                "fromDate": chunk_start,
                "toDate": chunk_end,
            }
            result = self._post_chart("/charts/intraday", payload, adjusted=False)
            frames.append(result.data)
            snapshots.append(result.snapshot)

        data = _normalize_ohlcv(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        return IndianDataFrame(
            data=data.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True),
            snapshot=DataSnapshot(
                provider=self.name,
                endpoint="/charts/intraday",
                params={
                    "securityId": instrument.dhan_security_id,
                    "exchangeSegment": instrument.exchange.dhan_segment,
                    "interval": str(interval_minutes),
                    "fromDate": start_datetime,
                    "toDate": end_datetime,
                    "chunks": len(snapshots),
                },
                adjusted=False,
                fetched_at=_now(),
                cache_key=",".join(s.cache_key or "" for s in snapshots),
                fallback_unofficial=False,
                version="v2",
            ),
        )

    def get_benchmark_ohlcv(self, benchmark: str, start_date: str, end_date: str) -> IndianDataFrame:
        payload = _benchmark_payload(benchmark) | {
            "fromDate": start_date,
            "toDate": end_date,
        }
        return self._post_chart("/charts/historical", payload, adjusted=False)

    def _base_payload(self, instrument: IndianInstrument) -> dict[str, Any]:
        if not instrument.dhan_security_id:
            raise IndianDataProviderUnavailable(
                f"missing Dhan securityId for {instrument.qualified_symbol}"
            )
        return {
            "securityId": instrument.dhan_security_id,
            "exchangeSegment": instrument.exchange.dhan_segment,
            "instrument": "EQUITY",
            "expiryCode": 0,
            "oi": False,
        }

    def _post_chart(self, path: str, payload: dict[str, Any], adjusted: bool | None) -> IndianDataFrame:
        cache_key = _cache_key(path, payload)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            raw = self._post_with_retries(path, payload)
            cache_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")

        return IndianDataFrame(
            data=_dhan_chart_to_frame(raw),
            snapshot=DataSnapshot(
                provider=self.name,
                endpoint=path,
                params=_public_params(payload),
                adjusted=adjusted,
                fetched_at=_now(),
                cache_key=cache_key,
                fallback_unofficial=False,
                version="v2",
            ),
        )

    def _post_with_retries(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "access-token": self.access_token,
        }
        last_error: requests.RequestException | None = None
        for attempt in range(self.max_retries + 1):
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            try:
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if getattr(response, "status_code", None) != 429 or attempt >= self.max_retries:
                    raise IndianDataProviderUnavailable(str(exc)) from exc
                retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
                delay = float(retry_after) if retry_after else min(self.retry_base_seconds * (2**attempt), 30.0)
                time.sleep(delay)
        raise IndianDataProviderUnavailable(str(last_error))


def create_indian_provider(
    name: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    allow_fallback: bool = True,
) -> IndianMarketDataProvider:
    config = config or DEFAULT_CONFIG
    india_config = config.get("india", {}) if isinstance(config, dict) else {}
    provider = (name or india_config.get("data_provider") or os.getenv("TRADINGAGENTS_INDIA_DATA_PROVIDER") or "dhan").lower()
    cache_dir = config.get("data_cache_dir", DEFAULT_CONFIG["data_cache_dir"]) if isinstance(config, dict) else DEFAULT_CONFIG["data_cache_dir"]

    if provider == "yfinance":
        raise ValueError("yfinance is not supported for India data; use dhan")
    if provider in {"dhan", "dhanhq"}:
        return DhanHQProvider(cache_dir=cache_dir)
    raise ValueError(f"unsupported Indian data provider: {provider}")


def dhan_intraday_chunks(start_datetime: str, end_datetime: str, max_days: int = 90) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_datetime)
    end = pd.Timestamp(end_datetime)
    if end < start:
        raise ValueError("end_datetime must be on or after start_datetime")
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        chunks.append((_format_timestamp(cursor, start_datetime), _format_timestamp(chunk_end, end_datetime)))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def snapshot_dict(snapshot: DataSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def _benchmark_payload(benchmark: str) -> dict[str, Any]:
    normalized = benchmark.strip().upper().replace(" ", "")
    aliases = {
        "^NSEI": ("13", "NIFTY"),
        "NIFTY": ("13", "NIFTY"),
        "NIFTY50": ("13", "NIFTY"),
        "NIFTY-50": ("13", "NIFTY"),
        "^NSEBANK": ("25", "BANKNIFTY"),
        "NIFTYBANK": ("25", "BANKNIFTY"),
        "BANKNIFTY": ("25", "BANKNIFTY"),
    }
    if normalized not in aliases:
        raise IndianDataProviderUnavailable(f"missing Dhan index mapping for benchmark {benchmark}")
    security_id, _symbol = aliases[normalized]
    return {
        "securityId": security_id,
        "exchangeSegment": "IDX_I",
        "instrument": "INDEX",
        "expiryCode": 0,
        "oi": False,
    }


def _normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    frame = data.reset_index(drop=False)
    if "Datetime" in frame.columns and "Date" not in frame.columns:
        frame = frame.rename(columns={"Datetime": "Date"})
    if "index" in frame.columns and "Date" not in frame.columns:
        frame = frame.rename(columns={"index": "Date"})
    columns = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
    frame = frame[columns].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["Date", "Close"]).reset_index(drop=True)


def _dhan_chart_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    timestamps = payload.get("timestamp", [])
    for i, ts in enumerate(timestamps):
        rows.append(
            {
                "Date": datetime.fromtimestamp(ts),
                "Open": _at(payload, "open", i),
                "High": _at(payload, "high", i),
                "Low": _at(payload, "low", i),
                "Close": _at(payload, "close", i),
                "Volume": _at(payload, "volume", i),
            }
        )
    return _normalize_ohlcv(pd.DataFrame(rows))


def _at(payload: dict[str, Any], key: str, index: int):
    values = payload.get(key, [])
    return values[index] if index < len(values) else None


def _cache_key(endpoint: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _public_params(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "access-token"}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _format_timestamp(value: pd.Timestamp, original: str) -> str:
    if " " in original or "T" in original:
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value.strftime("%Y-%m-%d")


def get_indian_ohlcv_csv(
    ticker: str,
    start_date: str,
    end_date: str,
    provider_name: str | None = None,
) -> str:
    registry = IndianInstrumentRegistry()
    instrument = registry.resolve(ticker)
    provider = create_indian_provider(provider_name)
    result = provider.get_ohlcv(instrument, start_date, end_date)
    if result.data.empty:
        return f"No Indian OHLCV data found for {ticker} between {start_date} and {end_date}"
    return result.data.to_csv(index=False)


def get_indian_ohlcv_csv_dhan(ticker: str, start_date: str, end_date: str) -> str:
    return get_indian_ohlcv_csv(ticker, start_date, end_date, provider_name="dhan")
