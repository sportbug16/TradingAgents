"""Model-neutral US data packets for fast LLM screening."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.backtesting.portfolio import CorrelationRiskModel
from tradingagents.default_config import DEFAULT_CONFIG

from .data import create_us_data_provider, snapshot_dict
from .market import DEFAULT_US_BENCHMARK, USInstrumentRegistry


@dataclass(frozen=True)
class USDataPacket:
    packet_version: str
    config_hash: str
    ticker: str
    decision_date: str
    horizon_sessions: int
    instrument: dict[str, Any]
    data_snapshots: dict[str, dict[str, Any]]
    price_features: dict[str, Any]
    benchmark_features: dict[str, Any]
    correlation_features: dict[str, Any]
    data_quality: dict[str, Any]


def generate_packets(
    *,
    tickers: list[str] | str,
    dates: list[str],
    output_dir: str | Path,
    price_start: str,
    price_end: str,
    horizon_sessions: int = 20,
    data_provider_name: str = "massive",
    fallback_provider: str = "yfinance",
    adjusted_prices: bool = True,
    benchmark: str = DEFAULT_US_BENCHMARK,
) -> dict[str, Any]:
    registry = USInstrumentRegistry()
    symbols = registry.default_symbols() if tickers == "mega_caps" else registry.validate_many(list(tickers))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_config = {
        "tickers": symbols,
        "dates": dates,
        "price_start": price_start,
        "price_end": price_end,
        "horizon_sessions": horizon_sessions,
        "data_provider": data_provider_name,
        "fallback_provider": fallback_provider,
        "adjusted_prices": adjusted_prices,
        "benchmark": benchmark,
    }
    config_hash = _hash(base_config)
    expected_paths = [
        output_path / f"{ticker}_{decision_date}_{config_hash}.json"
        for ticker in symbols
        for decision_date in dates
    ]
    if expected_paths and all(path.exists() for path in expected_paths):
        packets = [json.loads(path.read_text(encoding="utf-8")) for path in expected_paths]
        manifest = _manifest(config_hash, [str(path) for path in expected_paths], base_config)
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"manifest": manifest, "packets": packets}

    config = DEFAULT_CONFIG.copy()
    config["data_cache_dir"] = str(Path(output_dir) / "market_cache")
    config["us"] = {
        **DEFAULT_CONFIG.get("us", {}),
        "data_provider": data_provider_name,
        "fallback_provider": fallback_provider,
        "adjusted_prices": adjusted_prices,
    }
    provider = create_us_data_provider(config)

    prices_by_ticker = {}
    snapshots = {}
    for ticker in symbols:
        result = provider.get_ohlcv(ticker, price_start, price_end, adjusted=adjusted_prices)
        prices_by_ticker[ticker] = result.data
        snapshots[ticker] = snapshot_dict(result.snapshot)

    spy = provider.get_benchmark_ohlcv(benchmark, price_start, price_end, adjusted=adjusted_prices)
    qqq = provider.get_benchmark_ohlcv("QQQ", price_start, price_end, adjusted=adjusted_prices)
    correlation = CorrelationRiskModel(lookback_sessions=120).matrix(prices_by_ticker)
    packet_paths = []
    packets = []

    for ticker in symbols:
        instrument = asdict(registry.resolve(ticker))
        for decision_date in dates:
            packet = USDataPacket(
                packet_version="1",
                config_hash=config_hash,
                ticker=ticker,
                decision_date=decision_date,
                horizon_sessions=horizon_sessions,
                instrument=instrument,
                data_snapshots={
                    ticker: snapshots[ticker],
                    benchmark: snapshot_dict(spy.snapshot),
                    "QQQ": snapshot_dict(qqq.snapshot),
                },
                price_features=_price_features(prices_by_ticker[ticker], decision_date),
                benchmark_features={
                    benchmark: _benchmark_features(spy.data, decision_date),
                    "QQQ": _benchmark_features(qqq.data, decision_date),
                },
                correlation_features=_correlation_features(ticker, correlation),
                data_quality={
                    "fallback_unofficial": snapshots[ticker].get("fallback_unofficial", False),
                    "adjusted_prices": adjusted_prices,
                    "provider": snapshots[ticker].get("provider"),
                },
            )
            packet_path = output_path / f"{ticker}_{decision_date}_{config_hash}.json"
            payload = asdict(packet)
            packet_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            packet_paths.append(str(packet_path))
            packets.append(payload)

    manifest = _manifest(config_hash, packet_paths, base_config)
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "packets": packets}


def load_packets(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    if root.is_file():
        data = json.loads(root.read_text(encoding="utf-8"))
        if "packets" in data:
            return list(data["packets"])
        if "packet_paths" in data:
            return [json.loads(Path(p).read_text(encoding="utf-8")) for p in data["packet_paths"]]
        return [data]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in manifest["packet_paths"]]


def _price_features(prices: pd.DataFrame, decision_date: str) -> dict[str, Any]:
    frame = _through(prices, decision_date)
    if frame.empty:
        return {}
    close = frame["Close"].astype(float)
    last = frame.iloc[-1]
    ret_5 = _pct(close, 5)
    ret_20 = _pct(close, 20)
    ret_60 = _pct(close, 60)
    sma_20 = _mean(close, 20)
    sma_60 = _mean(close, 60)
    return {
        "last_date": str(last["Date"]),
        "close": float(last["Close"]),
        "volume": float(last.get("Volume", 0) or 0),
        "return_5d": ret_5,
        "return_20d": ret_20,
        "return_60d": ret_60,
        "sma_20": sma_20,
        "sma_60": sma_60,
        "volatility_20d": close.pct_change().tail(20).std() * (252 ** 0.5) if len(close) > 20 else None,
        "above_sma_20": bool(close.iloc[-1] > sma_20) if sma_20 is not None else None,
        "above_sma_60": bool(close.iloc[-1] > sma_60) if sma_60 is not None else None,
    }


def _benchmark_features(prices: pd.DataFrame, decision_date: str) -> dict[str, Any]:
    frame = _through(prices, decision_date)
    if frame.empty:
        return {}
    close = frame["Close"].astype(float)
    return {
        "close": float(close.iloc[-1]),
        "return_20d": _pct(close, 20),
        "return_60d": _pct(close, 60),
    }


def _correlation_features(ticker: str, corr: pd.DataFrame) -> dict[str, Any]:
    if corr.empty or ticker not in corr:
        return {"top_correlations": [], "cluster": "unassigned"}
    series = corr[ticker].drop(labels=[ticker], errors="ignore").sort_values(ascending=False)
    top = [{"ticker": key, "correlation": float(value)} for key, value in series.head(5).items()]
    cluster = "high_beta_tech" if any(item["correlation"] >= 0.75 for item in top) else "single_name"
    return {"top_correlations": top, "cluster": cluster}


def _through(prices: pd.DataFrame, decision_date: str) -> pd.DataFrame:
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame[frame["Date"] <= pd.Timestamp(decision_date)].sort_values("Date").reset_index(drop=True)


def _pct(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def _mean(close: pd.Series, periods: int) -> float | None:
    if len(close) < periods:
        return None
    return float(close.tail(periods).mean())


def _manifest(config_hash: str, packet_paths: list[str], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_hash": config_hash,
        "packet_count": len(packet_paths),
        "packet_paths": packet_paths,
        "config": config,
    }


def _hash(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
