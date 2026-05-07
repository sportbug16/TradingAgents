"""Model-neutral Indian data packets for research and fast screening."""

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

from .data import create_indian_provider, snapshot_dict
from .market import DEFAULT_INDIAN_BENCHMARK, IndianInstrumentRegistry


@dataclass(frozen=True)
class IndiaDataPacket:
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
    tickers: list[str],
    dates: list[str],
    output_dir: str | Path,
    price_start: str,
    price_end: str,
    horizon_sessions: int = 3,
    data_provider_name: str = "dhan",
    allow_fallback: bool = False,
    benchmark: str = DEFAULT_INDIAN_BENCHMARK,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = IndianInstrumentRegistry()
    instruments = [registry.resolve(ticker) for ticker in tickers]
    symbols = [instrument.data_symbol for instrument in instruments]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_config = {
        "tickers": symbols,
        "dates": dates,
        "price_start": price_start,
        "price_end": price_end,
        "horizon_sessions": horizon_sessions,
        "data_provider": data_provider_name,
        "allow_fallback": allow_fallback,
        "benchmark": benchmark,
    }
    config_hash = _hash(base_config)
    expected_paths = [
        output_path / f"{instrument.data_symbol}_{decision_date}_{config_hash}.json"
        for instrument in instruments
        for decision_date in dates
    ]
    if expected_paths and all(path.exists() for path in expected_paths):
        packets = [json.loads(path.read_text(encoding="utf-8")) for path in expected_paths]
        manifest = _manifest(config_hash, [str(path) for path in expected_paths], base_config)
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"manifest": manifest, "packets": packets}

    provider_config = (config or DEFAULT_CONFIG).copy()
    provider_config["data_cache_dir"] = str(output_path / "market_cache")
    provider = create_indian_provider(data_provider_name, config=provider_config, allow_fallback=allow_fallback)

    prices_by_ticker = {}
    snapshots = {}
    for instrument in instruments:
        result = provider.get_ohlcv(instrument, price_start, price_end)
        prices_by_ticker[instrument.data_symbol] = result.data
        snapshots[instrument.data_symbol] = snapshot_dict(result.snapshot)

    benchmark_result = provider.get_benchmark_ohlcv(benchmark, price_start, price_end)
    correlation = CorrelationRiskModel(lookback_sessions=120).matrix(prices_by_ticker)
    packets = []
    packet_paths = []

    for instrument in instruments:
        symbol = instrument.data_symbol
        for decision_date in dates:
            snapshot = snapshots[symbol]
            packet = IndiaDataPacket(
                packet_version="1",
                config_hash=config_hash,
                ticker=symbol,
                decision_date=decision_date,
                horizon_sessions=horizon_sessions,
                instrument=asdict(instrument),
                data_snapshots={
                    symbol: snapshot,
                    benchmark: snapshot_dict(benchmark_result.snapshot),
                },
                price_features=_price_features(prices_by_ticker[symbol], decision_date),
                benchmark_features={benchmark: _benchmark_features(benchmark_result.data, decision_date)},
                correlation_features=_correlation_features(symbol, correlation),
                data_quality={
                    "provider": snapshot.get("provider"),
                    "fallback_unofficial": snapshot.get("fallback_unofficial", False),
                    "benchmark_fallback_unofficial": benchmark_result.snapshot.fallback_unofficial,
                    "adjusted_prices": snapshot.get("adjusted"),
                },
            )
            packet_path = output_path / f"{symbol}_{decision_date}_{config_hash}.json"
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
    avg_volume_20 = _mean(frame["Volume"].astype(float), 20) if "Volume" in frame else None
    return {
        "last_date": str(last["Date"]),
        "close": float(last["Close"]),
        "volume": float(last.get("Volume", 0) or 0),
        "avg_volume_20d": avg_volume_20,
        "return_5d": _pct(close, 5),
        "return_20d": _pct(close, 20),
        "return_60d": _pct(close, 60),
        "sma_20": _mean(close, 20),
        "sma_60": _mean(close, 60),
        "volatility_20d": close.pct_change().tail(20).std() * (252 ** 0.5) if len(close) > 20 else None,
    }


def _benchmark_features(prices: pd.DataFrame, decision_date: str) -> dict[str, Any]:
    frame = _through(prices, decision_date)
    if frame.empty:
        return {}
    close = frame["Close"].astype(float)
    return {
        "close": float(close.iloc[-1]),
        "return_5d": _pct(close, 5),
        "return_20d": _pct(close, 20),
        "return_60d": _pct(close, 60),
    }


def _correlation_features(ticker: str, corr: pd.DataFrame) -> dict[str, Any]:
    if corr.empty or ticker not in corr:
        return {"top_correlations": [], "cluster": "unassigned"}
    series = corr[ticker].drop(labels=[ticker], errors="ignore").sort_values(ascending=False)
    top = [{"ticker": key, "correlation": float(value)} for key, value in series.head(5).items()]
    cluster = "correlated_liquid_largecap" if any(item["correlation"] >= 0.75 for item in top) else "single_name"
    return {"top_correlations": top, "cluster": cluster}


def _through(prices: pd.DataFrame, decision_date: str) -> pd.DataFrame:
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame[frame["Date"] <= pd.Timestamp(decision_date)].sort_values("Date").reset_index(drop=True)


def _pct(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def _mean(values: pd.Series, periods: int) -> float | None:
    if len(values) < periods:
        return None
    return float(values.tail(periods).mean())


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
