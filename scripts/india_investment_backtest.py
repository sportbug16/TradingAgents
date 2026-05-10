"""Run an Indian equities investment/swing-position study.

This is the primary India research harness. It uses provider-backed daily
OHLCV, deterministic candidate signals, and multi-session investment horizons.
LLM ratings can feed the same simulator later; live market-hours code should
only monitor precomputed decisions and risk.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.backtesting.baselines import (
    buy_and_hold_baseline,
    equal_weight_baseline,
    low_volatility_baseline,
    random_entry_baseline,
    simple_momentum_baseline,
)
from tradingagents.backtesting.investment import InvestmentHorizonSimulator
from tradingagents.backtesting.rules import Signal
from tradingagents.india.data import create_indian_provider, snapshot_dict
from tradingagents.india.market import IndianInstrumentRegistry, IndianTradingCalendar
from tradingagents.india.packets import generate_packets


DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]


def main() -> None:
    args = _parse_args()
    registry = IndianInstrumentRegistry()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    provider = create_indian_provider(args.data_provider, allow_fallback=args.allow_fallback)
    prices_by_ticker, data_snapshots, instrument_meta = _download_many(provider, registry, tickers, args.start, args.end)
    benchmark_result = provider.get_benchmark_ohlcv(args.benchmark, args.start, args.end)
    horizons = _horizons(args.horizons)
    calendar = IndianTradingCalendar.from_csv(args.holidays_csv) if args.holidays_csv else IndianTradingCalendar()
    signals = _investment_signals(prices_by_ticker, lookback=args.lookback, max_horizon=max(horizons))

    packet_result = None
    if args.packet_output:
        packet_result = generate_packets(
            tickers=tickers,
            dates=args.packet_dates.split(",") if args.packet_dates else _signal_dates(signals),
            output_dir=args.packet_output,
            price_start=args.start,
            price_end=args.end,
            horizon_sessions=max(horizons),
            data_provider_name=args.data_provider,
            allow_fallback=args.allow_fallback,
            benchmark=args.benchmark,
        )

    horizon_results = {}
    for horizon in horizons:
        result = InvestmentHorizonSimulator(
            holding_sessions=horizon,
            calendar=calendar,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
        ).simulate(prices_by_ticker, signals)
        horizon_results[str(horizon)] = {
            "strategy_metrics": asdict(result.metrics),
            "trade_count": len(result.trades),
            "rating_bucket_forward_returns": _rating_bucket_returns(result.trades),
            "sector_forward_returns": _sector_returns(result.trades, instrument_meta),
            "sample_trades": [asdict(t) for t in result.trades[:20]],
        }

    summary = {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "study_type": "india_investment_swing",
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "horizons": horizons,
        "data_provider": getattr(provider, "name", args.data_provider),
        "allow_fallback": args.allow_fallback,
        "data_snapshots": data_snapshots,
        "benchmark_snapshot": snapshot_dict(benchmark_result.snapshot),
        "packet_manifest": packet_result["manifest"] if packet_result else None,
        "signal_count": len(signals),
        "horizon_results": horizon_results,
        "baselines": {
            "benchmark_buy_hold": asdict(buy_and_hold_baseline(args.benchmark, benchmark_result.data)),
            "equal_weight_buy_hold": asdict(equal_weight_baseline(prices_by_ticker)),
            "simple_momentum": {
                str(h): asdict(simple_momentum_baseline(prices_by_ticker, lookback_sessions=args.lookback, hold_sessions=h))
                for h in horizons
            },
            "low_volatility": {
                str(h): asdict(low_volatility_baseline(prices_by_ticker, lookback_sessions=max(args.lookback, 20), hold_sessions=h))
                for h in horizons
            },
            "random_entry": {
                str(h): asdict(random_entry_baseline(prices_by_ticker, hold_sessions=h))
                for h in horizons
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizons", default="5,20,60,126")
    parser.add_argument("--transaction-cost-bps", type=float, default=12.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--take-profit-pct", type=float, default=None)
    parser.add_argument("--data-provider", default="dhan")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--holidays-csv", default=None)
    parser.add_argument("--packet-output", default=None)
    parser.add_argument("--packet-dates", default=None)
    parser.add_argument("--output", default="reports/india_investment_backtest.json")
    return parser.parse_args()


def _download_many(provider, registry: IndianInstrumentRegistry, tickers: list[str], start: str, end: str):
    prices = {}
    snapshots = {}
    instruments = {}
    for ticker in tickers:
        instrument = registry.resolve(ticker)
        result = provider.get_ohlcv(instrument, start, end)
        if result.data.empty:
            raise RuntimeError(f"no data downloaded for {ticker}")
        prices[instrument.data_symbol] = result.data
        snapshots[instrument.data_symbol] = snapshot_dict(result.snapshot)
        instruments[instrument.data_symbol] = asdict(instrument)
    return prices, snapshots, instruments


def _investment_signals(prices_by_ticker: dict[str, pd.DataFrame], lookback: int, max_horizon: int) -> list[Signal]:
    signals = []
    for ticker, prices in prices_by_ticker.items():
        frame = prices.copy().sort_values("Date").reset_index(drop=True)
        frame["sma"] = frame["Close"].rolling(lookback).mean()
        frame["momentum"] = frame["Close"].pct_change(lookback)
        frame["volatility"] = frame["Close"].pct_change().rolling(20).std()
        for idx in range(lookback, len(frame) - max_horizon):
            row = frame.iloc[idx]
            if pd.isna(row["sma"]) or pd.isna(row["momentum"]) or pd.isna(row["volatility"]):
                continue
            close = float(row["Close"])
            sma = float(row["sma"])
            momentum = float(row["momentum"])
            volatility = float(row["volatility"])
            if close > sma and momentum > 0.08 and volatility < 0.04:
                rating = "Buy"
            elif close > sma and momentum > 0.02:
                rating = "Overweight"
            else:
                rating = "Hold"
            signals.append(
                Signal(
                    ticker=ticker,
                    decision_date=str(pd.Timestamp(row["Date"]).date()),
                    rating=rating,
                    final_decision=(
                        "Deterministic investment signal: "
                        f"close={close:.2f}, sma={sma:.2f}, momentum={momentum:.2%}, vol20={volatility:.2%}"
                    ),
                )
            )
    return signals


def _rating_bucket_returns(trades) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for trade in trades:
        buckets.setdefault(trade.rating, []).append(trade.net_return)
    return {
        rating: {"count": len(values), "average_net_return": sum(values) / len(values)}
        for rating, values in sorted(buckets.items())
        if values
    }


def _sector_returns(trades, instrument_meta: dict[str, dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for trade in trades:
        sector = instrument_meta.get(trade.ticker, {}).get("sector") or "Unknown"
        buckets.setdefault(sector, []).append(trade.net_return)
    return {
        sector: {"count": len(values), "average_net_return": sum(values) / len(values)}
        for sector, values in sorted(buckets.items())
        if values
    }


def _signal_dates(signals: list[Signal]) -> list[str]:
    return sorted({signal.decision_date for signal in signals})


def _horizons(value: str) -> list[int]:
    horizons = [int(h.strip()) for h in value.split(",") if h.strip()]
    invalid = [h for h in horizons if h < 1 or h > 252]
    if invalid:
        raise ValueError(f"invalid India investment horizons: {invalid}; expected 1-252 sessions")
    return horizons


if __name__ == "__main__":
    main()
