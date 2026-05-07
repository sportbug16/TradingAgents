"""Run a small Indian equities short-horizon pilot backtest.

This script intentionally uses a deterministic technical signal instead of
LLM calls so the market-data and simulation path can be exercised cheaply and
reproducibly. The TradingAgents LLM graph can later feed the same simulator by
writing ``Signal`` objects from ``GraphBatchRunner``.
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
    random_entry_baseline,
    simple_momentum_baseline,
)
from tradingagents.backtesting.rules import Signal
from tradingagents.backtesting.simulator import ShortHorizonSimulator
from tradingagents.india.data import create_indian_provider, snapshot_dict
from tradingagents.india.market import IndianInstrumentRegistry, IndianTradingCalendar
from tradingagents.india.packets import generate_packets


DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]


def main() -> None:
    args = _parse_args()
    registry = IndianInstrumentRegistry()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    provider = create_indian_provider(args.data_provider, allow_fallback=args.allow_fallback)
    prices_by_ticker, data_snapshots = _download_many(provider, registry, tickers, args.start, args.end)
    benchmark_result = provider.get_benchmark_ohlcv(args.benchmark, args.start, args.end)
    horizons = _horizons(args)
    calendar = IndianTradingCalendar.from_csv(args.holidays_csv) if args.holidays_csv else IndianTradingCalendar()

    packet_result = None
    if args.packet_output:
        packet_result = generate_packets(
            tickers=tickers,
            dates=args.packet_dates.split(",") if args.packet_dates else _signal_dates(prices_by_ticker, args.lookback, max(horizons)),
            output_dir=args.packet_output,
            price_start=args.start,
            price_end=args.end,
            horizon_sessions=max(horizons),
            data_provider_name=args.data_provider,
            allow_fallback=args.allow_fallback,
            benchmark=args.benchmark,
        )

    signals = _technical_signals(prices_by_ticker, lookback=args.lookback, max_horizon=max(horizons))
    horizon_results = {}
    for horizon in horizons:
        simulator = ShortHorizonSimulator(
            holding_sessions=horizon,
            calendar=calendar,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
        )
        result = simulator.simulate(prices_by_ticker, signals)
        horizon_results[str(horizon)] = {
            "strategy_metrics": asdict(result.metrics),
            "trade_count": len(result.trades),
            "sample_trades": [asdict(t) for t in result.trades[:20]],
        }

    summary = {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
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
            "simple_momentum": asdict(simple_momentum_baseline(prices_by_ticker)),
            "random_entry": {
                str(horizon): asdict(random_entry_baseline(prices_by_ticker, hold_sessions=horizon))
                for horizon in horizons
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--holding-sessions", type=int, choices=[1, 3, 5], default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=12.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--data-provider", default="dhan")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--holidays-csv", default=None)
    parser.add_argument("--packet-output", default=None)
    parser.add_argument("--packet-dates", default=None)
    parser.add_argument("--output", default="reports/india_backtest_pilot.json")
    return parser.parse_args()


def _download_many(provider, registry: IndianInstrumentRegistry, tickers: list[str], start: str, end: str):
    prices = {}
    snapshots = {}
    for ticker in tickers:
        instrument = registry.resolve(ticker)
        result = provider.get_ohlcv(instrument, start, end)
        if result.data.empty:
            raise RuntimeError(f"no data downloaded for {ticker}")
        prices[instrument.data_symbol] = result.data
        snapshots[instrument.data_symbol] = snapshot_dict(result.snapshot)
    return prices, snapshots


def _technical_signals(prices_by_ticker: dict[str, pd.DataFrame], lookback: int, max_horizon: int = 5) -> list[Signal]:
    signals: list[Signal] = []
    for ticker, prices in prices_by_ticker.items():
        frame = prices.copy().sort_values("Date").reset_index(drop=True)
        frame["sma"] = frame["Close"].rolling(lookback).mean()
        frame["momentum"] = frame["Close"].pct_change(lookback)
        for idx in range(lookback, len(frame) - max_horizon):
            row = frame.iloc[idx]
            close = float(row["Close"])
            sma = float(row["sma"])
            momentum = float(row["momentum"])
            if pd.isna(sma) or pd.isna(momentum):
                continue
            if close > sma and momentum > 0.03:
                rating = "Buy"
            elif close > sma and momentum > 0:
                rating = "Overweight"
            else:
                rating = "Hold"
            signals.append(
                Signal(
                    ticker=ticker,
                    decision_date=str(pd.Timestamp(row["Date"]).date()),
                    rating=rating,
                    final_decision=f"Deterministic pilot signal: close={close:.2f}, sma={sma:.2f}, momentum={momentum:.2%}",
                )
            )
    return signals


def _signal_dates(prices_by_ticker: dict[str, pd.DataFrame], lookback: int, max_horizon: int) -> list[str]:
    dates = set()
    for prices in prices_by_ticker.values():
        frame = prices.copy().sort_values("Date").reset_index(drop=True)
        for idx in range(lookback, len(frame) - max_horizon):
            dates.add(str(pd.Timestamp(frame.iloc[idx]["Date"]).date()))
    return sorted(dates)


def _horizons(args: argparse.Namespace) -> list[int]:
    if args.holding_sessions:
        return [args.holding_sessions]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    invalid = [h for h in horizons if h not in {1, 3, 5}]
    if invalid:
        raise ValueError(f"invalid India horizons: {invalid}; expected 1, 3, and/or 5")
    return horizons


if __name__ == "__main__":
    main()
