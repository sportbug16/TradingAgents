"""Backtest Indian LLM screening decisions as long-only investment signals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.backtesting.baselines import (
    buy_and_hold_baseline,
    equal_weight_baseline,
    low_volatility_baseline,
    random_entry_baseline,
    simple_momentum_baseline,
)
from tradingagents.backtesting.investment import InvestmentHorizonSimulator
from tradingagents.india.data import create_indian_provider, snapshot_dict
from tradingagents.india.llm_backtest import (
    DEFAULT_RESEARCH_PROVIDERS,
    EXCLUDED_RESEARCH_PROVIDERS,
    build_signal_sets,
    extract_decisions,
    provider_labels,
)
from tradingagents.india.market import IndianInstrumentRegistry, IndianTradingCalendar


def main() -> None:
    load_dotenv()
    args = _parse_args()
    screen_results = json.loads(Path(args.screen_results).read_text(encoding="utf-8"))
    providers = provider_labels(args.providers)
    horizons = _horizons(args.horizons)
    decisions = extract_decisions(screen_results, providers=providers, horizons=horizons)
    tickers = _tickers(args, decisions)
    if not tickers:
        raise RuntimeError("no tickers found in screen results or --tickers")

    start = args.start or _default_start(decisions)
    end = args.end or _default_end(decisions, max(horizons))
    registry = IndianInstrumentRegistry()
    provider = create_indian_provider(args.data_provider, allow_fallback=args.allow_fallback)
    prices_by_ticker, data_snapshots = _download_many(provider, registry, tickers, start, end)
    benchmark_result = provider.get_benchmark_ohlcv(args.benchmark, start, end)
    calendar = IndianTradingCalendar.from_csv(args.holidays_csv) if args.holidays_csv else IndianTradingCalendar()

    horizon_results = {}
    for horizon in horizons:
        signal_sets = build_signal_sets(decisions, horizon)
        horizon_results[str(horizon)] = {
            name: _simulate_signal_set(
                name=name,
                signals=signals,
                prices_by_ticker=prices_by_ticker,
                horizon=horizon,
                calendar=calendar,
                args=args,
            )
            for name, signals in sorted(signal_sets.items())
        }

    summary = {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "study_type": "india_llm_backtest",
        "screen_results": args.screen_results,
        "provider_selection": {
            "included": providers,
            "default_research_providers": list(DEFAULT_RESEARCH_PROVIDERS),
            "excluded": sorted(EXCLUDED_RESEARCH_PROVIDERS),
        },
        "tickers": tickers,
        "start": start,
        "end": end,
        "horizons": horizons,
        "data_provider": getattr(provider, "name", args.data_provider),
        "allow_fallback": args.allow_fallback,
        "data_snapshots": data_snapshots,
        "benchmark_snapshot": snapshot_dict(benchmark_result.snapshot),
        "decision_count": len(decisions),
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
            "random_entry": {str(h): asdict(random_entry_baseline(prices_by_ticker, hold_sessions=h)) for h in horizons},
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-results", required=True)
    parser.add_argument("--providers", default=",".join(DEFAULT_RESEARCH_PROVIDERS))
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizons", default="5,20,60,126")
    parser.add_argument("--transaction-cost-bps", type=float, default=12.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--take-profit-pct", type=float, default=None)
    parser.add_argument("--data-provider", default="dhan")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--holidays-csv", default=None)
    parser.add_argument("--output", default="reports/india_llm_backtest/default.json")
    return parser.parse_args()


def _simulate_signal_set(name, signals, prices_by_ticker, horizon, calendar, args) -> dict:
    simulator = InvestmentHorizonSimulator(
        holding_sessions=horizon,
        calendar=calendar,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
    )
    usable_signals = [s for s in signals if s.ticker in prices_by_ticker]
    result = simulator.simulate(prices_by_ticker, usable_signals)
    return {
        "signal_count": len(signals),
        "usable_signal_count": len(usable_signals),
        "trade_count": len(result.trades),
        "strategy_metrics": asdict(result.metrics),
        "rating_bucket_forward_returns": _rating_bucket_returns(result.trades),
        "sample_trades": [asdict(t) for t in result.trades[:20]],
        "note": "no trades usually means ratings were non-entry or price history did not cover exits"
        if not result.trades
        else None,
    }


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


def _rating_bucket_returns(trades) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for trade in trades:
        buckets.setdefault(trade.rating, []).append(trade.net_return)
    return {
        rating: {"count": len(values), "average_net_return": sum(values) / len(values)}
        for rating, values in sorted(buckets.items())
        if values
    }


def _tickers(args: argparse.Namespace, decisions) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    return sorted({d.ticker for d in decisions})


def _horizons(value: str) -> list[int]:
    horizons = [int(h.strip()) for h in value.split(",") if h.strip()]
    invalid = [h for h in horizons if h < 1 or h > 252]
    if invalid:
        raise ValueError(f"invalid India LLM backtest horizons: {invalid}; expected 1-252 sessions")
    return horizons


def _default_start(decisions) -> str:
    dates = [datetime.strptime(d.decision_date, "%Y-%m-%d").date() for d in decisions]
    return (min(dates) - timedelta(days=420)).isoformat() if dates else "2025-01-01"


def _default_end(decisions, max_horizon: int) -> str:
    dates = [datetime.strptime(d.decision_date, "%Y-%m-%d").date() for d in decisions]
    return (max(dates) + timedelta(days=max_horizon * 2 + 14)).isoformat() if dates else datetime.utcnow().date().isoformat()


if __name__ == "__main__":
    main()
