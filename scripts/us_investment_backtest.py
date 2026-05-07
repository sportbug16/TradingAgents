"""Run a US Nasdaq/S&P 500 investment-horizon pilot backtest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.backtesting.baselines import (
    buy_and_hold_baseline,
    equal_weight_baseline,
    random_entry_baseline,
    simple_momentum_baseline,
)
from tradingagents.backtesting.investment import InvestmentHorizonSimulator
from tradingagents.backtesting.portfolio import PortfolioBacktestSimulator
from tradingagents.backtesting.rules import Signal
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.us.data import create_us_data_provider, snapshot_dict
from tradingagents.us.market import DEFAULT_US_BENCHMARK, DEFAULT_US_TICKERS, USInstrumentRegistry


def main() -> None:
    load_dotenv()
    args = _parse_args()
    registry = USInstrumentRegistry()
    tickers = registry.validate_many([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    data_provider = create_us_data_provider(_data_config(args))
    prices_by_ticker, data_snapshots = _download_many(
        data_provider,
        tickers,
        args.start,
        args.end,
        adjusted=args.adjusted_prices,
    )
    benchmark, benchmark_snapshot = _download_frame(data_provider, args.benchmark, args.start, args.end, adjusted=args.adjusted_prices, benchmark=True)
    qqq, qqq_snapshot = _download_frame(data_provider, "QQQ", args.start, args.end, adjusted=args.adjusted_prices, benchmark=True)

    horizon_results = {}
    for horizon in [int(x.strip()) for x in args.horizons.split(",") if x.strip()]:
        signals = _investment_signals(
            prices_by_ticker,
            fast_window=args.fast_window,
            slow_window=args.slow_window,
            rebalance_sessions=horizon,
        )
        simulator = InvestmentHorizonSimulator(
            holding_sessions=horizon,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            stop_loss_pct=args.stop_loss_pct,
        )
        result = simulator.simulate(prices_by_ticker, signals)
        portfolio = PortfolioBacktestSimulator(
            holding_sessions=horizon,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            max_gross_exposure=args.max_gross_exposure,
            max_single_name_exposure=args.max_single_name_exposure,
            max_correlation_cluster_exposure=args.max_correlation_cluster_exposure,
            correlation_lookback_sessions=args.correlation_lookback_sessions,
            correlation_threshold=args.correlation_threshold,
        ).simulate(prices_by_ticker, signals)
        horizon_results[str(horizon)] = {
            "signal_count": len(signals),
            "trade_count": len(result.trades),
            "strategy_metrics": asdict(result.metrics),
            "portfolio_metrics": asdict(portfolio.metrics),
            "portfolio_equity_curve": portfolio.equity_curve,
            "skipped_signals": portfolio.skipped_signals,
            "correlation_matrix": portfolio.correlation_matrix,
            "sample_trades": [asdict(t) for t in result.trades[:15]],
        }

    summary = {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "universe": "configured Nasdaq-listed S&P 500 seed",
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "benchmark": args.benchmark,
        "data_provider": getattr(data_provider, "name", args.data_provider),
        "adjusted_prices": args.adjusted_prices,
        "data_snapshots": data_snapshots,
        "benchmark_snapshots": {
            args.benchmark: snapshot_dict(benchmark_snapshot),
            "QQQ": snapshot_dict(qqq_snapshot),
        },
        "horizons_sessions": [int(x.strip()) for x in args.horizons.split(",") if x.strip()],
        "horizon_results": horizon_results,
        "baselines": {
            "benchmark_buy_hold": asdict(buy_and_hold_baseline(args.benchmark, benchmark)),
            "qqq_buy_hold": asdict(buy_and_hold_baseline("QQQ", qqq)),
            "equal_weight_buy_hold": asdict(equal_weight_baseline(prices_by_ticker)),
            "simple_momentum": asdict(simple_momentum_baseline(prices_by_ticker, lookback_sessions=60, hold_sessions=20)),
            "random_entry_20d": asdict(random_entry_baseline(prices_by_ticker, hold_sessions=20)),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=",".join(DEFAULT_US_TICKERS))
    parser.add_argument("--benchmark", default=DEFAULT_US_BENCHMARK)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--horizons", default="5,20,60,126")
    parser.add_argument("--data-provider", default="massive")
    parser.add_argument("--fallback-provider", default="yfinance")
    parser.add_argument("--adjusted-prices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-window", type=int, default=20)
    parser.add_argument("--slow-window", type=int, default=60)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--max-gross-exposure", type=float, default=1.0)
    parser.add_argument("--max-single-name-exposure", type=float, default=0.25)
    parser.add_argument("--max-correlation-cluster-exposure", type=float, default=0.50)
    parser.add_argument("--correlation-lookback-sessions", type=int, default=120)
    parser.add_argument("--correlation-threshold", type=float, default=0.75)
    parser.add_argument("--output", default="reports/us_investment_backtest.json")
    return parser.parse_args()


def _data_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["data_cache_dir"] = str(Path(args.output).parent / "us_market_cache")
    config["us"] = {
        **DEFAULT_CONFIG.get("us", {}),
        "data_provider": args.data_provider,
        "fallback_provider": args.fallback_provider,
        "adjusted_prices": args.adjusted_prices,
    }
    return config


def _download_many(data_provider, tickers: list[str], start: str, end: str, *, adjusted: bool) -> tuple[dict[str, pd.DataFrame], dict]:
    prices = {}
    snapshots = {}
    for ticker in tickers:
        frame, snapshot = _download_frame(data_provider, ticker, start, end, adjusted=adjusted)
        prices[ticker] = frame
        snapshots[ticker] = snapshot_dict(snapshot)
    return prices, snapshots


def _download_frame(data_provider, ticker: str, start: str, end: str, *, adjusted: bool, benchmark: bool = False):
    result = (
        data_provider.get_benchmark_ohlcv(ticker, start, end, adjusted=adjusted)
        if benchmark
        else data_provider.get_ohlcv(ticker, start, end, adjusted=adjusted)
    )
    if result.data.empty:
        raise RuntimeError(f"no data downloaded for {ticker}")
    columns = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in result.data.columns]
    return result.data[columns].copy(), result.snapshot


def _investment_signals(
    prices_by_ticker: dict[str, pd.DataFrame],
    fast_window: int,
    slow_window: int,
    rebalance_sessions: int,
) -> list[Signal]:
    signals: list[Signal] = []
    for ticker, prices in prices_by_ticker.items():
        frame = prices.copy().sort_values("Date").reset_index(drop=True)
        frame["fast"] = frame["Close"].rolling(fast_window).mean()
        frame["slow"] = frame["Close"].rolling(slow_window).mean()
        frame["momentum"] = frame["Close"].pct_change(slow_window)
        start_idx = slow_window
        for idx in range(start_idx, len(frame) - rebalance_sessions - 2, rebalance_sessions):
            row = frame.iloc[idx]
            close = float(row["Close"])
            fast = float(row["fast"])
            slow = float(row["slow"])
            momentum = float(row["momentum"])
            if pd.isna(fast) or pd.isna(slow) or pd.isna(momentum):
                continue
            if close > fast > slow and momentum > 0.15:
                rating = "Buy"
            elif close > fast and close > slow and momentum > 0:
                rating = "Overweight"
            elif close < slow and momentum < -0.10:
                rating = "Sell"
            else:
                rating = "Hold"
            signals.append(
                Signal(
                    ticker=ticker,
                    decision_date=str(pd.Timestamp(row["Date"]).date()),
                    rating=rating,
                    final_decision=(
                        f"US investment pilot signal: close={close:.2f}, "
                        f"fast={fast:.2f}, slow={slow:.2f}, momentum={momentum:.2%}"
                    ),
                )
            )
    return signals


if __name__ == "__main__":
    main()
