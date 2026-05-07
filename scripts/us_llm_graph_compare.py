"""Compare LLM-provider TradingAgentsGraph signals on US equities.

This runner is intentionally small by default because each signal invokes the
full TradingAgentsGraph decision stack. It loads API keys from `.env`, skips
providers whose keys are absent, records every LLM decision, then feeds the
parsed ratings into the investment-horizon simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.backtesting.baselines import buy_and_hold_baseline, equal_weight_baseline
from tradingagents.backtesting.investment import InvestmentHorizonSimulator
from tradingagents.backtesting.portfolio import PortfolioBacktestSimulator
from tradingagents.backtesting.rules import Signal
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.rate_limiter import RollingLLMRateLimiter
from tradingagents.us.data import create_us_data_provider, snapshot_dict
from tradingagents.us.llm_providers import PROVIDER_DEFAULTS, arg_prefix
from tradingagents.us.market import DEFAULT_US_BENCHMARK, USInstrumentRegistry


def main() -> None:
    load_dotenv()
    args = _parse_args()
    registry = USInstrumentRegistry()
    tickers = registry.validate_many([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    analysts = [a.strip().lower() for a in args.analysts.split(",") if a.strip()]

    data_config = _data_config(args)
    data_provider = create_us_data_provider(data_config)
    prices_by_ticker, data_snapshots = _download_many(
        data_provider,
        tickers,
        args.price_start,
        args.price_end,
        adjusted=args.adjusted_prices,
    )
    benchmark_frame, benchmark_snapshot = _download_frame(
        data_provider,
        args.benchmark,
        args.price_start,
        args.price_end,
        adjusted=args.adjusted_prices,
        benchmark=True,
    )
    qqq_frame, qqq_snapshot = _download_frame(
        data_provider,
        "QQQ",
        args.price_start,
        args.price_end,
        adjusted=args.adjusted_prices,
        benchmark=True,
    )

    summary = {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "tickers": tickers,
        "dates": dates,
        "analysts": analysts,
        "holding_sessions": args.holding_sessions,
        "horizons": _horizons(args),
        "data_provider": getattr(data_provider, "name", args.data_provider),
        "adjusted_prices": args.adjusted_prices,
        "data_snapshots": data_snapshots,
        "providers": {},
        "baselines": {
            "benchmark_buy_hold": asdict(buy_and_hold_baseline(args.benchmark, benchmark_frame)),
            "qqq_buy_hold": asdict(buy_and_hold_baseline("QQQ", qqq_frame)),
            "equal_weight_buy_hold": asdict(equal_weight_baseline(prices_by_ticker)),
        },
        "benchmark_snapshots": {
            args.benchmark: snapshot_dict(benchmark_snapshot),
            "QQQ": snapshot_dict(qqq_snapshot),
        },
    }

    for provider in providers:
        if provider not in PROVIDER_DEFAULTS:
            summary["providers"][provider] = {"status": "skipped", "reason": "unsupported provider"}
            continue
        defaults = PROVIDER_DEFAULTS[provider]
        if not os.getenv(defaults["key_env"]):
            summary["providers"][provider] = {
                "status": "skipped",
                "reason": f"missing {defaults['key_env']}",
            }
            continue

        provider_result = _run_provider(
            provider=provider,
            defaults=defaults,
            tickers=tickers,
            dates=dates,
            analysts=analysts,
            args=args,
            prices_by_ticker=prices_by_ticker,
        )
        summary["providers"][provider] = provider_result
        _write_summary(args.output, summary)

    _write_summary(args.output, summary)
    print(json.dumps(summary, indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", default="anthropic,google")
    parser.add_argument("--tickers", default="AAPL,NVDA,MSFT")
    parser.add_argument("--dates", default="2026-02-02,2026-03-02,2026-04-01")
    parser.add_argument("--analysts", default="market")
    parser.add_argument("--holding-sessions", type=int, default=20)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--price-start", default="2026-01-01")
    parser.add_argument("--price-end", default="2026-05-01")
    parser.add_argument("--benchmark", default=DEFAULT_US_BENCHMARK)
    parser.add_argument("--data-provider", default="massive")
    parser.add_argument("--fallback-provider", default="yfinance")
    parser.add_argument("--adjusted-prices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-gross-exposure", type=float, default=1.0)
    parser.add_argument("--max-single-name-exposure", type=float, default=0.25)
    parser.add_argument("--max-correlation-cluster-exposure", type=float, default=0.50)
    parser.add_argument("--correlation-lookback-sessions", type=int, default=120)
    parser.add_argument("--correlation-threshold", type=float, default=0.75)
    parser.add_argument("--output", default="reports/us_llm_graph_compare.json")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--anthropic-quick-model", default=None)
    parser.add_argument("--anthropic-deep-model", default=None)
    parser.add_argument("--google-quick-model", default=None)
    parser.add_argument("--google-deep-model", default=None)
    parser.add_argument("--anthropic-input-tokens-per-minute", type=int, default=30_000)
    parser.add_argument("--anthropic-requests-per-minute", type=int, default=50)
    parser.add_argument("--google-input-tokens-per-minute", type=int, default=30_000)
    parser.add_argument("--google-requests-per-minute", type=int, default=15)
    parser.add_argument("--openrouter-openai-4o-mini-input-tokens-per-minute", type=int, default=30_000)
    parser.add_argument("--openrouter-openai-4o-mini-requests-per-minute", type=int, default=60)
    parser.add_argument("--openrouter-deepseek-v4-input-tokens-per-minute", type=int, default=30_000)
    parser.add_argument("--openrouter-deepseek-v4-requests-per-minute", type=int, default=60)
    parser.add_argument("--rate-limit-safety-margin", type=float, default=0.85)
    parser.add_argument("--disable-rate-limits", action="store_true")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=0)
    parser.add_argument("--max-tool-output-chars", type=int, default=4_000)
    return parser.parse_args()


def _write_summary(output: str, summary: dict) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def _run_provider(
    provider: str,
    defaults: dict,
    tickers: list[str],
    dates: list[str],
    analysts: list[str],
    args: argparse.Namespace,
    prices_by_ticker: dict[str, pd.DataFrame],
) -> dict:
    defaults = defaults.copy()
    quick_override = getattr(args, f"{arg_prefix(provider)}_quick_model", None)
    deep_override = getattr(args, f"{arg_prefix(provider)}_deep_model", None)
    if quick_override:
        defaults["quick_model"] = quick_override
    if deep_override:
        defaults["deep_model"] = deep_override

    llm_provider = defaults.get("llm_provider", provider)
    config = DEFAULT_CONFIG.copy()
    config["us"] = {
        **DEFAULT_CONFIG.get("us", {}),
        "data_provider": args.data_provider,
        "fallback_provider": args.fallback_provider,
        "adjusted_prices": args.adjusted_prices,
    }
    config.update(
        {
            "llm_provider": llm_provider,
            "quick_think_llm": defaults["quick_model"],
            "deep_think_llm": defaults["deep_model"],
            "backend_url": None,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "checkpoint_enabled": args.checkpoint,
            "llm_timeout": args.llm_timeout,
            "llm_max_retries": args.llm_max_retries,
            "max_tool_output_chars": args.max_tool_output_chars,
            "benchmark_symbol": args.benchmark,
            "memory_log_path": f"reports/us_llm_memory_{provider}.md",
            "results_dir": f"reports/us_llm_results/{provider}",
            "data_cache_dir": f"reports/us_llm_cache/{provider}",
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "technical_indicators": "yfinance",
                "fundamental_data": "yfinance",
                "news_data": "yfinance",
            },
        }
    )
    config.update(defaults.get("extra", {}))

    limiter = None if args.disable_rate_limits else _rate_limiter_for(provider, args)
    callbacks = [limiter] if limiter else []
    graph = TradingAgentsGraph(
        selected_analysts=analysts,
        config=config,
        debug=False,
        callbacks=callbacks,
    )
    decisions = []
    signals = []
    for ticker in tickers:
        for decision_date in dates:
            try:
                final_state, rating = graph.propagate(ticker, decision_date)
                decision = final_state["final_trade_decision"]
                decisions.append(
                    {
                        "ticker": ticker,
                        "date": decision_date,
                        "rating": rating,
                        "final_trade_decision": decision,
                    }
                )
                signals.append(
                    Signal(
                        ticker=ticker,
                        decision_date=decision_date,
                        rating=rating,
                        final_decision=decision,
                    )
                )
            except Exception as exc:
                decisions.append(
                    {
                        "ticker": ticker,
                        "date": decision_date,
                        "error": str(exc),
                    }
                )

    horizon_results = {}
    for horizon in _horizons(args):
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
            "metrics": asdict(portfolio.metrics),
            "trades": [asdict(t) for t in portfolio.trades],
            "equity_curve": portfolio.equity_curve,
            "skipped_signals": portfolio.skipped_signals,
            "correlation_matrix": portfolio.correlation_matrix,
        }
    legacy_result = InvestmentHorizonSimulator(
        holding_sessions=args.holding_sessions,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    ).simulate(prices_by_ticker, signals)
    return {
        "status": "completed",
        "quick_model": config["quick_think_llm"],
        "deep_model": config["deep_think_llm"],
        "decision_count": len([d for d in decisions if "rating" in d]),
        "error_count": len([d for d in decisions if "error" in d]),
        "decisions": decisions,
        "metrics": asdict(legacy_result.metrics),
        "trades": [asdict(t) for t in legacy_result.trades],
        "portfolio_results": horizon_results,
        "rate_limit": None if limiter is None else asdict(limiter.snapshot()),
    }


def _rate_limiter_for(provider: str, args: argparse.Namespace) -> RollingLLMRateLimiter:
    prefix = arg_prefix(provider)
    return RollingLLMRateLimiter(
        max_requests_per_minute=getattr(args, f"{prefix}_requests_per_minute"),
        max_input_tokens_per_minute=getattr(args, f"{prefix}_input_tokens_per_minute"),
        safety_margin=args.rate_limit_safety_margin,
    )

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


def _horizons(args: argparse.Namespace) -> list[int]:
    if args.horizons:
        return [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    return [args.holding_sessions]


if __name__ == "__main__":
    main()
