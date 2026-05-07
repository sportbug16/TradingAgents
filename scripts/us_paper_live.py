"""Run a guarded US market paper-investing session."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time as time_module
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.broker import BrokerPosition, OrderSide, PaperBroker
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.us.data import create_us_data_provider, snapshot_dict
from tradingagents.us.market import DEFAULT_US_TICKERS, USInstrumentRegistry, US_TIMEZONE, is_us_market_open


def main() -> None:
    load_dotenv()
    args = _parse_args()
    registry = USInstrumentRegistry()
    tickers = registry.validate_many([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    broker, events = _load_broker(output_path, args.cash)
    data_provider = create_us_data_provider(_data_config(args))

    deadline = time_module.time() + args.duration_minutes * 60
    while args.once or time_module.time() < deadline:
        now = datetime.now(ZoneInfo(US_TIMEZONE))
        provider_market_open = data_provider.is_market_open()
        batch_events = _poll_once(
            tickers,
            broker,
            data_provider,
            order_value=args.order_value,
            fast_window=args.fast_window,
            slow_window=args.slow_window,
            adjusted=args.adjusted_prices,
            allow_trading=args.allow_outside_market_hours or provider_market_open or is_us_market_open(now),
        )
        events.extend(batch_events)
        snapshot = {
            "updated_at": now.isoformat(timespec="seconds"),
            "mode": "paper",
            "market": "US",
            "universe": "configured Nasdaq-listed S&P 500 seed",
            "data_provider": getattr(data_provider, "name", args.data_provider),
            "adjusted_prices": args.adjusted_prices,
            "tickers": tickers,
            "cash": broker.cash,
            "positions": [asdict(p) for p in broker.positions()],
            "orders": [asdict(o) for o in broker.orders()],
            "events": events[-300:],
        }
        output_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"updated_at": snapshot["updated_at"], "events": batch_events}, default=str))
        if args.once:
            break
        time_module.sleep(args.poll_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=",".join(DEFAULT_US_TICKERS))
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--order-value", type=float, default=5_000)
    parser.add_argument("--fast-window", type=int, default=20)
    parser.add_argument("--slow-window", type=int, default=60)
    parser.add_argument("--data-provider", default="massive")
    parser.add_argument("--fallback-provider", default="yfinance")
    parser.add_argument("--adjusted-prices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--duration-minutes", type=int, default=390)
    parser.add_argument("--output", default="reports/us_paper_live.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--allow-outside-market-hours", action="store_true")
    return parser.parse_args()


def _load_broker(output_path: Path, default_cash: float) -> tuple[PaperBroker, list[dict]]:
    if not output_path.exists():
        return PaperBroker(cash=default_cash), []
    try:
        snapshot = json.loads(output_path.read_text(encoding="utf-8"))
        positions = [
            BrokerPosition(
                symbol=p["symbol"],
                quantity=int(p["quantity"]),
                average_price=float(p["average_price"]),
                last_price=float(p["last_price"]),
            )
            for p in snapshot.get("positions", [])
        ]
        return PaperBroker(cash=float(snapshot.get("cash", default_cash)), positions=positions), list(snapshot.get("events", []))
    except Exception:
        return PaperBroker(cash=default_cash), []


def _poll_once(
    tickers: list[str],
    broker: PaperBroker,
    data_provider,
    order_value: float,
    fast_window: int,
    slow_window: int,
    adjusted: bool,
    allow_trading: bool,
) -> list[dict]:
    events = []
    for ticker in tickers:
        try:
            end = datetime.now(ZoneInfo(US_TIMEZONE)).date()
            start = end.replace(year=end.year - 1)
            market_data = data_provider.get_ohlcv(
                ticker,
                start.isoformat(),
                (end + pd_timedelta_days(1)).isoformat(),
                adjusted=adjusted,
            )
            data = market_data.data
            if len(data) < slow_window + 1:
                events.append(
                    {
                        "time": datetime.now(ZoneInfo(US_TIMEZONE)).isoformat(timespec="seconds"),
                        "ticker": ticker,
                        "error": f"insufficient market data rows: {len(data)}",
                        "data_snapshot": snapshot_dict(market_data.snapshot),
                    }
                )
                continue
            frame = data.reset_index(drop=True)
            last = frame.iloc[-1]
            close = float(last["Close"])
            fast = float(frame["Close"].tail(fast_window).mean())
            slow = float(frame["Close"].tail(slow_window).mean())
            broker.set_price(ticker, close)
            existing_qty = sum(p.quantity for p in broker.positions() if p.symbol == ticker)
            signal = "BUY" if close > fast > slow else "HOLD"
            order = None
            blocked_reason = None
            if signal == "BUY" and existing_qty == 0 and not allow_trading:
                blocked_reason = "market_closed"
            elif signal == "BUY" and existing_qty == 0:
                quantity = max(1, int(order_value // close))
                order = broker.place_order(ticker, OrderSide.BUY, quantity, manual_approval=True)
            events.append(
                {
                    "time": datetime.now(ZoneInfo(US_TIMEZONE)).isoformat(timespec="seconds"),
                    "ticker": ticker,
                    "close": close,
                    "fast_sma": fast,
                    "slow_sma": slow,
                    "signal": signal,
                    "order": asdict(order) if order else None,
                    "blocked_reason": blocked_reason,
                    "data_snapshot": snapshot_dict(market_data.snapshot),
                }
            )
        except Exception as exc:
            events.append(
                {
                    "time": datetime.now(ZoneInfo(US_TIMEZONE)).isoformat(timespec="seconds"),
                    "ticker": ticker,
                    "error": str(exc),
                }
            )
    return events


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


def pd_timedelta_days(days: int):
    from datetime import timedelta

    return timedelta(days=days)


if __name__ == "__main__":
    main()
