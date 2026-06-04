"""Run a guarded paper-live Indian equities monitor.

The runner polls market data, creates deterministic pilot signals, and routes
paper orders through ``PaperBroker``. LLM graph calls are deliberately kept out
of this market-hours monitor; LLM decisions should be precomputed after close
or before open and treated as a thesis/risk overlay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, time
import json
from pathlib import Path
import sys
import time as time_module

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.broker import BrokerPosition, OrderSide, PaperBroker
from tradingagents.india.data import create_indian_provider, snapshot_dict
from tradingagents.india.market import IndianInstrumentRegistry, IndianTradingCalendar


DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]


def main() -> None:
    load_dotenv()
    args = _parse_args()
    tickers = _load_tickers(args)
    events: list[dict] = []
    registry = IndianInstrumentRegistry()
    provider = create_indian_provider(args.data_provider, allow_fallback=args.allow_fallback)
    calendar = IndianTradingCalendar.from_csv(args.holidays_csv) if args.holidays_csv else IndianTradingCalendar()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    broker, previous_events = _load_broker(output_path, args.cash)
    events.extend(previous_events)

    deadline, runtime_mode = _resolve_deadline(datetime.now(), args, calendar)
    while True:
        now = datetime.now()
        if not args.once and now > deadline:
            break
        market_open = args.allow_outside_market_hours or _is_market_open(now, calendar)
        batch_events = _poll_once(
            tickers,
            registry,
            provider,
            broker,
            args.order_value,
            args.lookback_days,
            allow_trading=market_open,
        )
        events.extend(batch_events)
        snapshot = {
            "updated_at": now.isoformat(timespec="seconds"),
            "mode": "paper",
            "tickers": tickers,
            "data_provider": getattr(provider, "name", args.data_provider),
            "allow_fallback": args.allow_fallback,
            "runtime_mode": runtime_mode,
            "session_deadline": deadline.isoformat(timespec="seconds"),
            "market_open": market_open,
            "cash": broker.cash,
            "positions": [asdict(p) for p in broker.positions()],
            "orders": [asdict(o) for o in broker.orders()],
            "events": events[-200:],
            "errors": [event for event in events[-200:] if event.get("error")],
        }
        output_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"updated_at": snapshot["updated_at"], "events": batch_events}, default=str))
        if args.once:
            break
        remaining_seconds = (deadline - now).total_seconds()
        if remaining_seconds <= 0:
            break
        time_module.sleep(min(args.poll_seconds, remaining_seconds))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--tickers-file", default=None)
    parser.add_argument("--cash", type=float, default=1_000_000)
    parser.add_argument("--order-value", type=float, default=50_000)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--duration-minutes", type=int, default=375)
    parser.add_argument("--output", default="reports/india_paper_live.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--allow-outside-market-hours", action="store_true")
    parser.add_argument("--data-provider", default="dhan")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--holidays-csv", default=None)
    return parser.parse_args()


def _load_tickers(args: argparse.Namespace) -> list[str]:
    if not args.tickers_file:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    path = Path(args.tickers_file)
    if not path.exists():
        raise FileNotFoundError(f"tickers file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("tickers") or payload.get("symbols") or payload.get("watchlist")
        else:
            values = payload
        if not isinstance(values, list):
            raise ValueError("JSON tickers file must be a list or an object with tickers/symbols/watchlist")
        return [str(t).strip().upper() for t in values if str(t).strip()]

    text = path.read_text(encoding="utf-8")
    values = [item.strip() for line in text.splitlines() for item in line.split(",")]
    if values and values[0].lower() in {"ticker", "tickers", "symbol", "symbols"}:
        values = values[1:]
    return [value.upper() for value in values if value]


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
        broker = PaperBroker(cash=float(snapshot.get("cash", default_cash)), positions=positions)
        return broker, list(snapshot.get("events", []))
    except Exception:
        return PaperBroker(cash=default_cash), []


def _is_market_open(now: datetime, calendar: IndianTradingCalendar | None = None) -> bool:
    calendar = calendar or IndianTradingCalendar()
    if not calendar.is_session(now):
        return False
    current = now.time()
    return time(9, 15) <= current <= time(15, 30)


def _resolve_deadline(
    now: datetime,
    args: argparse.Namespace,
    calendar: IndianTradingCalendar,
) -> tuple[datetime, str]:
    if args.once:
        return now, "once"
    if args.allow_outside_market_hours:
        return now + timedelta(minutes=args.duration_minutes), "fixed_duration"
    if not calendar.is_session(now):
        return now, "non_session_day"

    market_close = datetime.combine(now.date(), time(15, 30))
    if now >= market_close:
        return now, "market_closed"
    return market_close, "market_close"


def _poll_once(
    tickers: list[str],
    registry: IndianInstrumentRegistry,
    provider,
    broker: PaperBroker,
    order_value: float,
    lookback_days: int,
    allow_trading: bool,
) -> list[dict]:
    events = []
    end = (datetime.now() + timedelta(days=1)).date().isoformat()
    start = (datetime.now() - timedelta(days=max(35, lookback_days * 3))).date().isoformat()
    for ticker in tickers:
        try:
            instrument = registry.resolve(ticker)
            result = provider.get_ohlcv(instrument, start, end)
            data = result.data
            if len(data) < lookback_days + 1:
                events.append(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "ticker": instrument.data_symbol,
                        "data_snapshot": snapshot_dict(result.snapshot),
                        "error": f"insufficient market data rows: {len(data)}",
                    }
                )
                continue
            frame = data.copy().sort_values("Date").reset_index(drop=True)
            last = frame.iloc[-1]
            close = float(last["Close"])
            sma = float(frame["Close"].tail(lookback_days).mean())
            broker.set_price(instrument.data_symbol, close)
            existing_qty = sum(p.quantity for p in broker.positions() if p.symbol == instrument.data_symbol)
            signal = "BUY" if close > sma else "HOLD"
            order = None
            blocked_reason = None
            if signal == "BUY" and existing_qty > 0:
                blocked_reason = "existing_position"
            if signal == "BUY" and existing_qty == 0 and not allow_trading:
                blocked_reason = "market_closed"
            elif signal == "BUY" and existing_qty == 0:
                quantity = max(1, int(order_value // close))
                order = broker.place_order(instrument.data_symbol, OrderSide.BUY, quantity, manual_approval=True)
            events.append(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "ticker": instrument.data_symbol,
                    "close": close,
                    "sma": sma,
                    "signal": signal,
                    "order": asdict(order) if order else None,
                    "blocked_reason": blocked_reason,
                    "data_snapshot": snapshot_dict(result.snapshot),
                }
            )
        except Exception as exc:
            events.append(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "ticker": ticker,
                    "error": str(exc),
                    "data_provider": getattr(provider, "name", None),
                }
            )
    return events


if __name__ == "__main__":
    main()
