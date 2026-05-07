"""Generate cached model-neutral Indian market data packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.india.packets import generate_packets


DEFAULT_TICKERS = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,SBIN.NS"


def main() -> None:
    load_dotenv()
    args = _parse_args()
    result = generate_packets(
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        dates=[d.strip() for d in args.dates.split(",") if d.strip()],
        output_dir=args.output,
        price_start=args.start,
        price_end=args.end,
        horizon_sessions=args.horizon_sessions,
        data_provider_name=args.data_provider,
        allow_fallback=args.allow_fallback,
        benchmark=args.benchmark,
    )
    print(json.dumps(result["manifest"], indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=DEFAULT_TICKERS)
    parser.add_argument("--dates", default="2026-02-02")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--horizon-sessions", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--data-provider", default="dhan")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--output", default="reports/india_data_packets/default")
    return parser.parse_args()


if __name__ == "__main__":
    main()
