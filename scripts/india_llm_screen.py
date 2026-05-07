"""Generate India packets and run one-call LLM screening."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.india.llm_providers import PROVIDER_DEFAULTS, provider_labels
from tradingagents.india.packets import generate_packets
from tradingagents.india.screening import merge_provider_outputs, screen_provider, write_provider_status


DEFAULT_TICKERS = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,SBIN.NS"


def main() -> None:
    load_dotenv()
    args = _parse_args()
    output_dir = Path(args.output)
    packet_result = generate_packets(
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        dates=[d.strip() for d in args.dates.split(",") if d.strip()],
        output_dir=args.packet_output or output_dir / "packets",
        price_start=args.start,
        price_end=args.end,
        horizon_sessions=args.horizon_sessions,
        data_provider_name=args.data_provider,
        allow_fallback=args.allow_fallback,
        benchmark=args.benchmark,
    )
    packets = packet_result["packets"]
    providers = provider_labels(args.providers)
    if args.parallel_providers and len(providers) > 1:
        _run_providers_parallel(args, providers, packets, output_dir)
    else:
        for provider in providers:
            _run_provider_inline(args, provider, packets, output_dir)
    merged = merge_provider_outputs(output_dir, providers)
    merged["packet_manifest"] = packet_result["manifest"]
    (output_dir / "merged.json").write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(json.dumps(merged, indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=DEFAULT_TICKERS)
    parser.add_argument("--dates", default="2026-02-02")
    parser.add_argument("--providers", default="anthropic,google,openrouter-openai-4o-mini,openrouter-deepseek-v4")
    parser.add_argument("--horizon-sessions", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--data-provider", default="dhan")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--provider-wall-timeout-seconds", type=int, default=None)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=0)
    parser.add_argument("--parallel-providers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packet-output", default=None)
    parser.add_argument("--output", default="reports/india_llm_screen/default")
    return parser.parse_args()


def _provider_timeout(args: argparse.Namespace, provider: str) -> int | None:
    return args.provider_wall_timeout_seconds or PROVIDER_DEFAULTS.get(provider, {}).get("wall_timeout_seconds")


def _run_provider_inline(args: argparse.Namespace, provider: str, packets: list[dict], output_dir: Path) -> None:
    screen_provider(
        provider=provider,
        packets=packets,
        output_dir=output_dir,
        timeout_seconds=_provider_timeout(args, provider),
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
    )


def _run_providers_parallel(args: argparse.Namespace, providers: list[str], packets: list[dict], output_dir: Path) -> None:
    ctx = mp.get_context("spawn")
    processes: dict[str, mp.Process] = {}
    for provider in providers:
        process = ctx.Process(
            target=_provider_process_entry,
            args=(provider, packets, str(output_dir), _provider_timeout(args, provider), args.llm_timeout, args.llm_max_retries),
        )
        process.start()
        processes[provider] = process

    for provider, process in processes.items():
        timeout = _provider_timeout(args, provider)
        process.join(None if timeout is None else timeout + 5)
        if process.is_alive():
            process.terminate()
            process.join(5)
            write_provider_status(output_dir, provider, "timeout", f"provider process exceeded {timeout} seconds")
        elif process.exitcode not in (0, None) and not (output_dir / f"{provider}.json").exists():
            write_provider_status(output_dir, provider, "error", f"provider process exited with code {process.exitcode}")


def _provider_process_entry(
    provider: str,
    packets: list[dict],
    output_dir: str,
    timeout_seconds: int | None,
    llm_timeout: float,
    llm_max_retries: int,
) -> None:
    load_dotenv()
    screen_provider(
        provider=provider,
        packets=packets,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        llm_timeout=llm_timeout,
        llm_max_retries=llm_max_retries,
    )


if __name__ == "__main__":
    main()
