"""Run full TradingAgentsGraph only on shortlisted US screen results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.us.llm_providers import PROVIDER_DEFAULTS
from tradingagents.us.shortlist import load_screen_results, select_shortlist


def main() -> None:
    load_dotenv()
    args = _parse_args()
    screen_results = load_screen_results(args.screen_results)
    jobs = select_shortlist(screen_results, max_jobs=args.max_jobs)
    summary = {"screen_results": args.screen_results, "jobs": jobs, "providers": {}}
    for job in jobs:
        provider = job["provider"]
        if provider not in PROVIDER_DEFAULTS:
            continue
        defaults = PROVIDER_DEFAULTS[provider]
        if not os.getenv(defaults["key_env"]):
            summary["providers"].setdefault(provider, {"status": "skipped", "reason": f"missing {defaults['key_env']}", "decisions": []})
            continue
        try:
            result = _run_job(job, defaults, args)
        except Exception as exc:
            result = {**job, "error": str(exc)}
        summary["providers"].setdefault(provider, {"status": "completed", "decisions": []})
        summary["providers"][provider]["decisions"].append(result)
    _write(args.output, summary)
    print(json.dumps(summary, indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-results", required=True)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--analysts", default="market")
    parser.add_argument("--holding-sessions", type=int, default=20)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-max-retries", type=int, default=0)
    parser.add_argument("--output", default="reports/us_full_graph_shortlist/default.json")
    return parser.parse_args()


def _run_job(job: dict, defaults: dict, args: argparse.Namespace) -> dict:
    provider = job["provider"]
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "batch_mode": True,
            "disable_memory_reflection": True,
            "llm_provider": defaults.get("llm_provider", provider),
            "quick_think_llm": defaults["quick_model"],
            "deep_think_llm": defaults["deep_model"],
            "llm_timeout": args.llm_timeout,
            "llm_max_retries": args.llm_max_retries,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "memory_log_path": f"reports/us_full_graph_memory_{provider}.md",
            "results_dir": f"reports/us_full_graph_results/{provider}",
            "data_cache_dir": f"reports/us_full_graph_cache/{provider}",
        }
    )
    config.update(defaults.get("extra", {}))
    graph = TradingAgentsGraph(
        selected_analysts=[a.strip() for a in args.analysts.split(",") if a.strip()],
        config=config,
        debug=False,
    )
    final_state, rating = graph.propagate(job["ticker"], job["decision_date"])
    return {
        **job,
        "full_graph_rating": rating,
        "final_trade_decision": final_state["final_trade_decision"],
    }


def _write(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
