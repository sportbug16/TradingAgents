"""Shortlist selection from Indian LLM screen results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AGGRESSIVE_RATINGS = {"Buy", "Overweight"}


def select_shortlist(screen_results: dict[str, Any], max_jobs: int = 10) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for provider, result in screen_results.get("providers", {}).items():
        for decision in result.get("decisions", []):
            if decision.get("error"):
                continue
            key = (decision["ticker"], decision["decision_date"])
            grouped.setdefault(key, []).append({**decision, "provider": provider})

    jobs = []
    for (ticker, decision_date), decisions in grouped.items():
        ratings = {d.get("rating") for d in decisions}
        has_entry = any(r in AGGRESSIVE_RATINGS for r in ratings)
        disagreement = len(ratings) >= 3 or (AGGRESSIVE_RATINGS & ratings and {"Sell", "Underweight"} & ratings)
        if not has_entry and not disagreement:
            continue
        for decision in decisions:
            if decision.get("rating") in AGGRESSIVE_RATINGS or disagreement:
                jobs.append(
                    {
                        "ticker": ticker,
                        "decision_date": decision_date,
                        "provider": decision["provider"],
                        "rating": decision.get("rating"),
                        "confidence": float(decision.get("confidence", 0.0)),
                        "reason": "entry_signal" if decision.get("rating") in AGGRESSIVE_RATINGS else "disagreement",
                    }
                )

    jobs.sort(key=lambda item: (item["reason"] != "entry_signal", -item["confidence"], item["ticker"], item["provider"]))
    return jobs[:max_jobs]


def load_screen_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
