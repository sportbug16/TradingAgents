"""Deterministic portfolio rules for TradingAgents ratings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_RATING_WEIGHTS = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": 0.0,
    "Sell": 0.0,
}


@dataclass(frozen=True)
class RatingPolicy:
    """Maps final 5-tier ratings to long-only cash-equity target weights."""

    weights: Mapping[str, float] | None = None
    max_position_weight: float = 1.0

    def weight_for(self, rating: str) -> float:
        raw = (self.weights or DEFAULT_RATING_WEIGHTS).get(rating.title(), 0.0)
        return max(0.0, min(float(raw), self.max_position_weight))


@dataclass(frozen=True)
class Signal:
    ticker: str
    decision_date: str
    rating: str
    final_decision: str = ""
    run_id: str | None = None


def rating_to_target_weight(rating: str, policy: RatingPolicy | None = None) -> float:
    return (policy or RatingPolicy()).weight_for(rating)
