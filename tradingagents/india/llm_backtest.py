"""Convert Indian LLM screening outputs into backtestable signal sets."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from tradingagents.backtesting.rules import Signal


DEFAULT_RESEARCH_PROVIDERS = ("anthropic", "google", "openai-gpt-5-4", "openrouter-deepseek-v4")
EXCLUDED_RESEARCH_PROVIDERS = {"openrouter-openai-4o-mini"}
AGGRESSIVE_RATINGS = {"Buy", "Overweight"}
DEFENSIVE_RATINGS = {"Sell", "Underweight"}
RATING_SCORES = {"Sell": -2.0, "Underweight": -1.0, "Hold": 0.0, "Overweight": 1.0, "Buy": 2.0}


@dataclass(frozen=True)
class LLMBacktestDecision:
    ticker: str
    decision_date: str
    horizon_sessions: int
    rating: str
    confidence: float
    thesis: str
    target_horizon: str
    provider: str
    model: str

    @property
    def signal_id(self) -> str:
        return f"{self.provider}:{self.model}:h{self.horizon_sessions}"


def provider_labels(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_RESEARCH_PROVIDERS)
    labels = [p.strip().lower() for p in value.split(",") if p.strip()]
    return [p for p in labels if p not in EXCLUDED_RESEARCH_PROVIDERS]


def extract_decisions(
    screen_results: dict[str, Any],
    *,
    providers: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
) -> list[LLMBacktestDecision]:
    selected_providers = set(providers or DEFAULT_RESEARCH_PROVIDERS)
    selected_horizons = set(horizons or ())
    legacy_horizons = _legacy_horizon_index(screen_results)
    decisions: list[LLMBacktestDecision] = []

    for provider, result in screen_results.get("providers", {}).items():
        if provider in EXCLUDED_RESEARCH_PROVIDERS or provider not in selected_providers:
            continue
        for index, decision in enumerate(result.get("decisions", [])):
            if decision.get("error"):
                continue
            horizon = _decision_horizon(decision, legacy_horizons.get(provider, []), index)
            if horizon is None or (selected_horizons and horizon not in selected_horizons):
                continue
            rating = str(decision.get("rating", "Hold")).title()
            if rating not in RATING_SCORES:
                rating = "Hold"
            decisions.append(
                LLMBacktestDecision(
                    ticker=str(decision["ticker"]).upper(),
                    decision_date=str(decision["decision_date"]),
                    horizon_sessions=horizon,
                    rating=rating,
                    confidence=max(0.0, min(float(decision.get("confidence", 0.0)), 1.0)),
                    thesis=str(decision.get("thesis", "")),
                    target_horizon=str(decision.get("target_horizon", "")),
                    provider=provider,
                    model=str(decision.get("model") or result.get("metadata", {}).get("model") or ""),
                )
            )
    return decisions


def build_signal_sets(decisions: Iterable[LLMBacktestDecision], horizon: int) -> dict[str, list[Signal]]:
    horizon_decisions = [d for d in decisions if d.horizon_sessions == horizon]
    signal_sets: dict[str, list[Signal]] = {}

    for provider in sorted({d.provider for d in horizon_decisions}):
        provider_decisions = [d for d in horizon_decisions if d.provider == provider]
        signal_sets[f"provider:{provider}"] = [_signal_from_decision(d, f"provider:{provider}:h{horizon}") for d in provider_decisions]

    grouped: dict[tuple[str, str], list[LLMBacktestDecision]] = {}
    for decision in horizon_decisions:
        grouped.setdefault((decision.ticker, decision.decision_date), []).append(decision)

    signal_sets["ensemble:consensus"] = _consensus_signals(grouped, horizon)
    signal_sets["ensemble:confidence_weighted"] = _confidence_weighted_signals(grouped, horizon)
    signal_sets["ensemble:disagreement_filtered"] = _disagreement_filtered_signals(grouped, horizon)
    return signal_sets


def _signal_from_decision(decision: LLMBacktestDecision, run_id: str) -> Signal:
    return Signal(
        ticker=decision.ticker,
        decision_date=decision.decision_date,
        rating=decision.rating,
        final_decision=(
            f"LLM screen provider={decision.provider} model={decision.model} "
            f"confidence={decision.confidence:.2f} thesis={decision.thesis}"
        ),
        run_id=run_id,
    )


def _consensus_signals(grouped: dict[tuple[str, str], list[LLMBacktestDecision]], horizon: int) -> list[Signal]:
    signals = []
    for (ticker, decision_date), decisions in grouped.items():
        aggressive = [d for d in decisions if d.rating in AGGRESSIVE_RATINGS]
        defensive = [d for d in decisions if d.rating in DEFENSIVE_RATINGS]
        min_votes = 2 if len(decisions) >= 3 else 1
        if len(aggressive) < min_votes or defensive:
            rating = "Hold"
        else:
            avg_score = sum(RATING_SCORES[d.rating] for d in aggressive) / len(aggressive)
            rating = "Buy" if avg_score >= 1.5 else "Overweight"
        signals.append(_ensemble_signal(ticker, decision_date, rating, decisions, f"ensemble:consensus:h{horizon}"))
    return signals


def _confidence_weighted_signals(grouped: dict[tuple[str, str], list[LLMBacktestDecision]], horizon: int) -> list[Signal]:
    signals = []
    for (ticker, decision_date), decisions in grouped.items():
        score = _weighted_score(decisions)
        signals.append(
            _ensemble_signal(
                ticker,
                decision_date,
                _score_to_rating(score),
                decisions,
                f"ensemble:confidence_weighted:h{horizon}",
            )
        )
    return signals


def _disagreement_filtered_signals(grouped: dict[tuple[str, str], list[LLMBacktestDecision]], horizon: int) -> list[Signal]:
    signals = []
    for (ticker, decision_date), decisions in grouped.items():
        ratings = {d.rating for d in decisions}
        has_disagreement = bool(ratings & AGGRESSIVE_RATINGS) and bool(ratings & DEFENSIVE_RATINGS)
        score = _weighted_score(decisions)
        rating = _score_to_rating(score) if has_disagreement and score > 0 else "Hold"
        signals.append(_ensemble_signal(ticker, decision_date, rating, decisions, f"ensemble:disagreement_filtered:h{horizon}"))
    return signals


def _ensemble_signal(
    ticker: str,
    decision_date: str,
    rating: str,
    decisions: list[LLMBacktestDecision],
    run_id: str,
) -> Signal:
    votes = ",".join(f"{d.provider}:{d.rating}:{d.confidence:.2f}" for d in sorted(decisions, key=lambda d: d.provider))
    return Signal(
        ticker=ticker,
        decision_date=decision_date,
        rating=rating,
        final_decision=f"LLM ensemble rating={rating}; votes={votes}",
        run_id=run_id,
    )


def _weighted_score(decisions: list[LLMBacktestDecision]) -> float:
    total_weight = sum(max(d.confidence, 0.01) for d in decisions)
    if total_weight <= 0:
        return 0.0
    return sum(RATING_SCORES[d.rating] * max(d.confidence, 0.01) for d in decisions) / total_weight


def _score_to_rating(score: float) -> str:
    if score >= 1.25:
        return "Buy"
    if score >= 0.5:
        return "Overweight"
    if score > -0.5:
        return "Hold"
    if score > -1.25:
        return "Underweight"
    return "Sell"


def _decision_horizon(decision: dict[str, Any], legacy_horizons: list[int], index: int) -> int | None:
    raw = decision.get("horizon_sessions")
    if raw not in (None, ""):
        try:
            horizon = int(raw)
            return horizon if horizon > 0 else None
        except (TypeError, ValueError):
            pass
    if index < len(legacy_horizons):
        return legacy_horizons[index]
    return _parse_horizon_text(decision.get("target_horizon"))


def _parse_horizon_text(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _legacy_horizon_index(screen_results: dict[str, Any]) -> dict[str, list[int]]:
    manifests = screen_results.get("packet_manifest", {}).get("manifests", [])
    horizons: list[int] = []
    for manifest in manifests:
        config = manifest.get("config", {})
        horizon = config.get("horizon_sessions")
        if horizon is None:
            continue
        count = int(manifest.get("packet_count") or len(manifest.get("packet_paths", [])) or 0)
        horizons.extend([int(horizon)] * count)
    return {provider: list(horizons) for provider in screen_results.get("providers", {})}
