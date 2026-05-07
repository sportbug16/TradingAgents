"""Fast one-call LLM screening for US data packets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
from typing import Any

from tradingagents.llm_clients import create_llm_client

from .llm_providers import PROVIDER_DEFAULTS


RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")


@dataclass(frozen=True)
class ScreenDecision:
    ticker: str
    decision_date: str
    rating: str
    confidence: float
    thesis: str
    target_horizon: str
    full_graph_recommended: bool
    provider: str
    model: str
    error: str | None = None


def screen_provider(
    *,
    provider: str,
    packets: list[dict[str, Any]],
    output_dir: str | Path,
    timeout_seconds: int | None = None,
    llm_timeout: float = 60.0,
    llm_max_retries: int = 0,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    provider_output = output_path / f"{provider}.json"
    if provider not in PROVIDER_DEFAULTS:
        result = provider_result(provider, "skipped", [], reason="unsupported provider")
        _write(provider_output, result)
        return result
    defaults = PROVIDER_DEFAULTS[provider]
    if not os.getenv(defaults["key_env"]):
        result = provider_result(provider, "skipped", [], reason=f"missing {defaults['key_env']}")
        _write(provider_output, result)
        return result

    timeout_seconds = timeout_seconds or defaults.get("wall_timeout_seconds")
    try:
        with _wall_timeout(timeout_seconds):
            client = create_llm_client(
                provider=defaults.get("llm_provider", provider),
                model=defaults["quick_model"],
                base_url=None,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
                **_client_kwargs(defaults),
            ).get_llm()
            decisions = [_screen_packet(client, provider, defaults["quick_model"], packet) for packet in packets]
            result = provider_result(provider, "completed", [asdict(d) for d in decisions])
    except TimeoutError as exc:
        result = provider_result(provider, "timeout", [], reason=str(exc))
    except Exception as exc:
        result = provider_result(provider, "error", [], reason=str(exc))
    _write(provider_output, result)
    return result


def merge_provider_outputs(output_dir: str | Path, providers: list[str]) -> dict[str, Any]:
    root = Path(output_dir)
    provider_results = {}
    for provider in providers:
        path = root / f"{provider}.json"
        if path.exists():
            provider_results[provider] = json.loads(path.read_text(encoding="utf-8"))
        else:
            provider_results[provider] = provider_result(provider, "missing", [], reason="provider output not found")
    merged = {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "providers": provider_results,
    }
    _write(root / "merged.json", merged)
    return merged


def parse_screen_response(text: str) -> dict[str, Any]:
    payload = _extract_json_object(str(text))
    rating = str(payload.get("rating", "")).title()
    if rating not in RATINGS:
        raise ValueError(f"invalid rating: {rating}")
    confidence = max(0.0, min(float(payload.get("confidence", 0.0)), 1.0))
    return {
        "rating": rating,
        "confidence": confidence,
        "thesis": str(payload.get("thesis", ""))[:1200],
        "target_horizon": str(payload.get("target_horizon", ""))[:80],
        "full_graph_recommended": _to_bool(payload.get("full_graph_recommended", rating in {"Buy", "Overweight"})),
    }


def _screen_packet(client, provider: str, model: str, packet: dict[str, Any]) -> ScreenDecision:
    prompt = _screen_prompt(packet)
    try:
        response = client.invoke(prompt)
        parsed = parse_screen_response(getattr(response, "content", response))
        return ScreenDecision(
            ticker=packet["ticker"],
            decision_date=packet["decision_date"],
            provider=provider,
            model=model,
            **parsed,
        )
    except Exception as exc:
        return ScreenDecision(
            ticker=packet["ticker"],
            decision_date=packet["decision_date"],
            provider=provider,
            model=model,
            rating="Hold",
            confidence=0.0,
            thesis="",
            target_horizon=str(packet.get("horizon_sessions", "")),
            full_graph_recommended=False,
            error=str(exc),
        )


def _screen_prompt(packet: dict[str, Any]) -> str:
    compact = {
        "ticker": packet["ticker"],
        "decision_date": packet["decision_date"],
        "horizon_sessions": packet["horizon_sessions"],
        "instrument": packet["instrument"],
        "price_features": packet["price_features"],
        "benchmark_features": packet["benchmark_features"],
        "correlation_features": packet["correlation_features"],
        "data_quality": packet["data_quality"],
    }
    return (
        "You are screening US mega-cap equities for a long-only investment study. "
        "Use only the JSON data provided. Return exactly one JSON object with keys: "
        "rating, confidence, thesis, target_horizon, full_graph_recommended. "
        "rating must be one of Buy, Overweight, Hold, Underweight, Sell. "
        "confidence is 0.0 to 1.0. full_graph_recommended is true if a deeper agent run is warranted.\n\n"
        f"DATA:\n{json.dumps(compact, separators=(',', ':'), default=str)}"
    )


def write_provider_status(
    output_dir: str | Path,
    provider: str,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    result = provider_result(provider, status, [], reason=reason)
    _write(Path(output_dir) / f"{provider}.json", result)
    return result


def provider_result(provider: str, status: str, decisions: list[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    return {
        "provider": provider,
        "status": status,
        "reason": reason,
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "decision_count": len([d for d in decisions if not d.get("error")]),
        "error_count": len([d for d in decisions if d.get("error")]),
        "decisions": decisions,
    }


def _client_kwargs(defaults: dict[str, Any]) -> dict[str, Any]:
    extra = defaults.get("extra", {})
    kwargs: dict[str, Any] = {}
    if "anthropic_effort" in extra:
        kwargs["effort"] = extra["anthropic_effort"]
    if "google_thinking_level" in extra:
        kwargs["thinking_level"] = extra["google_thinking_level"]
    return kwargs


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("screen response did not contain a JSON object")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class _wall_timeout:
    def __init__(self, seconds: int | None):
        self.seconds = seconds
        self.previous = None

    def __enter__(self):
        if not self.seconds or not hasattr(signal, "SIGALRM"):
            return self
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(int(self.seconds))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous)
        return False

    def _raise(self, signum, frame):
        raise TimeoutError(f"provider wall timeout after {self.seconds} seconds")
