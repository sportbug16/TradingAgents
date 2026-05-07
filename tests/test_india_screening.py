import json

import pytest

from tradingagents.india import screening
from tradingagents.india.llm_providers import PROVIDER_DEFAULTS
from tradingagents.india.screening import merge_provider_outputs, parse_screen_response, screen_provider, write_provider_status
from tradingagents.india.shortlist import select_shortlist


def _packet(ticker="RELIANCE.NS"):
    return {
        "ticker": ticker,
        "decision_date": "2026-02-02",
        "horizon_sessions": 3,
        "instrument": {"symbol": ticker, "sector": "Energy", "exchange": "NSE"},
        "price_features": {"return_5d": 0.03, "volatility_20d": 0.2},
        "benchmark_features": {"^NSEI": {"return_5d": 0.01}},
        "correlation_features": {"cluster": "single_name", "top_correlations": []},
        "data_quality": {"provider": "fixture", "fallback_unofficial": False},
    }


@pytest.mark.unit
def test_india_screen_response_parses_json():
    parsed = parse_screen_response(
        'notes {"rating":"overweight","confidence":0.7,"thesis":"INR swing setup",'
        '"target_horizon":"3 sessions","full_graph_recommended":true}'
    )
    assert parsed["rating"] == "Overweight"
    assert parsed["confidence"] == 0.7
    assert parsed["full_graph_recommended"] is True


@pytest.mark.unit
def test_india_screen_provider_makes_one_call_per_packet(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    calls = []

    class _Response:
        content = json.dumps(
            {
                "rating": "Buy",
                "confidence": 0.8,
                "thesis": "strong relative move",
                "target_horizon": "3 sessions",
                "full_graph_recommended": True,
            }
        )

    class _LLM:
        def invoke(self, prompt):
            calls.append(prompt)
            return _Response()

    class _Client:
        def get_llm(self):
            return _LLM()

    monkeypatch.setattr(screening, "create_llm_client", lambda **kwargs: _Client())
    result = screen_provider(
        provider="openrouter-openai-4o-mini",
        packets=[_packet("RELIANCE.NS"), _packet("TCS.NS")],
        output_dir=tmp_path,
        timeout_seconds=5,
    )

    assert len(calls) == 2
    assert "INR" in calls[0]
    assert result["decision_count"] == 2
    assert result["decisions"][0]["model"] == PROVIDER_DEFAULTS["openrouter-openai-4o-mini"]["quick_model"]


@pytest.mark.unit
def test_india_screen_merge_keeps_partial_provider_outputs(tmp_path):
    write_provider_status(tmp_path, "anthropic", "timeout", "slow provider")
    merged = merge_provider_outputs(tmp_path, ["anthropic", "google"])
    assert merged["providers"]["anthropic"]["status"] == "timeout"
    assert merged["providers"]["google"]["status"] == "missing"


@pytest.mark.unit
def test_india_shortlist_includes_entry_and_disagreement():
    results = {
        "providers": {
            "anthropic": {"decisions": [{**_packet("RELIANCE.NS"), "rating": "Buy", "confidence": 0.8}]},
            "google": {"decisions": [{**_packet("RELIANCE.NS"), "rating": "Sell", "confidence": 0.7}]},
            "openrouter-openai-4o-mini": {"decisions": [{**_packet("TCS.NS"), "rating": "Hold", "confidence": 0.9}]},
        }
    }
    jobs = select_shortlist(results, max_jobs=10)
    assert any(job["ticker"] == "RELIANCE.NS" for job in jobs)
    assert all(job["ticker"] != "TCS.NS" for job in jobs)
