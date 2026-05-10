import json

import pytest

import scripts.india_llm_screen as india_llm_screen
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
    monkeypatch.setenv("OPENAI_API_KEY", "key")
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
        provider="openai-gpt-4o-mini",
        packets=[_packet("RELIANCE.NS"), _packet("TCS.NS")],
        output_dir=tmp_path,
        timeout_seconds=5,
    )

    assert len(calls) == 2
    assert "INR" in calls[0]
    assert result["decision_count"] == 2
    assert result["decisions"][0]["model"] == PROVIDER_DEFAULTS["openai-gpt-4o-mini"]["quick_model"]


@pytest.mark.unit
def test_india_openrouter_deepseek_screening_uses_throughput_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    created_kwargs = {}

    class _Response:
        content = json.dumps(
            {
                "rating": "Hold",
                "confidence": 0.5,
                "thesis": "range-bound setup",
                "target_horizon": "20 sessions",
                "full_graph_recommended": False,
            }
        )

    class _LLM:
        def invoke(self, prompt):
            return _Response()

    class _Client:
        def get_llm(self):
            return _LLM()

    def _create(**kwargs):
        created_kwargs.update(kwargs)
        return _Client()

    monkeypatch.setattr(screening, "create_llm_client", _create)
    result = screen_provider(
        provider="openrouter-deepseek-v4",
        packets=[_packet("RELIANCE.NS")],
        output_dir=tmp_path,
        timeout_seconds=5,
    )

    assert result["status"] == "completed"
    assert created_kwargs["provider"] == "openrouter"
    assert created_kwargs["model"] == "deepseek/deepseek-v4-pro"
    assert created_kwargs["extra_body"] == {"provider": {"sort": "throughput"}}


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
            "openai-gpt-4o-mini": {"decisions": [{**_packet("TCS.NS"), "rating": "Hold", "confidence": 0.9}]},
        }
    }
    jobs = select_shortlist(results, max_jobs=10)
    assert any(job["ticker"] == "RELIANCE.NS" for job in jobs)
    assert all(job["ticker"] != "TCS.NS" for job in jobs)


@pytest.mark.unit
def test_india_llm_screen_defaults_to_investment_horizons(monkeypatch):
    monkeypatch.setattr("sys.argv", ["india_llm_screen.py"])
    args = india_llm_screen._parse_args()
    assert india_llm_screen._horizons(args.horizons) == [5, 20, 60, 126]
    assert args.horizon_sessions is None
