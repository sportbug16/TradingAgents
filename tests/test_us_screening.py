import json

import pytest

from tradingagents.us import screening
from tradingagents.us.llm_providers import PROVIDER_DEFAULTS
from tradingagents.us.screening import merge_provider_outputs, parse_screen_response, screen_provider, write_provider_status


def _packet(ticker="AAPL"):
    return {
        "ticker": ticker,
        "decision_date": "2026-02-02",
        "horizon_sessions": 20,
        "instrument": {"symbol": ticker, "sector": "Information Technology"},
        "price_features": {"return_20d": 0.05, "volatility_20d": 0.2},
        "benchmark_features": {"SPY": {"return_20d": 0.02}, "QQQ": {"return_20d": 0.03}},
        "correlation_features": {"cluster": "high_beta_tech", "top_correlations": []},
        "data_quality": {"provider": "fixture", "fallback_unofficial": False},
    }


@pytest.mark.unit
def test_parse_screen_response_extracts_json_and_coerces_fields():
    parsed = parse_screen_response(
        'analysis first {"rating":"buy","confidence":1.5,"thesis":"uses {braces} safely",'
        '"target_horizon":"20 sessions","full_graph_recommended":"false"}'
    )
    assert parsed["rating"] == "Buy"
    assert parsed["confidence"] == 1.0
    assert parsed["full_graph_recommended"] is False


@pytest.mark.unit
def test_parse_screen_response_rejects_invalid_rating():
    with pytest.raises(ValueError):
        parse_screen_response('{"rating":"Strong Buy","confidence":0.8}')


@pytest.mark.unit
def test_screen_provider_makes_one_call_per_packet(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    calls = []

    class _Response:
        content = json.dumps(
            {
                "rating": "Overweight",
                "confidence": 0.7,
                "thesis": "relative strength with controlled risk",
                "target_horizon": "20 sessions",
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
        packets=[_packet("AAPL"), _packet("MSFT")],
        output_dir=tmp_path,
        timeout_seconds=5,
    )

    assert len(calls) == 2
    assert result["status"] == "completed"
    assert result["decision_count"] == 2
    assert result["decisions"][0]["model"] == PROVIDER_DEFAULTS["openai-gpt-4o-mini"]["quick_model"]
    assert json.loads((tmp_path / "openai-gpt-4o-mini.json").read_text(encoding="utf-8"))["status"] == "completed"


@pytest.mark.unit
def test_openrouter_deepseek_screening_uses_throughput_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    created_kwargs = {}

    class _Response:
        content = json.dumps(
            {
                "rating": "Hold",
                "confidence": 0.5,
                "thesis": "balanced setup",
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
        packets=[_packet("AAPL")],
        output_dir=tmp_path,
        timeout_seconds=5,
    )

    assert result["status"] == "completed"
    assert created_kwargs["provider"] == "openrouter"
    assert created_kwargs["model"] == "deepseek/deepseek-v4-pro"
    assert created_kwargs["extra_body"] == {"provider": {"sort": "throughput"}}


@pytest.mark.unit
def test_merge_preserves_completed_and_timeout_provider_results(tmp_path):
    write_provider_status(tmp_path, "anthropic", "timeout", "provider process exceeded 1 seconds")
    (tmp_path / "google.json").write_text(
        json.dumps({"provider": "google", "status": "completed", "decision_count": 1, "error_count": 0, "decisions": [_packet()]}),
        encoding="utf-8",
    )

    merged = merge_provider_outputs(tmp_path, ["anthropic", "google", "openai-gpt-4o-mini"])

    assert merged["providers"]["anthropic"]["status"] == "timeout"
    assert merged["providers"]["google"]["decision_count"] == 1
    assert merged["providers"]["openai-gpt-4o-mini"]["status"] == "missing"
