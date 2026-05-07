import json

import pytest

import scripts.us_llm_full_graph_shortlist as shortlist_script


@pytest.mark.unit
def test_full_graph_shortlist_passes_batch_flags_and_records_result(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    configs = []

    class _Graph:
        def __init__(self, selected_analysts, config, debug):
            configs.append(config)

        def propagate(self, ticker, decision_date):
            return {"final_trade_decision": f"{ticker} decision"}, "Buy"

    monkeypatch.setattr(shortlist_script, "TradingAgentsGraph", _Graph)
    screen_path = tmp_path / "merged.json"
    screen_path.write_text(
        json.dumps(
            {
                "providers": {
                    "openrouter-openai-4o-mini": {
                        "decisions": [
                            {
                                "ticker": "AAPL",
                                "decision_date": "2026-02-02",
                                "rating": "Buy",
                                "confidence": 0.8,
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "full.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "us_llm_full_graph_shortlist.py",
            "--screen-results",
            str(screen_path),
            "--max-jobs",
            "1",
            "--output",
            str(output),
        ],
    )

    shortlist_script.main()

    assert configs[0]["batch_mode"] is True
    assert configs[0]["disable_memory_reflection"] is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    decision = payload["providers"]["openrouter-openai-4o-mini"]["decisions"][0]
    assert decision["full_graph_rating"] == "Buy"
    assert decision["final_trade_decision"] == "AAPL decision"


@pytest.mark.unit
def test_openrouter_screening_labels_are_restricted():
    defaults = shortlist_script.PROVIDER_DEFAULTS
    assert defaults["openrouter-openai-4o-mini"]["quick_model"] == "openai/gpt-4o-mini"
    assert defaults["openrouter-deepseek-v4"]["quick_model"] == "deepseek/deepseek-v4-pro"
