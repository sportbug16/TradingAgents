import json

import pytest

import scripts.india_llm_full_graph_shortlist as shortlist_script


@pytest.mark.unit
def test_india_full_graph_shortlist_sets_batch_and_india_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    configs = []

    class _Graph:
        def __init__(self, selected_analysts, config, debug):
            configs.append(config)

        def propagate(self, ticker, decision_date):
            return {"final_trade_decision": f"{ticker} final"}, "Overweight"

    monkeypatch.setattr(shortlist_script, "TradingAgentsGraph", _Graph)
    screen_path = tmp_path / "merged.json"
    screen_path.write_text(
        json.dumps(
            {
                "providers": {
                    "openai-gpt-4o-mini": {
                        "decisions": [
                            {
                                "ticker": "RELIANCE.NS",
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
            "india_llm_full_graph_shortlist.py",
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
    assert configs[0]["market_profile"] == "india"
    payload = json.loads(output.read_text(encoding="utf-8"))
    decision = payload["providers"]["openai-gpt-4o-mini"]["decisions"][0]
    assert decision["full_graph_rating"] == "Overweight"
    assert decision["final_trade_decision"] == "RELIANCE.NS final"
