import json

import pandas as pd
import pytest

import scripts.india_llm_backtest as india_llm_backtest
from tradingagents.india import llm_backtest
from tradingagents.india.data import DataSnapshot, IndianDataFrame


def _screen_results():
    return {
        "packet_manifest": {
            "manifests": [
                {"packet_count": 1, "config": {"horizon_sessions": 5}},
                {"packet_count": 1, "config": {"horizon_sessions": 20}},
            ]
        },
        "providers": {
            "openai-gpt-5-4": {
                "metadata": {"model": "gpt-5.4"},
                "decisions": [
                    {
                        "ticker": "RELIANCE.NS",
                        "decision_date": "2026-01-05",
                        "horizon_sessions": 5,
                        "rating": "Buy",
                        "confidence": 0.8,
                        "thesis": "entry",
                        "target_horizon": "5 sessions",
                        "model": "gpt-5.4",
                    }
                ],
            },
            "openrouter-deepseek-v4": {
                "metadata": {"model": "deepseek/deepseek-v4-pro"},
                "decisions": [
                    {
                        "ticker": "RELIANCE.NS",
                        "decision_date": "2026-01-05",
                        "rating": "Overweight",
                        "confidence": 0.7,
                        "thesis": "legacy horizon",
                        "target_horizon": "5 sessions",
                    }
                ],
            },
            "openrouter-openai-4o-mini": {
                "decisions": [
                    {
                        "ticker": "RELIANCE.NS",
                        "decision_date": "2026-01-05",
                        "horizon_sessions": 5,
                        "rating": "Buy",
                        "confidence": 0.9,
                    }
                ],
            },
        },
    }


def _prices(start=100, step=1, periods=80):
    rows = []
    price = start
    for day in pd.bdate_range("2025-12-01", periods=periods):
        rows.append(
            {
                "Date": day.date().isoformat(),
                "Open": price,
                "High": price + 1,
                "Low": price - 1,
                "Close": price,
                "Volume": 1000,
            }
        )
        price += step
    return pd.DataFrame(rows)


class _Provider:
    name = "fixture"

    def get_ohlcv(self, instrument, start, end):
        return IndianDataFrame(
            _prices(),
            DataSnapshot(
                provider="fixture",
                endpoint="bars",
                params={"ticker": instrument.data_symbol},
                adjusted=False,
                fetched_at="2026-01-01T00:00:00Z",
            ),
        )

    def get_benchmark_ohlcv(self, benchmark, start, end):
        return IndianDataFrame(
            _prices(300, 1),
            DataSnapshot(
                provider="fixture",
                endpoint="benchmark",
                params={"ticker": benchmark},
                adjusted=False,
                fetched_at="2026-01-01T00:00:00Z",
            ),
        )


@pytest.mark.unit
def test_extract_decisions_preserves_horizon_and_excludes_openrouter_4omini():
    providers = llm_backtest.provider_labels(None)
    decisions = llm_backtest.extract_decisions(_screen_results(), providers=providers, horizons=[5])

    assert {d.provider for d in decisions} == {"openai-gpt-5-4", "openrouter-deepseek-v4"}
    assert {d.horizon_sessions for d in decisions} == {5}
    assert all(d.provider != "openrouter-openai-4o-mini" for d in decisions)


@pytest.mark.unit
def test_build_signal_sets_includes_provider_and_ensemble_policies():
    decisions = llm_backtest.extract_decisions(_screen_results(), providers=llm_backtest.provider_labels(None), horizons=[5])
    signal_sets = llm_backtest.build_signal_sets(decisions, 5)

    assert "provider:openai-gpt-5-4" in signal_sets
    assert "provider:openrouter-deepseek-v4" in signal_sets
    assert signal_sets["ensemble:consensus"][0].rating in {"Buy", "Overweight"}
    assert signal_sets["ensemble:confidence_weighted"][0].rating in {"Buy", "Overweight"}
    assert signal_sets["ensemble:disagreement_filtered"][0].rating == "Hold"


@pytest.mark.unit
def test_india_llm_backtest_writes_expected_sections(monkeypatch, tmp_path):
    screen_path = tmp_path / "screen.json"
    screen_path.write_text(json.dumps(_screen_results()), encoding="utf-8")
    output = tmp_path / "llm_backtest.json"
    monkeypatch.setattr(india_llm_backtest, "create_indian_provider", lambda *args, **kwargs: _Provider())
    monkeypatch.setattr(
        "sys.argv",
        [
            "india_llm_backtest.py",
            "--screen-results",
            str(screen_path),
            "--horizons",
            "5",
            "--providers",
            "openai-gpt-5-4,openrouter-deepseek-v4,openrouter-openai-4o-mini",
            "--output",
            str(output),
        ],
    )

    india_llm_backtest.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["study_type"] == "india_llm_backtest"
    assert payload["provider_selection"]["included"] == ["openai-gpt-5-4", "openrouter-deepseek-v4"]
    assert "provider:openai-gpt-5-4" in payload["horizon_results"]["5"]
    assert "simple_momentum" in payload["baselines"]
