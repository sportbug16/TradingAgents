import pytest

from tradingagents.us.shortlist import select_shortlist


def _decision(provider, ticker, rating, confidence=0.5):
    return {
        "provider": provider,
        "ticker": ticker,
        "decision_date": "2026-02-02",
        "rating": rating,
        "confidence": confidence,
    }


@pytest.mark.unit
def test_select_shortlist_includes_entry_signals_and_disagreement():
    screen_results = {
        "providers": {
            "anthropic": {
                "decisions": [
                    _decision("anthropic", "AAPL", "Buy", 0.8),
                    _decision("anthropic", "MSFT", "Hold", 0.4),
                    _decision("anthropic", "NVDA", "Buy", 0.9),
                ]
            },
            "google": {
                "decisions": [
                    _decision("google", "AAPL", "Hold", 0.7),
                    _decision("google", "MSFT", "Hold", 0.7),
                    _decision("google", "NVDA", "Sell", 0.6),
                ]
            },
            "openrouter-openai-4o-mini": {
                "decisions": [
                    _decision("openrouter-openai-4o-mini", "AAPL", "Overweight", 0.6),
                    _decision("openrouter-openai-4o-mini", "MSFT", "Hold", 0.9),
                    _decision("openrouter-openai-4o-mini", "NVDA", "Hold", 0.5),
                ]
            },
        }
    }

    jobs = select_shortlist(screen_results, max_jobs=10)

    assert any(job["ticker"] == "AAPL" and job["reason"] == "entry_signal" for job in jobs)
    assert any(job["ticker"] == "NVDA" and job["reason"] == "disagreement" for job in jobs)
    assert all(job["ticker"] != "MSFT" for job in jobs)


@pytest.mark.unit
def test_select_shortlist_respects_max_jobs_ordering():
    screen_results = {
        "providers": {
            "anthropic": {
                "decisions": [
                    _decision("anthropic", "AAPL", "Buy", 0.3),
                    _decision("anthropic", "MSFT", "Buy", 0.9),
                ]
            }
        }
    }

    jobs = select_shortlist(screen_results, max_jobs=1)

    assert jobs == [
        {
            "ticker": "MSFT",
            "decision_date": "2026-02-02",
            "provider": "anthropic",
            "rating": "Buy",
            "confidence": 0.9,
            "reason": "entry_signal",
        }
    ]
