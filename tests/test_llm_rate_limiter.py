import pytest

from tradingagents.llm_clients.rate_limiter import RollingLLMRateLimiter


@pytest.mark.unit
def test_rate_limiter_rejects_single_call_above_budget():
    limiter = RollingLLMRateLimiter(
        max_requests_per_minute=50,
        max_input_tokens_per_minute=30,
        safety_margin=1.0,
    )
    assert limiter.raise_error is True
    with pytest.raises(RuntimeError):
        limiter.on_llm_start({}, ["x" * 300])


@pytest.mark.unit
def test_rate_limiter_counts_chat_model_calls():
    limiter = RollingLLMRateLimiter(
        max_requests_per_minute=50,
        max_input_tokens_per_minute=30_000,
        safety_margin=1.0,
    )
    limiter.on_chat_model_start({}, [[type("Msg", (), {"content": "hello"})()]])
    snapshot = limiter.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.estimated_input_tokens > 0
