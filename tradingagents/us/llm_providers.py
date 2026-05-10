"""US LLM provider labels used by batch comparison scripts."""

from __future__ import annotations


PROVIDER_DEFAULTS = {
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "llm_provider": "anthropic",
        "quick_model": "claude-opus-4-6",
        "deep_model": "claude-opus-4-6",
        "extra": {"anthropic_effort": "medium"},
        "wall_timeout_seconds": 30 * 60,
    },
    "google": {
        "key_env": "GOOGLE_API_KEY",
        "llm_provider": "google",
        "quick_model": "gemini-2.5-flash",
        "deep_model": "gemini-2.5-flash",
        "extra": {"google_thinking_level": "minimal"},
        "wall_timeout_seconds": 20 * 60,
    },
    "openai-gpt-4o-mini": {
        "key_env": "OPENAI_API_KEY",
        "llm_provider": "openai",
        "quick_model": "gpt-4o-mini",
        "deep_model": "gpt-4o-mini",
        "extra": {},
        "wall_timeout_seconds": 15 * 60,
    },
    "openai-gpt-5-4": {
        "key_env": "OPENAI_API_KEY",
        "llm_provider": "openai",
        "quick_model": "gpt-5.4",
        "deep_model": "gpt-5.4",
        "extra": {"openai_reasoning_effort": "none"},
        "wall_timeout_seconds": 30 * 60,
    },
    "openrouter-deepseek-v4": {
        "key_env": "OPENROUTER_API_KEY",
        "llm_provider": "openrouter",
        "quick_model": "deepseek/deepseek-v4-pro",
        "deep_model": "deepseek/deepseek-v4-pro",
        "extra": {"openrouter_provider_routing": {"sort": "throughput"}},
        "wall_timeout_seconds": 30 * 60,
    },
}


def provider_labels(value: str) -> list[str]:
    return [p.strip().lower() for p in value.split(",") if p.strip()]


def arg_prefix(provider: str) -> str:
    return provider.replace("-", "_")
