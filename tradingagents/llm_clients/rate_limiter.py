"""Callback-based LLM rate limiting for provider comparison runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(frozen=True)
class RateLimitSnapshot:
    request_count: int
    estimated_input_tokens: int
    sleep_seconds: float
    max_requests_per_minute: int
    max_input_tokens_per_minute: int
    effective_requests_per_minute: int
    effective_input_tokens_per_minute: int
    peak_window_requests: int
    peak_window_input_tokens: int


class RollingLLMRateLimiter(BaseCallbackHandler):
    """Throttle chat-model calls using a rolling 60-second window.

    The limiter uses a conservative character-based input-token estimate. It
    does not count cache reads separately because provider cache accounting is
    not exposed before the call; keeping a safety margin avoids crossing the
    user-specified raw limit.
    """

    def __init__(
        self,
        max_requests_per_minute: int,
        max_input_tokens_per_minute: int,
        safety_margin: float = 0.9,
        window_seconds: float = 60.0,
    ) -> None:
        super().__init__()
        self.raise_error = True
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        if max_input_tokens_per_minute <= 0:
            raise ValueError("max_input_tokens_per_minute must be positive")
        if not 0 < safety_margin <= 1:
            raise ValueError("safety_margin must be in (0, 1]")
        self.max_requests = max(1, int(max_requests_per_minute * safety_margin))
        self.max_tokens = max(1, int(max_input_tokens_per_minute * safety_margin))
        self.raw_max_requests = max_requests_per_minute
        self.raw_max_tokens = max_input_tokens_per_minute
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self.sleep_seconds = 0.0
        self.request_count = 0
        self.estimated_input_tokens = 0
        self.peak_window_requests = 0
        self.peak_window_input_tokens = 0

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._reserve(sum(_estimate_tokens(p) for p in prompts))

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        estimated = sum(_estimate_messages(batch) for batch in messages)
        self._reserve(estimated)

    def snapshot(self) -> RateLimitSnapshot:
        with self._lock:
            return RateLimitSnapshot(
                request_count=self.request_count,
                estimated_input_tokens=self.estimated_input_tokens,
                sleep_seconds=self.sleep_seconds,
                max_requests_per_minute=self.raw_max_requests,
                max_input_tokens_per_minute=self.raw_max_tokens,
                effective_requests_per_minute=self.max_requests,
                effective_input_tokens_per_minute=self.max_tokens,
                peak_window_requests=self.peak_window_requests,
                peak_window_input_tokens=self.peak_window_input_tokens,
            )

    def _reserve(self, estimated_tokens: int) -> None:
        estimated_tokens = max(1, estimated_tokens)
        if estimated_tokens > self.max_tokens:
            raise RuntimeError(
                "single LLM call estimated at "
                f"{estimated_tokens} input tokens, above safe per-minute budget {self.max_tokens}"
            )

        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                used_requests = len(self._events)
                used_tokens = sum(tokens for _, tokens in self._events)
                if used_requests + 1 <= self.max_requests and used_tokens + estimated_tokens <= self.max_tokens:
                    self._events.append((now, estimated_tokens))
                    self.request_count += 1
                    self.estimated_input_tokens += estimated_tokens
                    self.peak_window_requests = max(self.peak_window_requests, len(self._events))
                    self.peak_window_input_tokens = max(
                        self.peak_window_input_tokens,
                        used_tokens + estimated_tokens,
                    )
                    return
                oldest_time = self._events[0][0] if self._events else now
                sleep_for = max(0.25, self.window_seconds - (now - oldest_time) + 0.05)
                self.sleep_seconds += sleep_for
            time.sleep(sleep_for)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()


def _estimate_messages(messages: list[Any]) -> int:
    return sum(_estimate_tokens(_message_to_text(m)) for m in messages)


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str, ensure_ascii=False)
    except TypeError:
        return str(content)


def _estimate_tokens(text: str) -> int:
    # Conservative for English financial prose and CSV snippets.
    return max(1, (len(text) + 2) // 3)
