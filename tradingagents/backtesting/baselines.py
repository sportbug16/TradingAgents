"""Baseline strategies for comparative Indian equity studies."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class BaselineResult:
    name: str
    returns: list[float]

    @property
    def total_return(self) -> float:
        equity = 1.0
        for ret in self.returns:
            equity *= 1.0 + ret
        return equity - 1.0


def buy_and_hold_baseline(name: str, prices: pd.DataFrame) -> BaselineResult:
    frame = _prepare(prices)
    if len(frame) < 2:
        return BaselineResult(name=name, returns=[])
    ret = float(frame["Close"].iloc[-1] / frame["Open"].iloc[0] - 1.0)
    return BaselineResult(name=name, returns=[ret])


def equal_weight_baseline(prices_by_ticker: dict[str, pd.DataFrame]) -> BaselineResult:
    returns = []
    for prices in prices_by_ticker.values():
        result = buy_and_hold_baseline("component", prices)
        if result.returns:
            returns.append(result.returns[0])
    if not returns:
        return BaselineResult(name="equal_weight", returns=[])
    return BaselineResult(name="equal_weight", returns=[sum(returns) / len(returns)])


def simple_momentum_baseline(
    prices_by_ticker: dict[str, pd.DataFrame],
    lookback_sessions: int = 20,
    hold_sessions: int = 5,
    top_n: int = 10,
) -> BaselineResult:
    scores = []
    for ticker, prices in prices_by_ticker.items():
        frame = _prepare(prices)
        exit_idx = lookback_sessions + 1 + hold_sessions
        if len(frame) <= exit_idx:
            continue
        score = frame["Close"].iloc[lookback_sessions] / frame["Close"].iloc[0] - 1.0
        forward = frame["Close"].iloc[exit_idx] / frame["Open"].iloc[lookback_sessions + 1] - 1.0
        scores.append((score, forward, ticker))
    selected = sorted(scores, reverse=True)[:top_n]
    if not selected:
        return BaselineResult(name="simple_momentum", returns=[])
    return BaselineResult(
        name="simple_momentum",
        returns=[sum(forward for _, forward, _ in selected) / len(selected)],
    )


def low_volatility_baseline(
    prices_by_ticker: dict[str, pd.DataFrame],
    lookback_sessions: int = 60,
    hold_sessions: int = 20,
    top_n: int = 10,
) -> BaselineResult:
    scores = []
    for ticker, prices in prices_by_ticker.items():
        frame = _prepare(prices)
        exit_idx = lookback_sessions + 1 + hold_sessions
        if len(frame) <= exit_idx:
            continue
        volatility = frame["Close"].pct_change().iloc[1 : lookback_sessions + 1].std()
        if pd.isna(volatility):
            continue
        forward = frame["Close"].iloc[exit_idx] / frame["Open"].iloc[lookback_sessions + 1] - 1.0
        scores.append((float(volatility), forward, ticker))
    selected = sorted(scores)[:top_n]
    if not selected:
        return BaselineResult(name="low_volatility", returns=[])
    return BaselineResult(
        name="low_volatility",
        returns=[sum(forward for _, forward, _ in selected) / len(selected)],
    )


def random_entry_baseline(
    prices_by_ticker: dict[str, pd.DataFrame],
    hold_sessions: int = 3,
    sample_size: int = 100,
    seed: int = 42,
) -> BaselineResult:
    rng = random.Random(seed)
    candidates = []
    for ticker, prices in prices_by_ticker.items():
        frame = _prepare(prices)
        for idx in range(0, max(len(frame) - hold_sessions - 1, 0)):
            entry = frame["Open"].iloc[idx + 1]
            exit_ = frame["Open"].iloc[idx + 1 + hold_sessions]
            candidates.append((ticker, float(exit_ / entry - 1.0)))
    if not candidates:
        return BaselineResult(name="random_entry", returns=[])
    sample = rng.sample(candidates, min(sample_size, len(candidates)))
    return BaselineResult(name="random_entry", returns=[ret for _, ret in sample])


def _prepare(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date").reset_index(drop=True)
