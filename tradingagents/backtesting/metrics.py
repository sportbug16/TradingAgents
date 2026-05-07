"""Portfolio and trade metrics for short-horizon studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    average_win: float
    average_loss: float
    turnover: float
    exposure: float
    trade_count: int


def calculate_metrics(
    trade_returns: Iterable[float],
    holding_days: Iterable[int],
    target_weights: Iterable[float],
    trading_days_per_year: int = 252,
) -> BacktestMetrics:
    returns = list(trade_returns)
    days = list(holding_days)
    weights = list(target_weights)
    if not returns:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)

    total = equity - 1.0
    total_days = max(sum(days), 1)
    annualized = (1.0 + total) ** (trading_days_per_year / total_days) - 1.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    downside = [min(r, 0.0) for r in returns]
    downside_var = sum(r * r for r in downside) / len(returns)
    scale = math.sqrt(trading_days_per_year / max(sum(days) / len(days), 1))
    sharpe = (mean / math.sqrt(variance) * scale) if variance > 0 else 0.0
    sortino = (mean / math.sqrt(downside_var) * scale) if downside_var > 0 else 0.0

    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r < 0]
    turnover = sum(abs(w) for w in weights)
    exposure = sum(1 for w in weights if w > 0) / len(weights)
    return BacktestMetrics(
        total_return=total,
        annualized_return=annualized,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=len(winners) / len(returns),
        average_win=sum(winners) / len(winners) if winners else 0.0,
        average_loss=sum(losers) / len(losers) if losers else 0.0,
        turnover=turnover,
        exposure=exposure,
        trade_count=len(returns),
    )
