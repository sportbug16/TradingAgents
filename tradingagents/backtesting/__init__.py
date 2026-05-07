"""Backtesting and experiment utilities for TradingAgents."""

from .baselines import (
    BaselineResult,
    buy_and_hold_baseline,
    equal_weight_baseline,
    random_entry_baseline,
    simple_momentum_baseline,
)
from .experiments import BatchRunConfig, ExperimentRegistry, GraphBatchRunner
from .investment import InvestmentHorizonSimulator, WeekdayTradingCalendar
from .metrics import BacktestMetrics, calculate_metrics
from .portfolio import CorrelationRiskModel, PortfolioBacktestResult, PortfolioBacktestSimulator
from .rules import RatingPolicy, Signal, rating_to_target_weight
from .simulator import BacktestResult, SimulatedTrade, ShortHorizonSimulator

__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "BaselineResult",
    "BatchRunConfig",
    "ExperimentRegistry",
    "GraphBatchRunner",
    "InvestmentHorizonSimulator",
    "CorrelationRiskModel",
    "PortfolioBacktestResult",
    "PortfolioBacktestSimulator",
    "RatingPolicy",
    "Signal",
    "ShortHorizonSimulator",
    "SimulatedTrade",
    "WeekdayTradingCalendar",
    "calculate_metrics",
    "buy_and_hold_baseline",
    "equal_weight_baseline",
    "random_entry_baseline",
    "rating_to_target_weight",
    "simple_momentum_baseline",
]
