"""Short-horizon long-only simulator for TradingAgents ratings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from tradingagents.india.market import IndianTradingCalendar

from .metrics import BacktestMetrics, calculate_metrics
from .rules import RatingPolicy, Signal, rating_to_target_weight


@dataclass(frozen=True)
class SimulatedTrade:
    ticker: str
    decision_date: str
    entry_date: str
    exit_date: str
    rating: str
    target_weight: float
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    holding_sessions: int
    run_id: str | None = None


@dataclass(frozen=True)
class BacktestResult:
    trades: list[SimulatedTrade]
    metrics: BacktestMetrics


class ShortHorizonSimulator:
    """Simulate EOD signals entered next session and exited after N sessions."""

    def __init__(
        self,
        holding_sessions: int = 3,
        policy: RatingPolicy | None = None,
        calendar: IndianTradingCalendar | None = None,
        transaction_cost_bps: float = 12.0,
        slippage_bps: float = 5.0,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ):
        if holding_sessions not in {1, 3, 5}:
            raise ValueError("v1 supports 1, 3, or 5 holding sessions")
        self.holding_sessions = holding_sessions
        self.policy = policy or RatingPolicy()
        self.calendar = calendar or IndianTradingCalendar()
        self.round_trip_cost = (transaction_cost_bps + slippage_bps) * 2 / 10000
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def simulate(self, prices_by_ticker: dict[str, pd.DataFrame], signals: Sequence[Signal]) -> BacktestResult:
        trades: list[SimulatedTrade] = []
        for signal in signals:
            weight = rating_to_target_weight(signal.rating, self.policy)
            if weight <= 0:
                continue
            prices = _prepare_prices(prices_by_ticker[signal.ticker])
            trade = self._simulate_signal(signal, prices, weight)
            if trade:
                trades.append(trade)

        metrics = calculate_metrics(
            [t.net_return for t in trades],
            [t.holding_sessions for t in trades],
            [t.target_weight for t in trades],
        )
        return BacktestResult(trades=trades, metrics=metrics)

    def _simulate_signal(self, signal: Signal, prices: pd.DataFrame, weight: float) -> SimulatedTrade | None:
        entry_day = self.calendar.next_session(signal.decision_date)
        exit_day = self.calendar.add_sessions(entry_day, self.holding_sessions)

        entry_row = _row_on_or_after(prices, entry_day)
        if entry_row is None:
            return None

        exit_row = _row_on_or_after(prices, exit_day)
        actual_sessions = self.holding_sessions
        if exit_row is None:
            return None

        entry_price = float(entry_row["Open"])
        exit_price = float(exit_row["Open"])

        path = prices[(prices["Date"] >= entry_row["Date"]) & (prices["Date"] <= exit_row["Date"])]
        if self.stop_loss_pct is not None:
            stop_price = entry_price * (1.0 - self.stop_loss_pct)
            stop_hits = path[path["Low"] <= stop_price]
            if not stop_hits.empty:
                exit_row = stop_hits.iloc[0]
                exit_price = stop_price
                actual_sessions = max(1, len(path[path["Date"] <= exit_row["Date"]]) - 1)
        if self.take_profit_pct is not None:
            target_price = entry_price * (1.0 + self.take_profit_pct)
            target_hits = path[path["High"] >= target_price]
            if not target_hits.empty and target_hits.iloc[0]["Date"] <= exit_row["Date"]:
                exit_row = target_hits.iloc[0]
                exit_price = target_price
                actual_sessions = max(1, len(path[path["Date"] <= exit_row["Date"]]) - 1)

        gross = (exit_price - entry_price) / entry_price
        net = weight * gross - self.round_trip_cost
        return SimulatedTrade(
            ticker=signal.ticker,
            decision_date=signal.decision_date,
            entry_date=str(pd.Timestamp(entry_row["Date"]).date()),
            exit_date=str(pd.Timestamp(exit_row["Date"]).date()),
            rating=signal.rating,
            target_weight=weight,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_return=gross,
            net_return=net,
            holding_sessions=actual_sessions,
            run_id=signal.run_id,
        )


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required price columns: {sorted(missing)}")
    return frame.sort_values("Date").reset_index(drop=True)


def _row_on_or_after(prices: pd.DataFrame, day) -> pd.Series | None:
    cutoff = pd.Timestamp(day)
    rows = prices[prices["Date"] >= cutoff]
    if rows.empty:
        return None
    return rows.iloc[0]
