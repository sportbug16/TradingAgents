"""Date-indexed multi-ticker portfolio simulator with correlation caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from tradingagents.backtesting.investment import InvestmentHorizonSimulator
from tradingagents.backtesting.metrics import BacktestMetrics, calculate_metrics
from tradingagents.backtesting.rules import RatingPolicy, Signal, rating_to_target_weight
from tradingagents.backtesting.simulator import SimulatedTrade
from tradingagents.us.market import USTradingCalendar


@dataclass(frozen=True)
class PortfolioBacktestResult:
    trades: list[SimulatedTrade]
    metrics: BacktestMetrics
    equity_curve: list[dict]
    skipped_signals: list[dict]
    correlation_matrix: dict[str, dict[str, float]]


class CorrelationRiskModel:
    """Rolling close-return correlations for portfolio exposure checks."""

    def __init__(self, lookback_sessions: int = 120, threshold: float = 0.75):
        self.lookback_sessions = lookback_sessions
        self.threshold = threshold

    def matrix(self, prices_by_ticker: dict[str, pd.DataFrame], through_date: str | None = None) -> pd.DataFrame:
        returns = {}
        for ticker, prices in prices_by_ticker.items():
            frame = prices.copy()
            frame["Date"] = pd.to_datetime(frame["Date"])
            if through_date:
                frame = frame[frame["Date"] <= pd.Timestamp(through_date)]
            frame = frame.sort_values("Date").tail(self.lookback_sessions + 1)
            if len(frame) >= 3:
                returns[ticker] = frame.set_index("Date")["Close"].pct_change()
        if not returns:
            return pd.DataFrame()
        return pd.DataFrame(returns).corr().fillna(0.0)

    def cluster_weight(
        self,
        ticker: str,
        open_weights: dict[str, float],
        prices_by_ticker: dict[str, pd.DataFrame],
        through_date: str,
    ) -> float:
        corr = self.matrix(prices_by_ticker, through_date)
        if corr.empty or ticker not in corr:
            return 0.0
        total = 0.0
        for held_ticker, weight in open_weights.items():
            if held_ticker == ticker or corr.get(held_ticker, pd.Series()).get(ticker, 0.0) >= self.threshold:
                total += weight
        return total


class PortfolioBacktestSimulator:
    """Simulate simultaneous long-only positions on a daily equity curve."""

    def __init__(
        self,
        holding_sessions: int = 20,
        policy: RatingPolicy | None = None,
        calendar: USTradingCalendar | None = None,
        transaction_cost_bps: float = 1.0,
        slippage_bps: float = 2.0,
        max_gross_exposure: float = 1.0,
        max_single_name_exposure: float = 0.25,
        max_correlation_cluster_exposure: float = 0.50,
        correlation_lookback_sessions: int = 120,
        correlation_threshold: float = 0.75,
    ):
        self.holding_sessions = holding_sessions
        self.policy = policy or RatingPolicy(max_position_weight=max_single_name_exposure)
        self.max_gross_exposure = max_gross_exposure
        self.max_single_name_exposure = max_single_name_exposure
        self.max_correlation_cluster_exposure = max_correlation_cluster_exposure
        self.risk_model = CorrelationRiskModel(correlation_lookback_sessions, correlation_threshold)
        self.single_trade_simulator = InvestmentHorizonSimulator(
            holding_sessions=holding_sessions,
            policy=self.policy,
            calendar=calendar,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )

    def simulate(self, prices_by_ticker: dict[str, pd.DataFrame], signals: Sequence[Signal]) -> PortfolioBacktestResult:
        candidate_trades = []
        skipped = []
        for signal in sorted(signals, key=lambda s: (s.decision_date, s.ticker)):
            weight = min(
                rating_to_target_weight(signal.rating, self.policy),
                self.max_single_name_exposure,
            )
            if weight <= 0:
                continue
            result = self.single_trade_simulator.simulate(prices_by_ticker, [signal])
            if result.trades:
                trade = result.trades[0]
                candidate_trades.append(trade)
        accepted: list[SimulatedTrade] = []
        for trade in sorted(candidate_trades, key=lambda t: (t.entry_date, t.ticker)):
            open_weights = {
                other.ticker: other.target_weight
                for other in accepted
                if other.entry_date <= trade.entry_date < other.exit_date
            }
            gross = sum(open_weights.values())
            cluster_weight = self.risk_model.cluster_weight(
                trade.ticker,
                open_weights,
                prices_by_ticker,
                trade.decision_date,
            )
            if gross + trade.target_weight > self.max_gross_exposure:
                skipped.append({"ticker": trade.ticker, "decision_date": trade.decision_date, "reason": "max_gross_exposure"})
                continue
            if cluster_weight + trade.target_weight > self.max_correlation_cluster_exposure:
                skipped.append({"ticker": trade.ticker, "decision_date": trade.decision_date, "reason": "max_correlation_cluster_exposure"})
                continue
            accepted.append(trade)

        equity_curve = _build_equity_curve(accepted)
        metrics = calculate_metrics(
            [trade.net_return for trade in accepted],
            [trade.holding_sessions for trade in accepted],
            [trade.target_weight for trade in accepted],
        )
        corr = self.risk_model.matrix(prices_by_ticker).round(4).to_dict() if prices_by_ticker else {}
        return PortfolioBacktestResult(
            trades=accepted,
            metrics=metrics,
            equity_curve=equity_curve,
            skipped_signals=skipped,
            correlation_matrix=corr,
        )


def _build_equity_curve(trades: list[SimulatedTrade]) -> list[dict]:
    if not trades:
        return []
    dates = sorted({trade.entry_date for trade in trades} | {trade.exit_date for trade in trades})
    equity = 1.0
    curve = []
    for day in dates:
        for trade in [t for t in trades if t.exit_date == day]:
            equity *= 1.0 + trade.net_return
        exposure = sum(t.target_weight for t in trades if t.entry_date <= day < t.exit_date)
        curve.append({"date": day, "equity": equity, "gross_exposure": exposure})
    return curve
