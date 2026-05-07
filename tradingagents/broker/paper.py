"""Paper broker with live-trading-style safety checks."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from .base import BrokerOrder, BrokerPosition, BrokerRiskLimits, OrderSide, OrderStatus


class PaperBroker:
    """In-memory paper broker for validating order flow before live trading."""

    def __init__(
        self,
        cash: float = 1_000_000.0,
        risk_limits: BrokerRiskLimits | None = None,
        prices: dict[str, float] | None = None,
        positions: list[BrokerPosition] | None = None,
    ):
        self.cash = cash
        self.risk_limits = risk_limits or BrokerRiskLimits()
        self._prices = {k.upper(): float(v) for k, v in (prices or {}).items()}
        self._positions: dict[str, BrokerPosition] = {
            p.symbol.upper(): p for p in (positions or [])
        }
        self._orders: dict[str, BrokerOrder] = {}
        self._daily_realized_pnl = 0.0

    def set_price(self, symbol: str, price: float) -> None:
        symbol = symbol.upper()
        self._prices[symbol] = float(price)
        if symbol in self._positions:
            self._positions[symbol] = replace(self._positions[symbol], last_price=float(price))

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: float | None = None,
        manual_approval: bool = False,
    ) -> BrokerOrder:
        symbol = symbol.upper()
        if quantity <= 0:
            return self._reject(symbol, side, quantity, limit_price, "quantity must be positive")
        if self.risk_limits.require_manual_approval and not manual_approval:
            return self._reject(symbol, side, quantity, limit_price, "manual approval required")

        price = float(limit_price if limit_price is not None else self._prices.get(symbol, 0.0))
        if price <= 0:
            return self._reject(symbol, side, quantity, limit_price, "no valid fill price")

        order_value = quantity * price
        if order_value > self.risk_limits.max_order_value:
            return self._reject(symbol, side, quantity, limit_price, "max order value exceeded")
        if self._daily_realized_pnl < -self.risk_limits.max_daily_loss:
            return self._reject(symbol, side, quantity, limit_price, "daily loss limit exceeded")

        if side is OrderSide.BUY:
            return self._fill_buy(symbol, quantity, price, limit_price)
        return self._fill_sell(symbol, quantity, price, limit_price)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        order = self._orders[order_id]
        if order.status is OrderStatus.FILLED:
            return order
        cancelled = replace(order, status=OrderStatus.CANCELLED)
        self._orders[order_id] = cancelled
        return cancelled

    def positions(self) -> list[BrokerPosition]:
        return list(self._positions.values())

    def orders(self) -> list[BrokerOrder]:
        return list(self._orders.values())

    def _fill_buy(self, symbol: str, quantity: int, price: float, limit_price: float | None) -> BrokerOrder:
        order_value = quantity * price
        if order_value > self.cash:
            return self._reject(symbol, OrderSide.BUY, quantity, limit_price, "insufficient cash")

        existing = self._positions.get(symbol)
        new_qty = quantity + (existing.quantity if existing else 0)
        new_value = order_value + (existing.quantity * existing.average_price if existing else 0.0)
        exposure = new_qty * price
        if exposure > self.risk_limits.max_symbol_exposure:
            return self._reject(symbol, OrderSide.BUY, quantity, limit_price, "max symbol exposure exceeded")

        self.cash -= order_value
        self._positions[symbol] = BrokerPosition(
            symbol=symbol,
            quantity=new_qty,
            average_price=new_value / new_qty,
            last_price=price,
        )
        return self._filled(symbol, OrderSide.BUY, quantity, limit_price, price)

    def _fill_sell(self, symbol: str, quantity: int, price: float, limit_price: float | None) -> BrokerOrder:
        existing = self._positions.get(symbol)
        if existing is None or existing.quantity < quantity:
            return self._reject(symbol, OrderSide.SELL, quantity, limit_price, "cash-equity shorting disabled")

        self.cash += quantity * price
        self._daily_realized_pnl += quantity * (price - existing.average_price)
        remaining = existing.quantity - quantity
        if remaining == 0:
            del self._positions[symbol]
        else:
            self._positions[symbol] = BrokerPosition(
                symbol=symbol,
                quantity=remaining,
                average_price=existing.average_price,
                last_price=price,
            )
        return self._filled(symbol, OrderSide.SELL, quantity, limit_price, price)

    def _filled(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: float | None,
        filled_price: float,
    ) -> BrokerOrder:
        order = BrokerOrder(
            order_id=str(uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            status=OrderStatus.FILLED,
            filled_price=filled_price,
        )
        self._orders[order.order_id] = order
        return order

    def _reject(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: float | None,
        reason: str,
    ) -> BrokerOrder:
        order = BrokerOrder(
            order_id=str(uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            status=OrderStatus.REJECTED,
            rejection_reason=reason,
        )
        self._orders[order.order_id] = order
        return order
