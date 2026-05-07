"""Broker interfaces and shared order models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class BrokerRiskLimits:
    max_order_value: float = 100_000.0
    max_symbol_exposure: float = 250_000.0
    max_daily_loss: float = 10_000.0
    live_trading_enabled: bool = False
    require_manual_approval: bool = True


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    limit_price: float | None
    status: OrderStatus
    filled_price: float | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: int
    average_price: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price


class Broker(Protocol):
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: float | None = None,
        manual_approval: bool = False,
    ) -> BrokerOrder:
        ...

    def cancel_order(self, order_id: str) -> BrokerOrder:
        ...

    def positions(self) -> list[BrokerPosition]:
        ...

    def orders(self) -> list[BrokerOrder]:
        ...
