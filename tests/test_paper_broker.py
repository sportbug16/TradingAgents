import pytest

from tradingagents.broker import BrokerRiskLimits, OrderSide, OrderStatus, PaperBroker


@pytest.mark.unit
class TestPaperBroker:
    def test_requires_manual_approval_by_default(self):
        broker = PaperBroker(prices={"RELIANCE.NS": 100})
        order = broker.place_order("RELIANCE.NS", OrderSide.BUY, 10)
        assert order.status is OrderStatus.REJECTED
        assert "manual approval" in order.rejection_reason

    def test_buy_and_sell_updates_cash_and_position(self):
        broker = PaperBroker(cash=10_000, prices={"RELIANCE.NS": 100})
        buy = broker.place_order("RELIANCE.NS", OrderSide.BUY, 10, manual_approval=True)
        assert buy.status is OrderStatus.FILLED
        assert broker.cash == 9_000
        assert broker.positions()[0].quantity == 10

        broker.set_price("RELIANCE.NS", 110)
        sell = broker.place_order("RELIANCE.NS", OrderSide.SELL, 5, manual_approval=True)
        assert sell.status is OrderStatus.FILLED
        assert broker.positions()[0].quantity == 5
        assert broker.cash == 9_550

    def test_rejects_short_sell_for_cash_equity(self):
        broker = PaperBroker(prices={"RELIANCE.NS": 100})
        order = broker.place_order("RELIANCE.NS", OrderSide.SELL, 1, manual_approval=True)
        assert order.status is OrderStatus.REJECTED
        assert "shorting disabled" in order.rejection_reason

    def test_enforces_max_order_value(self):
        broker = PaperBroker(
            prices={"RELIANCE.NS": 100},
            risk_limits=BrokerRiskLimits(max_order_value=500, require_manual_approval=False),
        )
        order = broker.place_order("RELIANCE.NS", OrderSide.BUY, 10)
        assert order.status is OrderStatus.REJECTED
        assert "max order value" in order.rejection_reason
