from datetime import date

import pytest

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.india.market import (
    IndianExchange,
    IndianInstrumentRegistry,
    IndianTradingCalendar,
    infer_indian_exchange,
    is_indian_ticker,
    normalize_indian_ticker,
)


@pytest.mark.unit
class TestIndianTickerNormalization:
    def test_normalizes_exchange_aliases(self):
        assert normalize_indian_ticker("reliance") == "RELIANCE.NS"
        assert normalize_indian_ticker("reliance.nse") == "RELIANCE.NS"
        assert normalize_indian_ticker("reliance.bse") == "RELIANCE.BO"
        assert normalize_indian_ticker("500325.bo") == "500325.BO"

    def test_infers_indian_exchange_suffixes(self):
        assert infer_indian_exchange("TCS.NS") is IndianExchange.NSE
        assert infer_indian_exchange("TCS.BO") is IndianExchange.BSE
        assert infer_indian_exchange("AAPL") is None
        assert is_indian_ticker("INFY.NS")


@pytest.mark.unit
class TestIndianInstrumentRegistry:
    def test_resolves_nse_and_bse_duplicate_listing(self):
        registry = IndianInstrumentRegistry()
        nse = registry.resolve("RELIANCE.NS")
        bse = registry.resolve("RELIANCE.BO")
        assert nse.exchange is IndianExchange.NSE
        assert bse.exchange is IndianExchange.BSE
        assert nse.isin == bse.isin
        assert nse.data_symbol == "RELIANCE.NS"
        assert bse.data_symbol == "500325.BO"

    def test_benchmark_mapping(self):
        registry = IndianInstrumentRegistry()
        assert registry.benchmark_for("HDFCBANK.NS") == "^NSEBANK"
        assert registry.benchmark_for("RELIANCE.BO") == "^BSESN"

    def test_loads_registry_from_env_csv(self, tmp_path, monkeypatch):
        registry_csv = tmp_path / "india_registry.csv"
        registry_csv.write_text(
            "symbol,exchange,isin,name,yfinance_symbol,dhan_security_id,sector,benchmark,avg_daily_value_inr\n"
            "ABC,NSE,INE000000001,ABC Ltd,ABC.NS,123,Industrials,^NSEI,100000\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TRADINGAGENTS_INDIA_REGISTRY_CSV", str(registry_csv))
        instrument = IndianInstrumentRegistry().resolve("ABC.NS")
        assert instrument.dhan_security_id == "123"
        assert instrument.sector == "Industrials"

    def test_list_equities_can_filter_exchange(self):
        registry = IndianInstrumentRegistry()
        assert all(i.exchange is IndianExchange.NSE for i in registry.list_equities("NSE"))


@pytest.mark.unit
class TestIndianCalendar:
    def test_next_session_skips_weekend_and_holiday(self):
        calendar = IndianTradingCalendar(holidays=["2026-01-26"])
        assert calendar.next_session("2026-01-23") == date(2026, 1, 27)
        assert calendar.add_sessions("2026-01-23", 3) == date(2026, 1, 29)

    def test_loads_holidays_from_csv(self, tmp_path):
        holidays_csv = tmp_path / "holidays.csv"
        holidays_csv.write_text("date\n2026-01-26\n", encoding="utf-8")
        calendar = IndianTradingCalendar.from_csv(holidays_csv)
        assert calendar.next_session("2026-01-23") == date(2026, 1, 27)


@pytest.mark.unit
class TestIndianPromptContext:
    def test_context_mentions_indian_market_constraints(self):
        context = build_instrument_context("RELIANCE.NS")
        assert "INR" in context
        assert "Asia/Kolkata" in context
        assert "T+1" in context
        assert "overnight shorting" in context


@pytest.mark.unit
class TestReturnBenchmarkSelection:
    def test_indian_ticker_uses_indian_benchmark(self):
        graph = object.__new__(TradingAgentsGraph)
        graph.config = {"india": {"default_benchmark": "^NSEI"}}
        symbol, benchmark = TradingAgentsGraph._return_symbols(graph, "RELIANCE.NS")
        assert symbol == "RELIANCE.NS"
        assert benchmark == "^NSEI"

    def test_global_ticker_defaults_to_spy(self):
        graph = object.__new__(TradingAgentsGraph)
        graph.config = {}
        symbol, benchmark = TradingAgentsGraph._return_symbols(graph, "NVDA")
        assert symbol == "NVDA"
        assert benchmark == "SPY"
