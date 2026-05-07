"""Explicit Indian-market data interfaces.

These protocols define the boundaries needed for paid data integrations. They
let Dhan, TrueData, Global Datafeeds, broker APIs, or offline fixtures be
swapped without changing graph or backtest code.
"""

from __future__ import annotations

from typing import Protocol

from .data import IndianDataFrame
from .market import IndianInstrument


class InstrumentMasterProvider(Protocol):
    def resolve(self, ticker: str) -> IndianInstrument:
        ...

    def list_equities(self, exchange: str | None = None) -> list[IndianInstrument]:
        ...


class OHLCVProvider(Protocol):
    def get_ohlcv(self, instrument: IndianInstrument, start_date: str, end_date: str) -> IndianDataFrame:
        ...


class IntradayOHLCVProvider(Protocol):
    def get_intraday_ohlcv(
        self,
        instrument: IndianInstrument,
        start_datetime: str,
        end_datetime: str,
        interval_minutes: int = 5,
    ) -> IndianDataFrame:
        ...


class FundamentalsProvider(Protocol):
    def get_fundamentals(self, instrument: IndianInstrument, as_of_date: str) -> dict:
        ...


class CorporateActionsProvider(Protocol):
    def get_corporate_actions(self, instrument: IndianInstrument, start_date: str, end_date: str):
        ...


class NewsProvider(Protocol):
    def get_news(self, instrument: IndianInstrument, start_date: str, end_date: str) -> list[dict]:
        ...


class BenchmarkProvider(Protocol):
    def get_benchmark_ohlcv(self, benchmark: str, start_date: str, end_date: str) -> IndianDataFrame:
        ...
