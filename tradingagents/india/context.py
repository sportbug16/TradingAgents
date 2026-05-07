"""Prompt context for Indian cash-equity analysis."""

from __future__ import annotations

from .market import IndiaMarketProfile, IndianInstrumentRegistry, is_indian_ticker


def build_indian_market_context(ticker: str, registry: IndianInstrumentRegistry | None = None) -> str:
    if not is_indian_ticker(ticker):
        return ""

    registry = registry or IndianInstrumentRegistry()
    instrument = registry.resolve(ticker)
    profile = IndiaMarketProfile(default_benchmark=instrument.benchmark)
    shorting = "not supported for overnight cash-equity positions"
    return (
        "Indian cash-equity market context: quote currency is INR; exchange is "
        f"{instrument.exchange.value}; normal market hours are {profile.market_open}-"
        f"{profile.market_close} {profile.timezone}; settlement is {profile.settlement_cycle}; "
        f"default benchmark is {instrument.benchmark}; overnight shorting is {shorting}. "
        "For short-horizon analysis, distinguish same-day noise from 1-5 session swing risk."
    )
