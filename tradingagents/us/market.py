"""US market metadata and universe validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


DEFAULT_US_BENCHMARK = "SPY"
US_TIMEZONE = "America/New_York"
DEFAULT_US_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "BRK-B",
    "LLY",
    "JPM",
    "WMT",
    "V",
    "ORCL",
    "XOM",
    "MA",
    "JNJ",
    "HD",
    "PG",
    "COST",
    "NFLX",
    "ABBV",
    "BAC",
    "KO",
    "AMD",
    "MRK",
    "CRM",
    "CSCO",
    "CVX",
    "WFC",
    "IBM",
    "MCD",
    "LIN",
    "ABT",
    "GE",
    "PM",
    "NOW",
    "ACN",
    "ISRG",
    "TMO",
    "DIS",
    "INTU",
    "PEP",
    "QCOM",
    "TXN",
]


@dataclass(frozen=True)
class USInstrument:
    symbol: str
    exchange: str
    name: str
    sector: str | None = None
    in_sp500: bool = True
    listed_on_nasdaq: bool = True
    benchmark: str = DEFAULT_US_BENCHMARK


@dataclass(frozen=True)
class USMarketProfile:
    timezone: str = US_TIMEZONE
    market_open: str = "09:30"
    market_close: str = "16:00"
    default_benchmark: str = DEFAULT_US_BENCHMARK
    allow_fractional_paper_orders: bool = False


_SEED = [
    USInstrument("AAPL", "NASDAQ", "Apple", "Information Technology"),
    USInstrument("MSFT", "NASDAQ", "Microsoft", "Information Technology"),
    USInstrument("NVDA", "NASDAQ", "NVIDIA", "Information Technology"),
    USInstrument("GOOG", "NASDAQ", "Alphabet Class C", "Communication Services"),
    USInstrument("GOOGL", "NASDAQ", "Alphabet Class A", "Communication Services"),
    USInstrument("AMZN", "NASDAQ", "Amazon", "Consumer Discretionary"),
    USInstrument("META", "NASDAQ", "Meta Platforms", "Communication Services"),
    USInstrument("AVGO", "NASDAQ", "Broadcom", "Information Technology"),
    USInstrument("TSLA", "NASDAQ", "Tesla", "Consumer Discretionary"),
    USInstrument("BRK-B", "NYSE", "Berkshire Hathaway Class B", "Financials", listed_on_nasdaq=False),
    USInstrument("LLY", "NYSE", "Eli Lilly", "Health Care", listed_on_nasdaq=False),
    USInstrument("JPM", "NYSE", "JPMorgan Chase", "Financials", listed_on_nasdaq=False),
    USInstrument("WMT", "NYSE", "Walmart", "Consumer Staples", listed_on_nasdaq=False),
    USInstrument("V", "NYSE", "Visa", "Financials", listed_on_nasdaq=False),
    USInstrument("ORCL", "NYSE", "Oracle", "Information Technology", listed_on_nasdaq=False),
    USInstrument("XOM", "NYSE", "Exxon Mobil", "Energy", listed_on_nasdaq=False),
    USInstrument("MA", "NYSE", "Mastercard", "Financials", listed_on_nasdaq=False),
    USInstrument("JNJ", "NYSE", "Johnson & Johnson", "Health Care", listed_on_nasdaq=False),
    USInstrument("HD", "NYSE", "Home Depot", "Consumer Discretionary", listed_on_nasdaq=False),
    USInstrument("PG", "NYSE", "Procter & Gamble", "Consumer Staples", listed_on_nasdaq=False),
    USInstrument("COST", "NASDAQ", "Costco", "Consumer Staples"),
    USInstrument("NFLX", "NASDAQ", "Netflix", "Communication Services"),
    USInstrument("ABBV", "NYSE", "AbbVie", "Health Care", listed_on_nasdaq=False),
    USInstrument("BAC", "NYSE", "Bank of America", "Financials", listed_on_nasdaq=False),
    USInstrument("KO", "NYSE", "Coca-Cola", "Consumer Staples", listed_on_nasdaq=False),
    USInstrument("AMD", "NASDAQ", "Advanced Micro Devices", "Information Technology"),
    USInstrument("MRK", "NYSE", "Merck", "Health Care", listed_on_nasdaq=False),
    USInstrument("CRM", "NYSE", "Salesforce", "Information Technology", listed_on_nasdaq=False),
    USInstrument("CSCO", "NASDAQ", "Cisco Systems", "Information Technology"),
    USInstrument("CVX", "NYSE", "Chevron", "Energy", listed_on_nasdaq=False),
    USInstrument("WFC", "NYSE", "Wells Fargo", "Financials", listed_on_nasdaq=False),
    USInstrument("IBM", "NYSE", "IBM", "Information Technology", listed_on_nasdaq=False),
    USInstrument("MCD", "NYSE", "McDonald's", "Consumer Discretionary", listed_on_nasdaq=False),
    USInstrument("LIN", "NASDAQ", "Linde", "Materials"),
    USInstrument("ABT", "NYSE", "Abbott Laboratories", "Health Care", listed_on_nasdaq=False),
    USInstrument("GE", "NYSE", "GE Aerospace", "Industrials", listed_on_nasdaq=False),
    USInstrument("PM", "NYSE", "Philip Morris International", "Consumer Staples", listed_on_nasdaq=False),
    USInstrument("NOW", "NYSE", "ServiceNow", "Information Technology", listed_on_nasdaq=False),
    USInstrument("ACN", "NYSE", "Accenture", "Information Technology", listed_on_nasdaq=False),
    USInstrument("ISRG", "NASDAQ", "Intuitive Surgical", "Health Care"),
    USInstrument("TMO", "NYSE", "Thermo Fisher Scientific", "Health Care", listed_on_nasdaq=False),
    USInstrument("DIS", "NYSE", "Walt Disney", "Communication Services", listed_on_nasdaq=False),
    USInstrument("INTU", "NASDAQ", "Intuit", "Information Technology"),
    USInstrument("PEP", "NASDAQ", "PepsiCo", "Consumer Staples"),
    USInstrument("QCOM", "NASDAQ", "Qualcomm", "Information Technology"),
    USInstrument("TXN", "NASDAQ", "Texas Instruments", "Information Technology"),
]


def normalize_us_ticker(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ticker must be a non-empty string")
    return value.strip().upper().replace(".", "-")


class USInstrumentRegistry:
    """Default registry for large-cap US S&P 500 names.

    The seed is intentionally curated and can be replaced with a current
    constituent/vendor universe later. Unknown symbols are rejected by default
    so paper runs stay inside the configured universe.
    """

    def __init__(self, instruments: list[USInstrument] | None = None):
        self._by_symbol = {i.symbol: i for i in (instruments or _SEED)}

    def resolve(self, ticker: str) -> USInstrument:
        symbol = normalize_us_ticker(ticker)
        if symbol not in self._by_symbol:
            raise ValueError(f"{symbol} is not in the configured Nasdaq/S&P 500 universe")
        return self._by_symbol[symbol]

    def validate_many(self, tickers: list[str]) -> list[str]:
        return [self.resolve(t).symbol for t in tickers]

    def default_symbols(self) -> list[str]:
        return list(self._by_symbol)


class USTradingCalendar:
    """US trading calendar backed by provider holiday rows when available."""

    def __init__(self, holidays: list[dict] | None = None):
        self.closed_dates = {
            row.get("date")
            for row in holidays or []
            if row.get("date") and row.get("status") == "closed" and row.get("exchange") in {None, "NYSE", "NASDAQ"}
        }

    def is_session(self, value: str | date | datetime) -> bool:
        day = _coerce_date(value)
        return day.weekday() < 5 and day.isoformat() not in self.closed_dates

    def next_session(self, value: str | date | datetime) -> date:
        day = _coerce_date(value) + timedelta(days=1)
        while not self.is_session(day):
            day += timedelta(days=1)
        return day

    def add_sessions(self, value: str | date | datetime, sessions: int) -> date:
        if sessions < 0:
            raise ValueError("sessions must be non-negative")
        day = _coerce_date(value)
        for _ in range(sessions):
            day = self.next_session(day)
        return day


def is_us_market_open(now: datetime | None = None) -> bool:
    current = now or datetime.now(ZoneInfo(US_TIMEZONE))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo(US_TIMEZONE))
    else:
        current = current.astimezone(ZoneInfo(US_TIMEZONE))
    if current.weekday() >= 5:
        return False
    return time(9, 30) <= current.time() <= time(16, 0)


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()
