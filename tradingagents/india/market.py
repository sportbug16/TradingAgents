"""Indian exchange metadata, symbol normalization, and market calendar helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
import csv
import io
import os
from pathlib import Path
from typing import Iterable


DEFAULT_INDIAN_BENCHMARK = "^NSEI"
INDIA_TIMEZONE = "Asia/Kolkata"
INDIA_MARKET_OPEN = "09:15"
INDIA_MARKET_CLOSE = "15:30"


class IndianExchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"

    @property
    def yfinance_suffix(self) -> str:
        return ".NS" if self is IndianExchange.NSE else ".BO"

    @property
    def dhan_segment(self) -> str:
        return "NSE_EQ" if self is IndianExchange.NSE else "BSE_EQ"


@dataclass(frozen=True)
class IndianInstrument:
    """Canonical representation of a listed Indian cash-equity instrument."""

    symbol: str
    exchange: IndianExchange
    isin: str | None = None
    name: str | None = None
    yfinance_symbol: str | None = None
    dhan_security_id: str | None = None
    tick_size: float = 0.05
    lot_size: int = 1
    sector: str | None = None
    benchmark: str = DEFAULT_INDIAN_BENCHMARK
    avg_daily_value_inr: float | None = None
    active: bool = True

    @property
    def qualified_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"

    @property
    def data_symbol(self) -> str:
        return self.yfinance_symbol or f"{self.symbol}{self.exchange.yfinance_suffix}"


@dataclass(frozen=True)
class IndiaMarketProfile:
    """Trading assumptions for Indian cash equities."""

    timezone: str = INDIA_TIMEZONE
    market_open: str = INDIA_MARKET_OPEN
    market_close: str = INDIA_MARKET_CLOSE
    settlement_cycle: str = "T+1"
    default_benchmark: str = DEFAULT_INDIAN_BENCHMARK
    allow_cash_equity_shorting: bool = False


_SEED_CSV = """symbol,exchange,isin,name,yfinance_symbol,dhan_security_id,sector,benchmark,avg_daily_value_inr
RELIANCE,NSE,INE002A01018,Reliance Industries,RELIANCE.NS,2885,Energy,^NSEI,10000000000
RELIANCE,BSE,INE002A01018,Reliance Industries,500325.BO,,Energy,^BSESN,1000000000
TCS,NSE,INE467B01029,Tata Consultancy Services,TCS.NS,11536,Information Technology,^NSEI,4000000000
TCS,BSE,INE467B01029,Tata Consultancy Services,532540.BO,,Information Technology,^BSESN,700000000
INFY,NSE,INE009A01021,Infosys,INFY.NS,1594,Information Technology,^NSEI,4500000000
INFY,BSE,INE009A01021,Infosys,500209.BO,,Information Technology,^BSESN,800000000
HDFCBANK,NSE,INE040A01034,HDFC Bank,HDFCBANK.NS,1333,Financial Services,^NSEBANK,6000000000
HDFCBANK,BSE,INE040A01034,HDFC Bank,500180.BO,,Financial Services,^BSESN,900000000
SBIN,NSE,INE062A01020,State Bank of India,SBIN.NS,3045,Financial Services,^NSEBANK,5000000000
SBIN,BSE,INE062A01020,State Bank of India,500112.BO,,Financial Services,^BSESN,700000000
"""


def _parse_exchange(value: str) -> IndianExchange:
    try:
        return IndianExchange(value.strip().upper())
    except ValueError as exc:
        raise ValueError(f"unsupported Indian exchange: {value!r}") from exc


def normalize_indian_ticker(value: str, default_exchange: IndianExchange = IndianExchange.NSE) -> str:
    """Normalize Indian cash-equity ticker input while preserving exchange choice.

    Accepted examples:
    - ``RELIANCE`` -> ``RELIANCE.NS`` by default
    - ``RELIANCE.NSE`` -> ``RELIANCE.NS``
    - ``RELIANCE.BSE`` -> ``RELIANCE.BO``
    - ``500325.BO`` remains unchanged for BSE scrip-code style symbols
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ticker must be a non-empty string")

    ticker = value.strip().upper()
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    if ticker.endswith(".NSE"):
        return f"{ticker[:-4]}.NS"
    if ticker.endswith(".BSE"):
        return f"{ticker[:-4]}.BO"
    if "." in ticker:
        return ticker
    return f"{ticker}{default_exchange.yfinance_suffix}"


def infer_indian_exchange(ticker: str) -> IndianExchange | None:
    ticker = ticker.strip().upper()
    if ticker.endswith(".NS") or ticker.endswith(".NSE"):
        return IndianExchange.NSE
    if ticker.endswith(".BO") or ticker.endswith(".BSE"):
        return IndianExchange.BSE
    return None


def is_indian_ticker(ticker: str) -> bool:
    return infer_indian_exchange(ticker) is not None


def _base_symbol(ticker: str) -> str:
    ticker = ticker.strip().upper()
    for suffix in (".NS", ".BO", ".NSE", ".BSE"):
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


class IndianInstrumentRegistry:
    """Registry resolving NSE/BSE listings and vendor IDs.

    The built-in seed covers common liquid names for tests and examples. For
    real research, pass a daily instrument dump exported from a paid vendor or
    exchange master file with the same column names as ``_SEED_CSV``.
    """

    def __init__(self, instruments: Iterable[IndianInstrument] | None = None):
        self._by_qualified: dict[str, IndianInstrument] = {}
        self._by_yfinance: dict[str, IndianInstrument] = {}
        self._by_isin_exchange: dict[tuple[str, IndianExchange], IndianInstrument] = {}
        source = instruments
        if source is None:
            registry_csv = os.getenv("TRADINGAGENTS_INDIA_REGISTRY_CSV")
            source = self._load_csv(registry_csv) if registry_csv else self._load_seed()
        for instrument in source:
            self.add(instrument)

    @classmethod
    def from_csv(cls, path: str | Path) -> "IndianInstrumentRegistry":
        return cls(cls._load_csv(path))

    @staticmethod
    def _load_csv(path: str | Path) -> list[IndianInstrument]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(_rows_to_instruments(csv.DictReader(f)))

    @staticmethod
    def _load_seed() -> list[IndianInstrument]:
        return list(_rows_to_instruments(csv.DictReader(io.StringIO(_SEED_CSV))))

    def add(self, instrument: IndianInstrument) -> None:
        self._by_qualified[instrument.qualified_symbol] = instrument
        self._by_yfinance[instrument.data_symbol.upper()] = instrument
        if instrument.isin:
            self._by_isin_exchange[(instrument.isin.upper(), instrument.exchange)] = instrument

    def resolve(
        self,
        ticker: str,
        default_exchange: IndianExchange | None = None,
    ) -> IndianInstrument:
        normalized = normalize_indian_ticker(
            ticker,
            default_exchange or IndianExchange.NSE,
        )
        if normalized in self._by_yfinance:
            return self._by_yfinance[normalized]

        exchange = infer_indian_exchange(normalized) or default_exchange or IndianExchange.NSE
        symbol = _base_symbol(normalized)
        qualified = f"{symbol}.{exchange.value}"
        if qualified in self._by_qualified:
            return self._by_qualified[qualified]

        return IndianInstrument(
            symbol=symbol,
            exchange=exchange,
            yfinance_symbol=normalized,
            benchmark=DEFAULT_INDIAN_BENCHMARK if exchange is IndianExchange.NSE else "^BSESN",
        )

    def resolve_isin(self, isin: str, exchange: IndianExchange) -> IndianInstrument | None:
        return self._by_isin_exchange.get((isin.upper(), exchange))

    def benchmark_for(self, ticker: str, default: str | None = None) -> str:
        try:
            return self.resolve(ticker).benchmark
        except ValueError:
            return default or DEFAULT_INDIAN_BENCHMARK

    def list_equities(self, exchange: IndianExchange | str | None = None) -> list[IndianInstrument]:
        if exchange is None:
            return list(self._by_qualified.values())
        parsed = _parse_exchange(exchange.value if isinstance(exchange, IndianExchange) else exchange)
        return [instrument for instrument in self._by_qualified.values() if instrument.exchange is parsed]


def _rows_to_instruments(rows: Iterable[dict]) -> Iterable[IndianInstrument]:
    for row in rows:
        if not row.get("symbol") or not row.get("exchange"):
            continue
        exchange = _parse_exchange(row["exchange"])
        avg_value = row.get("avg_daily_value_inr")
        tick_size = row.get("tick_size")
        lot_size = row.get("lot_size")
        active = row.get("active")
        yield IndianInstrument(
            symbol=row["symbol"].strip().upper(),
            exchange=exchange,
            isin=(row.get("isin") or "").strip().upper() or None,
            name=(row.get("name") or "").strip() or None,
            yfinance_symbol=(row.get("yfinance_symbol") or "").strip().upper()
            or None,
            dhan_security_id=(row.get("dhan_security_id") or "").strip() or None,
            tick_size=float(tick_size) if tick_size else 0.05,
            lot_size=int(float(lot_size)) if lot_size else 1,
            sector=(row.get("sector") or "").strip() or None,
            benchmark=(row.get("benchmark") or "").strip() or DEFAULT_INDIAN_BENCHMARK,
            avg_daily_value_inr=float(avg_value) if avg_value else None,
            active=str(active).strip().lower() not in {"0", "false", "no", "n"} if active is not None else True,
        )


class IndianTradingCalendar:
    """Small trading-calendar helper with injectable holidays.

    Weekends are always closed. Holiday data should be supplied by the caller
    for rigorous backtests because NSE/BSE publish the definitive list each
    year and ad-hoc market closures can happen.
    """

    def __init__(self, holidays: Iterable[str | date] | None = None):
        self.holidays = {_coerce_date(d) for d in holidays or ()}

    @classmethod
    def from_csv(cls, path: str | Path) -> "IndianTradingCalendar":
        holidays: list[str] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    value = row.get("date") or row.get("Date") or next(iter(row.values()), "")
                    if value:
                        holidays.append(value)
            else:
                f.seek(0)
                for row in csv.reader(f):
                    if row and row[0].strip().lower() != "date":
                        holidays.append(row[0])
        return cls(holidays)

    def is_session(self, value: str | date | datetime) -> bool:
        day = _coerce_date(value)
        return day.weekday() < 5 and day not in self.holidays

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


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()
