"""Indian market support for NSE/BSE equities."""

from .market import (
    DEFAULT_INDIAN_BENCHMARK,
    IndianExchange,
    IndianInstrument,
    IndianInstrumentRegistry,
    IndiaMarketProfile,
    infer_indian_exchange,
    is_indian_ticker,
    normalize_indian_ticker,
)
from .interfaces import (
    BenchmarkProvider,
    CorporateActionsProvider,
    FundamentalsProvider,
    InstrumentMasterProvider,
    IntradayOHLCVProvider,
    NewsProvider,
    OHLCVProvider,
)
from .data import (
    DataSnapshot,
    DhanHQProvider,
    FallbackIndianProvider,
    IndianDataFrame,
    IndianDataProviderUnavailable,
    YFinanceIndianProvider,
    create_indian_provider,
    dhan_intraday_chunks,
    snapshot_dict,
)
from .auth import (
    DhanAuthClient,
    DhanAuthError,
    DhanAuthResult,
    generate_totp,
)

__all__ = [
    "DEFAULT_INDIAN_BENCHMARK",
    "IndianExchange",
    "IndianInstrument",
    "IndianInstrumentRegistry",
    "IndiaMarketProfile",
    "BenchmarkProvider",
    "CorporateActionsProvider",
    "FundamentalsProvider",
    "InstrumentMasterProvider",
    "IntradayOHLCVProvider",
    "NewsProvider",
    "OHLCVProvider",
    "DataSnapshot",
    "DhanHQProvider",
    "FallbackIndianProvider",
    "IndianDataFrame",
    "IndianDataProviderUnavailable",
    "YFinanceIndianProvider",
    "create_indian_provider",
    "dhan_intraday_chunks",
    "snapshot_dict",
    "DhanAuthClient",
    "DhanAuthError",
    "DhanAuthResult",
    "generate_totp",
    "infer_indian_exchange",
    "is_indian_ticker",
    "normalize_indian_ticker",
]
