import json

import pandas as pd
import pytest

from tradingagents.india.data import DataSnapshot, IndianDataFrame
from tradingagents.india.packets import generate_packets


def _prices(start=100, step=1):
    rows = []
    price = start
    for day in pd.bdate_range("2026-01-01", periods=70):
        rows.append(
            {
                "Date": day.date().isoformat(),
                "Open": price,
                "High": price + 1,
                "Low": price - 1,
                "Close": price + step,
                "Volume": 1000,
            }
        )
        price += step
    return pd.DataFrame(rows)


class _Provider:
    name = "fixture"

    def get_ohlcv(self, instrument, start, end):
        return IndianDataFrame(
            _prices(),
            DataSnapshot(
                provider="fixture",
                endpoint="bars",
                params={"ticker": instrument.data_symbol, "start": start, "end": end},
                adjusted=False,
                fetched_at="2026-01-01T00:00:00Z",
                cache_key=f"{instrument.data_symbol}-bars",
            ),
        )

    def get_benchmark_ohlcv(self, benchmark, start, end):
        return IndianDataFrame(
            _prices(300),
            DataSnapshot(
                provider="yfinance",
                endpoint="benchmark",
                params={"ticker": benchmark, "start": start, "end": end},
                adjusted=True,
                fetched_at="2026-01-01T00:00:00Z",
                fallback_unofficial=True,
                cache_key=f"{benchmark}-bars",
            ),
        )


@pytest.mark.unit
def test_generate_india_packets_has_snapshot_quality_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.india.packets.create_indian_provider", lambda *args, **kwargs: _Provider())
    first = generate_packets(
        tickers=["RELIANCE.NS", "TCS.NS"],
        dates=["2026-02-02"],
        output_dir=tmp_path / "packets",
        price_start="2026-01-01",
        price_end="2026-04-15",
        data_provider_name="dhan",
        allow_fallback=False,
    )

    packet = first["packets"][0]
    assert packet["ticker"] == "RELIANCE.NS"
    assert packet["data_quality"]["provider"] == "fixture"
    assert packet["data_quality"]["benchmark_fallback_unofficial"] is True
    assert packet["price_features"]["return_20d"] is not None
    assert packet["correlation_features"]["top_correlations"]

    packet_path = tmp_path / "packets" / f"RELIANCE.NS_2026-02-02_{first['manifest']['config_hash']}.json"
    assert json.loads(packet_path.read_text(encoding="utf-8"))["config_hash"] == first["manifest"]["config_hash"]

    monkeypatch.setattr(
        "tradingagents.india.packets.create_indian_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packet cache should skip provider")),
    )
    cached = generate_packets(
        tickers=["RELIANCE.NS", "TCS.NS"],
        dates=["2026-02-02"],
        output_dir=tmp_path / "packets",
        price_start="2026-01-01",
        price_end="2026-04-15",
        data_provider_name="dhan",
        allow_fallback=False,
    )
    assert cached["packets"] == first["packets"]
