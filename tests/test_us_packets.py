import json

import pandas as pd
import pytest

from tradingagents.us.data import USDataFrame, USDataSnapshot
from tradingagents.us.packets import generate_packets


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

    def get_ohlcv(self, ticker, start, end, **kwargs):
        offset = 0 if ticker == "AAPL" else 25
        return USDataFrame(
            _prices(100 + offset, 1),
            USDataSnapshot(
                provider="fixture",
                endpoint="bars",
                params={"ticker": ticker, "start": start, "end": end},
                adjusted=kwargs.get("adjusted"),
                fetched_at="2026-01-01T00:00:00Z",
                cache_key=f"{ticker}-bars",
            ),
        )

    def get_benchmark_ohlcv(self, benchmark, start, end, **kwargs):
        return USDataFrame(
            _prices(300, 1),
            USDataSnapshot(
                provider="fixture",
                endpoint="benchmark",
                params={"ticker": benchmark, "start": start, "end": end},
                adjusted=kwargs.get("adjusted"),
                fetched_at="2026-01-01T00:00:00Z",
                cache_key=f"{benchmark}-bars",
            ),
        )


@pytest.mark.unit
def test_generate_us_data_packets_uses_deterministic_config_hash(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.us.packets.create_us_data_provider", lambda config: _Provider())

    first = generate_packets(
        tickers=["AAPL", "MSFT"],
        dates=["2026-02-02"],
        output_dir=tmp_path / "first",
        price_start="2026-01-01",
        price_end="2026-04-15",
        horizon_sessions=20,
        data_provider_name="massive",
    )
    second = generate_packets(
        tickers=["AAPL", "MSFT"],
        dates=["2026-02-02"],
        output_dir=tmp_path / "second",
        price_start="2026-01-01",
        price_end="2026-04-15",
        horizon_sessions=20,
        data_provider_name="massive",
    )

    assert first["manifest"]["config_hash"] == second["manifest"]["config_hash"]
    assert first["manifest"]["packet_count"] == 2
    packet = first["packets"][0]
    assert packet["ticker"] == "AAPL"
    assert packet["price_features"]["return_20d"] is not None
    assert packet["benchmark_features"]["SPY"]["return_20d"] is not None
    assert packet["correlation_features"]["top_correlations"]
    assert packet["data_quality"]["provider"] == "fixture"

    packet_path = tmp_path / "first" / f"AAPL_2026-02-02_{first['manifest']['config_hash']}.json"
    assert json.loads(packet_path.read_text(encoding="utf-8"))["config_hash"] == first["manifest"]["config_hash"]

    monkeypatch.setattr(
        "tradingagents.us.packets.create_us_data_provider",
        lambda config: (_ for _ in ()).throw(AssertionError("packet cache should skip provider creation")),
    )
    cached = generate_packets(
        tickers=["AAPL", "MSFT"],
        dates=["2026-02-02"],
        output_dir=tmp_path / "first",
        price_start="2026-01-01",
        price_end="2026-04-15",
        horizon_sessions=20,
        data_provider_name="massive",
    )
    assert cached["packets"] == first["packets"]
