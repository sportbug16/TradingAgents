"""Experiment registry and batch runner for TradingAgents studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Sequence
from uuid import uuid4

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .rules import Signal


@dataclass(frozen=True)
class BatchRunConfig:
    instruments: Sequence[str]
    decision_dates: Sequence[str]
    selected_analysts: Sequence[str] = ("market", "news", "fundamentals")
    checkpoint: bool = True
    data_snapshot_version: str = "live"
    config_overrides: dict | None = None


class ExperimentRegistry:
    """Append-only JSONL registry for reproducible graph experiments."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "runs.jsonl"

    def record(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]


class GraphBatchRunner:
    """Runs TradingAgentsGraph across instrument/date grids and records signals."""

    def __init__(
        self,
        registry: ExperimentRegistry,
        graph_factory: Callable[[list[str], dict], TradingAgentsGraph] | None = None,
    ):
        self.registry = registry
        self.graph_factory = graph_factory or self._default_graph_factory

    def run(self, batch: BatchRunConfig) -> list[Signal]:
        config = DEFAULT_CONFIG.copy()
        config["checkpoint_enabled"] = batch.checkpoint
        if batch.config_overrides:
            config.update(batch.config_overrides)

        graph = self.graph_factory(list(batch.selected_analysts), config)
        signals: list[Signal] = []
        for ticker in batch.instruments:
            for decision_date in batch.decision_dates:
                run_id = str(uuid4())
                started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                final_state, rating = graph.propagate(ticker, decision_date)
                final_decision = final_state["final_trade_decision"]
                parsed = parse_rating(final_decision, default=rating)
                record = {
                    "run_id": run_id,
                    "started_at": started_at,
                    "ticker": ticker,
                    "decision_date": decision_date,
                    "rating": parsed,
                    "final_decision": final_decision,
                    "model_provider": config.get("llm_provider"),
                    "quick_model": config.get("quick_think_llm"),
                    "deep_model": config.get("deep_think_llm"),
                    "config_hash": _stable_hash(config),
                    "git_sha": _git_sha(),
                    "data_snapshot_version": batch.data_snapshot_version,
                    "selected_analysts": list(batch.selected_analysts),
                }
                self.registry.record(record)
                signals.append(
                    Signal(
                        ticker=ticker,
                        decision_date=decision_date,
                        rating=parsed,
                        final_decision=final_decision,
                        run_id=run_id,
                    )
                )
        return signals

    @staticmethod
    def _default_graph_factory(selected_analysts: list[str], config: dict) -> TradingAgentsGraph:
        return TradingAgentsGraph(selected_analysts=selected_analysts, config=config, debug=False)


def _stable_hash(value: dict) -> str:
    blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
