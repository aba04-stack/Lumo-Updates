"""Evidence-based model of current capabilities and observed limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .core import AdaptiveCore


@dataclass(frozen=True, slots=True)
class SelfModelReport:
    """Operational self-description, not a claim of subjective self-awareness."""

    task_type: str
    selected_strategy_id: str
    observation_count: int
    mean_score: float | None
    success_rate: float | None
    evidence_strength: float
    recent_failure_events: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfModel:
    """Build a conservative capability report from persisted evaluation evidence."""

    FAILURE_EVENTS = {
        "task-execution-failed",
        "task-assessment-failed",
        "candidate-evaluation-failed",
        "candidate-proposal-failed",
    }

    def __init__(self, core: AdaptiveCore, *, history_window: int = 200) -> None:
        if isinstance(history_window, bool) or not isinstance(history_window, int):
            raise ValueError("history_window must be an integer")
        if not 1 <= history_window <= 10_000:
            raise ValueError("history_window must be in [1, 10000]")
        self.core = core
        self.history_window = history_window

    def report(self, task_type: str) -> SelfModelReport:
        selected = self.core.choose_strategy(task_type)
        performance = next(
            (item for item in self.core.performance(task_type)
             if item["strategy_id"] == selected.config.strategy_id),
            None,
        )
        count = int(performance["count"]) if performance else 0
        mean_score = float(performance["avg_score"]) if count else None
        success_rate = float(performance["success_rate"]) if count else None
        # Heuristic indicator of evidence volume, not a statistical confidence interval.
        evidence_strength = count / (count + 10.0) if count else 0.0
        failures: list[str] = []
        for event in self.core.store.events_of_types(
            list(self.FAILURE_EVENTS),
            limit=self.history_window,
            newest_first=True,
            task_type=task_type,
        ):
            if event["event_type"] not in failures:
                failures.append(event["event_type"])
        limits: list[str] = []
        if count < 5:
            limits.append("Too little evaluated experience for a stable capability estimate.")
        if mean_score is not None and mean_score < 0.6:
            limits.append("Recent measured performance is below the 0.60 reference threshold.")
        if failures:
            limits.append("Recent execution or evaluation failures were observed.")
        if not limits:
            limits.append("No major limitation detected by the currently available telemetry.")
        return SelfModelReport(
            task_type=task_type, selected_strategy_id=selected.config.strategy_id,
            observation_count=count, mean_score=mean_score, success_rate=success_rate,
            evidence_strength=evidence_strength, recent_failure_events=tuple(failures),
            limitations=tuple(limits),
        )
