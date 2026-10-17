"""End-to-end task loop that closes the feedback cycle."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .core import AdaptiveCore
from .models import StrategyConfig, validate_score


@dataclass(frozen=True, slots=True)
class Assessment:
    """Externally measured outcome; score must be normalized to [0, 1]."""

    score: float
    success: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", validate_score(self.score))
        if not isinstance(self.success, bool):
            raise ValueError("success must be a bool")
        try:
            copied = json.loads(
                json.dumps(dict(self.metadata), ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("assessment metadata must be finite JSON-compatible data") from exc
        object.__setattr__(self, "metadata", copied)


@dataclass(frozen=True, slots=True)
class RunResult:
    strategy_id: str
    output: Any
    assessment: Assessment
    duration_ms: float


Executor = Callable[[StrategyConfig, Any], Any]
Grader = Callable[[StrategyConfig, Any, Any], Assessment]


class AdaptiveRunner:
    """Choose -> execute -> grade independently -> persist feedback."""

    def __init__(self, core: AdaptiveCore) -> None:
        self.core = core
        self.store = core.store

    def run(
        self,
        task_type: str,
        payload: Any,
        *,
        executor: Executor,
        grader: Grader,
    ) -> RunResult:
        strategy = self.core.choose_strategy(task_type)
        started = time.perf_counter()
        try:
            output = executor(strategy.config, payload)
        except Exception as exc:
            self.store.record_event(
                "task-execution-failed",
                task_type,
                {
                    "strategy_id": strategy.config.strategy_id,
                    "error_type": type(exc).__name__,
                },
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000.0
        try:
            assessment = grader(strategy.config, payload, output)
            if not isinstance(assessment, Assessment):
                raise TypeError("grader must return an Assessment")
        except Exception as exc:
            self.store.record_event(
                "task-assessment-failed",
                task_type,
                {
                    "strategy_id": strategy.config.strategy_id,
                    "error_type": type(exc).__name__,
                },
            )
            # No score is recorded: a broken grader must not poison adaptation.
            raise

        self.core.record_result(
            strategy.config.strategy_id,
            assessment.score,
            assessment.success,
            dict(assessment.metadata),
        )
        self.store.record_event(
            "task-completed",
            task_type,
            {
                "strategy_id": strategy.config.strategy_id,
                "score": assessment.score,
                "success": assessment.success,
                "duration_ms": round(duration_ms, 3),
            },
        )
        return RunResult(
            strategy_id=strategy.config.strategy_id,
            output=output,
            assessment=assessment,
            duration_ms=duration_ms,
        )
