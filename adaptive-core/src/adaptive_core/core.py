"""Runtime strategy choice and online adaptation from measured outcomes."""

from __future__ import annotations

import math
from typing import Any

from .models import StrategyConfig, StoredStrategy, validate_score
from .store import Store


class NoActiveStrategyError(RuntimeError):
    """Raised when no approved strategy exists for a task type."""


class AdaptiveCore:
    """Select strategies based on observed quality while exploring alternatives."""

    def __init__(self, store: Store, *, exploration: float = 0.35) -> None:
        if isinstance(exploration, bool) or not isinstance(exploration, (int, float)):
            raise ValueError("exploration must be a non-negative number")
        if not math.isfinite(float(exploration)) or exploration < 0:
            raise ValueError("exploration must be a finite non-negative number")
        self.store = store
        self.exploration = float(exploration)

    def create_baseline(
        self, task_type: str, instructions: str = "Use the task instructions and verify the result."
    ) -> StoredStrategy:
        config = StrategyConfig(
            strategy_id=f"baseline:{task_type}",
            task_type=task_type,
            instructions=instructions,
        )
        return self.store.create_strategy(config, status="active")

    def register_candidate(
        self, config: StrategyConfig, *, parent_id: str | None = None
    ) -> StoredStrategy:
        if parent_id is None:
            active = self.store.list_strategies(config.task_type, status="active")
            if active:
                parent_id = active[0].config.strategy_id
        return self.store.create_strategy(config, status="candidate", parent_id=parent_id)

    def choose_strategy(self, task_type: str) -> StoredStrategy:
        active = self.store.list_strategies(task_type, status="active")
        if not active:
            raise NoActiveStrategyError(f"No active strategy for task_type={task_type!r}")
        stats = self.store.stats(task_type)
        total = self.store.total_observations(task_type)
        # Give each newly activated strategy a measured chance before ranking the field.
        # An untried promoted version is tested instead of being hidden by stale baseline metrics.
        unobserved = [strategy for strategy in active if strategy.config.strategy_id not in stats]
        if unobserved:
            return unobserved[-1]
        if total == 0:
            return active[-1]

        def upper_confidence(strategy: StoredStrategy) -> float:
            observed = stats.get(strategy.config.strategy_id)
            if observed is None:
                count = 0.0
                mean_reward = 0.0
            else:
                count = observed["count"]
                mean_reward = 0.75 * observed["avg_score"] + 0.25 * observed["success_rate"]
            bonus = self.exploration * math.sqrt(math.log(total + 1.0) / (count + 1.0))
            return mean_reward + bonus

        # This is an exploration/exploitation policy, not a claim of model intelligence.
        return max(active, key=upper_confidence)

    def record_result(
        self,
        strategy_id: str,
        score: float,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        strategy = self.store.get_strategy(strategy_id)
        return self.store.record_outcome(
            strategy.config.task_type, strategy_id, validate_score(score), success, metadata
        )

    def performance(self, task_type: str) -> list[dict[str, Any]]:
        stats = self.store.stats(task_type)
        return [
            {
                "strategy_id": strategy.config.strategy_id,
                "status": strategy.status,
                **stats.get(
                    strategy.config.strategy_id,
                    {"count": 0.0, "avg_score": 0.0, "success_rate": 0.0},
                ),
            }
            for strategy in self.store.list_strategies(task_type)
        ]
