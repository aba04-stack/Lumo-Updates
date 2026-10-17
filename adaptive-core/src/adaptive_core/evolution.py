"""Evaluation gates for promoting or rolling back strategy configurations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean
from typing import Any

from .core import AdaptiveCore
from .models import EvaluationCase, PromotionDecision, StrategyConfig, validate_score
from .store import Store

Evaluator = Callable[[StrategyConfig, EvaluationCase], float]
Proposer = Callable[[StrategyConfig, Mapping[str, Any]], StrategyConfig]


class EvolutionManager:
    """Compare candidate and incumbent on the same cases before activation."""

    def __init__(
        self,
        core: AdaptiveCore,
        *,
        minimum_cases: int = 5,
        minimum_gain: float = 0.03,
        max_case_regression: float = 0.15,
    ) -> None:
        if isinstance(minimum_cases, bool) or not isinstance(minimum_cases, int) or minimum_cases < 2:
            raise ValueError("minimum_cases must be an integer >= 2")
        for name, value in (
            ("minimum_gain", minimum_gain),
            ("max_case_regression", max_case_regression),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(float(value)) or value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.core = core
        self.store: Store = core.store
        self.minimum_cases = minimum_cases
        self.minimum_gain = float(minimum_gain)
        self.max_case_regression = float(max_case_regression)

    def propose_and_evaluate(
        self,
        task_type: str,
        cases: Sequence[EvaluationCase],
        *,
        proposer: Proposer,
        evaluator: Evaluator,
    ) -> PromotionDecision:
        """Ask an external proposer for a data-only revision, then gate it.

        The proposer can be an LLM, an optimizer, or a deterministic function.
        It receives aggregate performance, not raw user prompts or stored payloads.
        A proposal never becomes active until evaluate_candidate accepts it.
        """
        baseline = self.core.choose_strategy(task_type)
        recent = [
            {"event_type": event["event_type"], "created_at": event["created_at"]}
            for event in reversed(self.store.recent_events(limit=200))
            if event["task_type"] == task_type
        ][-20:]
        feedback: dict[str, Any] = {
            "task_type": task_type,
            "strategy_performance": self.core.performance(task_type),
            "recent_event_types": recent,
            "guardrails": {
                "minimum_gain": self.minimum_gain,
                "max_case_regression": self.max_case_regression,
                "minimum_cases": self.minimum_cases,
                "source_code_execution": False,
            },
        }
        try:
            candidate_config = proposer(baseline.config, feedback)
            if not isinstance(candidate_config, StrategyConfig):
                raise TypeError("proposer must return a StrategyConfig")
            if candidate_config.task_type != task_type:
                raise ValueError("proposed task_type does not match the target task")
            if candidate_config.version <= baseline.config.version:
                raise ValueError("candidate version must be greater than its baseline version")
            try:
                self.store.get_strategy(candidate_config.strategy_id)
            except KeyError:
                pass
            else:
                raise ValueError("candidate strategy_id already exists; versions are immutable")
        except Exception as exc:
            self.store.record_event(
                "candidate-proposal-failed",
                task_type,
                {
                    "baseline_id": baseline.config.strategy_id,
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:300],
                },
            )
            raise

        candidate = self.core.register_candidate(
            candidate_config, parent_id=baseline.config.strategy_id
        )
        self.store.record_event(
            "candidate-proposed",
            task_type,
            {
                "candidate_id": candidate.config.strategy_id,
                "baseline_id": baseline.config.strategy_id,
                "version": candidate.config.version,
            },
        )
        return self.evaluate_candidate(candidate.config.strategy_id, cases, evaluator)

    def evaluate_candidate(
        self,
        candidate_id: str,
        cases: Sequence[EvaluationCase],
        evaluator: Evaluator,
    ) -> PromotionDecision:
        candidate = self.store.get_strategy(candidate_id)
        if candidate.status != "candidate":
            raise ValueError("only a candidate strategy can be evaluated for promotion")
        if candidate.parent_id is not None:
            parent = self.store.get_strategy(candidate.parent_id)
            baseline = parent if parent.status == "active" else None
        else:
            baseline = None
        if baseline is None:
            baseline = self.core.choose_strategy(candidate.config.task_type)
        if baseline.config.strategy_id == candidate_id:
            raise ValueError("candidate cannot be its own baseline")

        case_list = list(cases)
        if len(case_list) < self.minimum_cases:
            return self._reject(
                candidate.config,
                baseline.config.strategy_id,
                f"need at least {self.minimum_cases} fixed evaluation cases",
                0,
            )
        if any(case.task_type != candidate.config.task_type for case in case_list):
            return self._reject(
                candidate.config, baseline.config.strategy_id, "case task_type mismatch", 0
            )
        if len({case.case_id for case in case_list}) != len(case_list):
            return self._reject(
                candidate.config, baseline.config.strategy_id, "evaluation case IDs must be unique", 0
            )

        baseline_scores: list[float] = []
        candidate_scores: list[float] = []
        try:
            # Same cases, same order, same evaluator: avoid incomparable test sets.
            for case in case_list:
                baseline_scores.append(validate_score(evaluator(baseline.config, case)))
                candidate_scores.append(validate_score(evaluator(candidate.config, case)))
        except Exception as exc:
            reason = f"evaluation failed closed ({type(exc).__name__})"
            self.store.record_event(
                "candidate-evaluation-failed",
                candidate.config.task_type,
                {"candidate_id": candidate_id, "reason": reason},
            )
            return PromotionDecision(
                False, reason, candidate_id, baseline.config.strategy_id,
                None, None, len(baseline_scores), None
            )

        baseline_mean = fmean(baseline_scores)
        candidate_mean = fmean(candidate_scores)
        max_regression = max(
            max(0.0, base - trial)
            for base, trial in zip(baseline_scores, candidate_scores, strict=True)
        )
        gain = candidate_mean - baseline_mean
        accepted = gain >= self.minimum_gain and max_regression <= self.max_case_regression
        if accepted:
            # Compare-and-set prevents promotion if another process changed the state.
            self.store.set_status(candidate_id, "active", expected_status="candidate")
            reason = f"promoted: mean gain {gain:.3f}, max case regression {max_regression:.3f}"
            event_type = "candidate-promoted"
        else:
            parts = []
            if gain < self.minimum_gain:
                parts.append(f"gain {gain:.3f} is below required {self.minimum_gain:.3f}")
            if max_regression > self.max_case_regression:
                parts.append(
                    f"max regression {max_regression:.3f} exceeds limit {self.max_case_regression:.3f}"
                )
            reason = "; ".join(parts) or "quality gate rejected candidate"
            event_type = "candidate-rejected"
        self.store.record_event(
            event_type,
            candidate.config.task_type,
            {
                "candidate_id": candidate_id,
                "baseline_id": baseline.config.strategy_id,
                "candidate_score": candidate_mean,
                "baseline_score": baseline_mean,
                "gain": gain,
                "max_case_regression": max_regression,
                "cases": len(case_list),
                "accepted": accepted,
                "reason": reason,
            },
        )
        return PromotionDecision(
            accepted,
            reason,
            candidate_id,
            baseline.config.strategy_id,
            candidate_mean,
            baseline_mean,
            len(case_list),
            max_regression,
        )

    def _reject(
        self, candidate: StrategyConfig, baseline_id: str, reason: str, cases: int
    ) -> PromotionDecision:
        self.store.record_event(
            "candidate-rejected",
            candidate.task_type,
            {"candidate_id": candidate.strategy_id, "baseline_id": baseline_id, "reason": reason},
        )
        return PromotionDecision(False, reason, candidate.strategy_id, baseline_id, None, None, cases, None)

    def rollback(self, strategy_id: str) -> None:
        strategy = self.store.get_strategy(strategy_id)
        if strategy.status != "active":
            raise ValueError("only an active strategy can be rolled back")
        active = self.store.list_strategies(strategy.config.task_type, status="active")
        if len(active) <= 1:
            raise ValueError("cannot disable the last active strategy; no safe fallback would remain")
        self.store.set_status(strategy_id, "disabled", expected_status="active")
        self.store.record_event(
            "strategy-rollback",
            strategy.config.task_type,
            {
                "disabled_strategy_id": strategy_id,
                "fallbacks": [
                    item.config.strategy_id for item in active
                    if item.config.strategy_id != strategy_id
                ],
            },
        )

    def restore(self, strategy_id: str) -> None:
        strategy = self.store.get_strategy(strategy_id)
        if strategy.status != "disabled":
            raise ValueError("only a disabled strategy can be restored")
        self.store.set_status(strategy_id, "active", expected_status="disabled")
        self.store.record_event(
            "strategy-restored",
            strategy.config.task_type,
            {"strategy_id": strategy_id},
        )
