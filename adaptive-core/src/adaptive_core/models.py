"""Typed data objects and validation for adaptive-core."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{field_name} must be 1-128 ASCII letters, digits, '.', '_', ':', or '-'."
        )


def validate_score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a number in [0, 1]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a finite number in [0, 1]")
    return score


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Versioned, data-only strategy configuration; this is never executable code."""

    strategy_id: str
    task_type: str
    instructions: str
    version: int = 1
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.strategy_id, "strategy_id")
        validate_identifier(self.task_type, "task_type")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions must be a non-empty string")
        if len(self.instructions) > 100_000:
            raise ValueError("instructions exceed 100,000 characters")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive integer")
        try:
            copied = json.loads(
                json.dumps(dict(self.parameters), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must be finite JSON-compatible data") from exc
        object.__setattr__(self, "parameters", copied)


@dataclass(frozen=True, slots=True)
class StoredStrategy:
    config: StrategyConfig
    status: str
    parent_id: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A fixed evaluation item shared by incumbent and candidate."""

    case_id: str
    task_type: str
    input_text: str
    expected: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.case_id, "case_id")
        validate_identifier(self.task_type, "task_type")
        if not isinstance(self.input_text, str) or len(self.input_text) > 200_000:
            raise ValueError("input_text must be a string no longer than 200,000 characters")
        if self.expected is not None and (
            not isinstance(self.expected, str) or len(self.expected) > 200_000
        ):
            raise ValueError("expected must be a string of at most 200,000 characters or None")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    accepted: bool
    reason: str
    candidate_id: str
    baseline_id: str | None
    candidate_score: float | None
    baseline_score: float | None
    cases_evaluated: int
    max_case_regression: float | None
