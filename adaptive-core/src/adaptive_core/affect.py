"""Learnable appraisal-to-affect control states; not a claim of subjective feeling."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any

from .models import validate_score
from .store import Store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_BASELINE = {
    "curiosity": 0.55,
    "satisfaction": 0.30,
    "frustration": 0.10,
    "caution": 0.35,
    "confidence": 0.25,
    "engagement": 0.45,
}
_FEATURES = ("novelty", "progress", "uncertainty", "contradiction")

# Small initial priors let the system behave before it has experience.
# These are feature-to-state weights, not event-to-emotion response scripts.
_INITIAL_AFFECT_WEIGHTS = {
    "curiosity": {"novelty": 0.06, "progress": 0.025, "uncertainty": 0.035, "contradiction": 0.07},
    "satisfaction": {"novelty": 0.005, "progress": 0.04, "uncertainty": -0.01, "contradiction": -0.04},
    "frustration": {"novelty": 0.0, "progress": -0.03, "uncertainty": 0.03, "contradiction": 0.05},
    "caution": {"novelty": 0.0, "progress": -0.01, "uncertainty": 0.06, "contradiction": 0.10},
    "confidence": {"novelty": 0.0, "progress": 0.04, "uncertainty": -0.045, "contradiction": -0.06},
    "engagement": {"novelty": 0.03, "progress": 0.04, "uncertainty": -0.01, "contradiction": -0.015},
}
# A weak inductive bias says which direction each control state has in the
# outcome-learning signal. Context-specific associations remain trainable.
_OUTCOME_POLARITY = {
    "curiosity": 0.20,
    "satisfaction": 1.00,
    "frustration": -1.00,
    "caution": -0.35,
    "confidence": 0.70,
    "engagement": 0.45,
}
_INITIAL_REWARD_WEIGHTS = {feature: 0.0 for feature in _FEATURES}
_WEIGHT_LIMIT = 0.30

# Older call sites can still send event names; events are translated to a
# coarse appraisal vector, then processed by the same weighted model.
_EVENT_FEATURES = {
    "task-started": {"novelty": 0.75, "progress": 0.05, "uncertainty": 0.10, "contradiction": 0.0},
    "knowledge-found": {"novelty": 0.80, "progress": 0.60, "uncertainty": 0.25, "contradiction": 0.0},
    "search-empty": {"novelty": 0.15, "progress": 0.05, "uncertainty": 0.55, "contradiction": 0.05},
    "task-succeeded": {"novelty": 0.10, "progress": 0.90, "uncertainty": 0.05, "contradiction": 0.0},
    "task-failed": {"novelty": 0.15, "progress": 0.0, "uncertainty": 0.65, "contradiction": 0.10},
    "contradiction-detected": {"novelty": 0.45, "progress": 0.10, "uncertainty": 0.80, "contradiction": 0.90},
    "uncertainty-detected": {"novelty": 0.50, "progress": 0.10, "uncertainty": 0.90, "contradiction": 0.10},
    "information-gain": {"novelty": 0.70, "progress": 0.90, "uncertainty": 0.20, "contradiction": 0.0},
    "no-progress": {"novelty": 0.10, "progress": 0.0, "uncertainty": 0.30, "contradiction": 0.05},
    "test-passed": {"novelty": 0.05, "progress": 0.90, "uncertainty": 0.05, "contradiction": 0.0},
    "test-failed": {"novelty": 0.10, "progress": 0.0, "uncertainty": 0.70, "contradiction": 0.10},
}


@dataclass(frozen=True, slots=True)
class AffectState:
    """Bounded, interpretable control variables influenced by observed events."""
    curiosity: float = _BASELINE["curiosity"]
    satisfaction: float = _BASELINE["satisfaction"]
    frustration: float = _BASELINE["frustration"]
    caution: float = _BASELINE["caution"]
    confidence: float = _BASELINE["confidence"]
    engagement: float = _BASELINE["engagement"]
    updated_at: str = "initial"


class AffectEngine:
    """Online-learned appraisal model for persistent affect-like control states.

    The initial weights are modest priors. Measured outcomes update a reward
    predictor and the context-to-state associations. This is a small
    reinforcement-style learner, not a neural model or evidence of feelings.
    """

    def __init__(
        self,
        store: Store,
        *,
        decay: float = 0.02,
        learning_rate: float = 0.05,
    ) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError("decay must be a finite number in [0, 1]")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.5:
            raise ValueError("learning_rate must be a finite number in (0, 0.5]")
        self.store = store
        self.decay = float(decay)
        self.learning_rate = float(learning_rate)

    @staticmethod
    def _features(values: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError("features must be a mapping of known feature names to scores")
        unknown = set(values) - set(_FEATURES)
        if unknown:
            raise ValueError(f"unsupported appraisal features: {', '.join(sorted(map(str, unknown))[:10])}")
        return {name: validate_score(values.get(name, 0.0)) for name in _FEATURES}

    @staticmethod
    def _clip_weight(value: float) -> float:
        return min(_WEIGHT_LIMIT, max(-_WEIGHT_LIMIT, value))

    def _load_model(self) -> dict[str, Any]:
        for event in self.store.events_of_types(["affect-model-trained"], limit=1000, newest_first=True):
            model = event["payload"].get("model")
            try:
                affect_weights = {
                    emotion: {
                        feature: float(model["affect_weights"][emotion][feature])
                        for feature in _FEATURES
                    }
                    for emotion in _BASELINE
                }
                reward_weights = {feature: float(model["reward_weights"][feature]) for feature in _FEATURES}
                samples = model["samples"]
                if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
                    continue
                affect_values = [v for row in affect_weights.values() for v in row.values()]
                if any(not math.isfinite(v) or abs(v) > _WEIGHT_LIMIT for v in affect_values):
                    continue
                if any(not math.isfinite(v) or abs(v) > 1.0 for v in reward_weights.values()):
                    continue
                last_error = float(model.get("last_prediction_error", 0.0))
                if not math.isfinite(last_error) or abs(last_error) > 1.0:
                    continue
                return {
                    "affect_weights": affect_weights,
                    "reward_weights": reward_weights,
                    "samples": samples,
                    "last_prediction_error": last_error,
                }
            except (KeyError, TypeError, ValueError):
                continue
        return {
            "affect_weights": {emotion: dict(row) for emotion, row in _INITIAL_AFFECT_WEIGHTS.items()},
            "reward_weights": dict(_INITIAL_REWARD_WEIGHTS),
            "samples": 0,
            "last_prediction_error": 0.0,
        }

    def model_snapshot(self) -> dict[str, Any]:
        """Return a copy of learned parameters for tests and diagnostics."""
        model = self._load_model()
        return {
            "affect_weights": {name: dict(row) for name, row in model["affect_weights"].items()},
            "reward_weights": dict(model["reward_weights"]),
            "samples": model["samples"],
            "last_prediction_error": model["last_prediction_error"],
        }

    def state(self) -> AffectState:
        # Skip malformed recent state events and restore the newest valid state
        # from this stream; unrelated events and bad payloads must not reset it.
        for event in self.store.events_of_types(
            ["affect-state-updated"], limit=1_000, newest_first=True
        ):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            state_payload = payload.get("state")
            if not isinstance(state_payload, dict):
                continue
            try:
                values = {name: validate_score(state_payload[name]) for name in _BASELINE}
                updated_at = state_payload.get("updated_at", event["created_at"])
                if isinstance(updated_at, str):
                    return AffectState(**values, updated_at=updated_at)
            except (KeyError, TypeError, ValueError):
                continue
        return AffectState()

    def _transition(self, features: dict[str, float], *, trigger: str) -> AffectState:
        previous = self.state()
        model = self._load_model()
        values: dict[str, float] = {}
        for emotion, baseline in _BASELINE.items():
            response = sum(
                model["affect_weights"][emotion][feature] * features[feature]
                for feature in _FEATURES
            )
            decayed = getattr(previous, emotion) + (baseline - getattr(previous, emotion)) * self.decay
            values[emotion] = min(1.0, max(0.0, decayed + response))
        state = AffectState(**values, updated_at=_now())
        self.store.record_event("affect-state-updated", "system", {
            "trigger": trigger, "features": features, "state": asdict(state),
            "model_samples": model["samples"],
        })
        return state

    def update(self, event_type: str, *, intensity: float = 1.0) -> AffectState:
        """Update from a known appraisal event using the shared weighted model."""
        if not isinstance(event_type, str) or event_type not in _EVENT_FEATURES:
            raise ValueError(f"unsupported appraisal event: {event_type}")
        intensity = validate_score(intensity)
        features = {
            name: _EVENT_FEATURES[event_type][name] * intensity
            for name in _FEATURES
        }
        return self._transition(features, trigger=event_type)

    def learn_from_outcome(self, *, features: dict[str, float], reward: float) -> dict[str, Any]:
        """Learn context/outcome associations using a bounded online prediction error.

        Reward must be an externally measured task outcome, not a claim that a
        source is true. The learner changes future emotional-control responses.
        """
        normalized = self._features(features)
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be a finite number in [-1, 1]")
        reward = float(reward)
        model = self._load_model()
        predicted = max(-1.0, min(1.0, sum(model["reward_weights"][feature] * normalized[feature] for feature in _FEATURES)))
        error = max(-1.0, min(1.0, reward - predicted))
        rate = self.learning_rate
        reward_weights = dict(model["reward_weights"])
        for feature in _FEATURES:
            reward_weights[feature] = max(-1.0, min(
                1.0, reward_weights[feature] + rate * error * normalized[feature]
            ))
        affect_weights = {emotion: dict(row) for emotion, row in model["affect_weights"].items()}
        for emotion, polarity in _OUTCOME_POLARITY.items():
            for feature in _FEATURES:
                old = affect_weights[emotion][feature]
                affect_weights[emotion][feature] = self._clip_weight(
                    old + rate * polarity * error * normalized[feature]
                )
        new_model = {
            "affect_weights": affect_weights,
            "reward_weights": reward_weights,
            "samples": model["samples"] + 1,
            "last_prediction_error": error,
        }
        self.store.record_event("affect-model-trained", "system", {
            "model_version": 1,
            "features": normalized,
            "reward": reward,
            "predicted_reward": predicted,
            "prediction_error": error,
            "model": new_model,
        })
        return {
            "samples": new_model["samples"],
            "predicted_reward": predicted,
            "prediction_error": error,
        }

    def appraise(
        self,
        *,
        succeeded: bool,
        progress: float = 0.0,
        uncertainty: float = 0.0,
        contradiction: float = 0.0,
        novelty: float = 0.0,
        feedback: float | None = None,
    ) -> AffectState:
        """Apply an appraisal, then learn its context from the measured outcome.

        Feedback, when supplied, is an outcome score from -1 to 1. Otherwise
        a conservative reward is inferred from task success and measured progress.
        """
        if not isinstance(succeeded, bool):
            raise ValueError("succeeded must be a bool")
        features = self._features({
            "novelty": novelty,
            "progress": progress,
            "uncertainty": uncertainty,
            "contradiction": contradiction,
        })
        if feedback is None:
            reward = float(progress) if succeeded else -max(0.25, 1.0 - float(progress))
        else:
            if isinstance(feedback, bool) or not isinstance(feedback, (int, float)) or not math.isfinite(feedback) or not -1.0 <= feedback <= 1.0:
                raise ValueError("feedback must be a finite number in [-1, 1]")
            reward = float(feedback)
        state = self._transition(features, trigger="appraisal")
        self.learn_from_outcome(features=features, reward=reward)
        return state
