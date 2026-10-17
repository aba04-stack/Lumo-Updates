from __future__ import annotations

import pytest

from adaptive_core import AffectEngine, Store


def test_affect_state_changes_from_events_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "affect.sqlite3"
    with Store(path) as store:
        affect = AffectEngine(store)
        initial = affect.state()
        after_failure = affect.update("task-failed")
        assert after_failure.frustration > initial.frustration
        assert after_failure.caution > initial.caution
        assert 0.0 <= after_failure.frustration <= 1.0
    with Store(path) as reopened:
        restored = AffectEngine(reopened).state()
        assert restored.frustration == pytest.approx(after_failure.frustration)
        assert restored.updated_at == after_failure.updated_at


def test_contradiction_increases_caution_and_curiosity_without_overflow() -> None:
    with Store(":memory:") as store:
        affect = AffectEngine(store)
        initial = affect.state()
        for _ in range(50):
            state = affect.update("contradiction-detected")
        assert state.caution > initial.caution
        assert state.curiosity > initial.curiosity
        for name in ("curiosity", "satisfaction", "frustration", "caution", "confidence", "engagement"):
            assert 0.0 <= getattr(state, name) <= 1.0


def test_unknown_appraisal_event_is_rejected() -> None:
    with Store(":memory:") as store:
        with pytest.raises(ValueError, match="unsupported"):
            AffectEngine(store).update("invented-feeling")


def test_outcome_training_updates_associations_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "learned-affect.sqlite3"
    with Store(path) as store:
        affect = AffectEngine(store, learning_rate=0.1)
        before = affect.model_snapshot()
        affect.appraise(
            succeeded=True,
            progress=0.9,
            uncertainty=0.1,
            contradiction=0.0,
            novelty=0.8,
            feedback=0.9,
        )
        after_success = affect.model_snapshot()
        assert after_success["samples"] == 1
        assert after_success["affect_weights"]["satisfaction"]["progress"] > before["affect_weights"]["satisfaction"]["progress"]
        assert after_success["reward_weights"]["progress"] > before["reward_weights"]["progress"]

    with Store(path) as reopened:
        restored = AffectEngine(reopened).model_snapshot()
        assert restored == after_success
        affect = AffectEngine(reopened, learning_rate=0.1)
        affect.appraise(
            succeeded=False,
            progress=0.05,
            uncertainty=0.9,
            contradiction=0.2,
            novelty=0.1,
            feedback=-0.8,
        )
        after_failure = affect.model_snapshot()
        assert after_failure["samples"] == 2
        assert after_failure["affect_weights"]["frustration"]["uncertainty"] > restored["affect_weights"]["frustration"]["uncertainty"]


def test_outcome_feedback_must_be_finite_and_bounded() -> None:
    with Store(":memory:") as store:
        affect = AffectEngine(store)
        with pytest.raises(ValueError, match="feedback"):
            affect.appraise(succeeded=True, feedback=2.0)
        with pytest.raises(ValueError, match="feedback"):
            affect.appraise(succeeded=False, feedback=float("nan"))
        with pytest.raises(ValueError, match="reward"):
            affect.learn_from_outcome(features={"progress": 0.5}, reward=2.0)

def test_reward_predictor_remains_bounded_during_repeated_learning() -> None:
    with Store(":memory:") as store:
        affect = AffectEngine(store)
        features = {
            "novelty": 1.0,
            "progress": 1.0,
            "uncertainty": 1.0,
            "contradiction": 1.0,
        }
        for _ in range(30):
            result = affect.learn_from_outcome(features=features, reward=1.0)
            assert -1.0 <= result["predicted_reward"] <= 1.0
            assert -1.0 <= result["prediction_error"] <= 1.0


def test_affect_state_ignores_malformed_latest_state_event() -> None:
    with Store(":memory:") as store:
        engine = AffectEngine(store)
        expected = engine.update("knowledge-found", intensity=0.8)
        store.record_event("affect-state-updated", "system", {"state": "corrupted"})
        assert engine.state() == expected


def test_corrupt_latest_affect_model_cannot_expand_learned_weight_bounds() -> None:
    with Store(":memory:") as store:
        engine = AffectEngine(store)
        engine.learn_from_outcome(features={"novelty": 1.0}, reward=0.5)
        expected = engine.model_snapshot()
        invalid_model = {
            "affect_weights": {
                emotion: {feature: 0.9 for feature in ("novelty", "progress", "uncertainty", "contradiction")}
                for emotion in ("curiosity", "satisfaction", "frustration", "caution", "confidence", "engagement")
            },
            "reward_weights": {feature: 0.9 for feature in ("novelty", "progress", "uncertainty", "contradiction")},
            "samples": 999,
            "last_prediction_error": 0.1,
        }
        store.record_event("affect-model-trained", "system", {"model": invalid_model})
        restored = engine.model_snapshot()
        assert restored == expected


def test_malformed_event_payloads_are_normalized_before_returning() -> None:
    with Store(":memory:") as store:
        store._connection.execute(
            "INSERT INTO events(event_type, task_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("legacy-list-payload", "test", "[]", "2026-10-17T00:00:00+00:00"),
        )
        event = store.recent_events(limit=1)[0]
        assert event["payload"]["_invalid_payload"] is True
        assert event["payload"]["_payload_error"] == "payload_not_object"


def test_appraisal_rejects_malformed_or_unknown_feature_inputs() -> None:
    with Store(":memory:") as store:
        engine = AffectEngine(store)
        with pytest.raises(ValueError, match="features must be a mapping"):
            engine.learn_from_outcome(features=[], reward=0.5)
        with pytest.raises(ValueError, match="unsupported appraisal features"):
            engine.learn_from_outcome(features={"novelty": 0.5, "surprise": 1.0}, reward=0.5)
        with pytest.raises(ValueError, match="unsupported appraisal event"):
            engine.update([])
