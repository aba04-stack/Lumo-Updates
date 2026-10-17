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
