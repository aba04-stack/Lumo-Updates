from __future__ import annotations

import pytest

from adaptive_core import AdaptiveCore, CognitiveSystem, MotivationEngine, Store


def test_drive_pressures_change_from_outcome_and_persist(tmp_path) -> None:
    path = tmp_path / "motivation.sqlite3"
    with Store(path) as store:
        engine = MotivationEngine(store)
        before = engine.state()
        after = engine.update_outcome(
            succeeded=False, progress=0.0, uncertainty=0.8, contradiction=0.2, novelty=0.1
        )
        assert after.competence > before.competence
        assert after.progress > before.progress
        assert after.uncertainty_reduction > before.uncertainty_reduction
        assert after.strongest() in {
            "curiosity", "competence", "coherence", "progress", "uncertainty_reduction"
        }

    with Store(path) as reopened:
        restored = MotivationEngine(reopened).state()
        assert restored == after
        for name in ("curiosity", "competence", "coherence", "progress", "uncertainty_reduction"):
            assert 0.0 <= getattr(restored, name) <= 1.0


def test_goal_lifecycle_is_persistent_and_idempotent(tmp_path) -> None:
    path = tmp_path / "goals.sqlite3"
    with Store(path) as store:
        engine = MotivationEngine(store)
        goal = engine.add_goal(
            "Investigate an unresolved question",
            drive="uncertainty_reduction",
            priority=0.7,
            rationale="verify_uncertainty",
        )
        duplicate = engine.add_goal(
            "Investigate an unresolved question",
            drive="uncertainty_reduction",
            priority=0.2,
            rationale="duplicate should not overwrite original",
        )
        assert duplicate.goal_id == goal.goal_id
        assert duplicate.priority == pytest.approx(0.7)
        active = engine.set_goal_status(goal.goal_id, "active")
        assert active.status == "active"
        deferred = engine.set_goal_status(goal.goal_id, "deferred")
        assert deferred.status == "deferred"
        assert len(engine.goals(status="deferred")) == 1

    with Store(path) as reopened:
        restored = MotivationEngine(reopened).goals()[0]
        assert restored.goal_id == goal.goal_id
        assert restored.status == "deferred"
        with pytest.raises(ValueError, match="invalid goal transition"):
            MotivationEngine(reopened).set_goal_status(goal.goal_id, "queued")


def test_drive_pressure_can_change_which_goal_is_preferred() -> None:
    with Store(":memory:") as store:
        engine = MotivationEngine(store)
        first = engine.add_goal("Explore new material", drive="curiosity", priority=0.45)
        second = engine.add_goal("Resolve uncertainty", drive="uncertainty_reduction", priority=0.45)
        # Strong unresolved uncertainty should alter the relative motivation score.
        state = engine.update_outcome(
            succeeded=False, progress=0.0, uncertainty=1.0, contradiction=0.8, novelty=0.0
        )
        assert engine.goal_score(second.drive, second.priority) > engine.goal_score(first.drive, first.priority)
        assert state.uncertainty_reduction > state.curiosity


def test_cognitive_outcome_updates_motivation() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("coding")
        cognitive = CognitiveSystem(core)
        before = cognitive.motivation.state()
        cognitive.learn_from_outcome(
            task_type="coding", strategy_id=baseline.config.strategy_id,
            summary="A failed test exposed an unresolved issue.",
            score=0.1, success=False,
            metadata={"uncertainty": 0.8, "novelty": 0.1},
        )
        after = cognitive.motivation.state()
        assert after.competence > before.competence
        assert after.uncertainty_reduction > before.uncertainty_reduction


def test_invalid_drive_signals_are_rejected() -> None:
    with Store(":memory:") as store:
        engine = MotivationEngine(store)
        with pytest.raises(ValueError):
            engine.update_outcome(succeeded=True, progress=float("nan"))
        with pytest.raises(ValueError, match="unsupported drive"):
            engine.add_goal("Bad goal", drive="human-soul")
