from __future__ import annotations

import pytest

from adaptive_core import AffectEngine, AdaptiveCore, CognitiveSystem, MotivationEngine, Store


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
        engine = MotivationEngine(reopened)
        needs_review = engine.set_goal_status(goal.goal_id, "needs_verification")
        assert needs_review.status == "needs_verification"
        with pytest.raises(ValueError, match="complete_goal"):
            engine.set_goal_status(goal.goal_id, "completed")
        completed = engine.complete_goal(
            goal.goal_id,
            method="manual_review",
            reviewer="test-reviewer",
            criteria=["The question has an explicit answer."],
            evidence_refs=["https://example.org/evidence"],
        )
        assert completed.status == "completed"
        verification_event = next(
            event for event in reopened.events_of_types(
                ["motivation-goal-verification-recorded"], limit=10, newest_first=True
            ) if event["payload"]["goal_id"] == goal.goal_id
        )
        assert verification_event["payload"]["criteria"] == ["The question has an explicit answer."]
        assert verification_event["payload"]["claim_verified"] is False
        assert verification_event["payload"]["reviewer_authenticated"] is False
        with pytest.raises(ValueError, match="invalid goal transition"):
            engine.set_goal_status(goal.goal_id, "queued")


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

def test_current_state_and_goal_history_survive_long_unrelated_event_log(tmp_path) -> None:
    path = tmp_path / "long-history.sqlite3"
    with Store(path) as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("reasoning")
        cognitive = CognitiveSystem(core)
        cognitive.learn_from_outcome(
            task_type="reasoning",
            strategy_id=baseline.config.strategy_id,
            summary="A failed test showed unresolved uncertainty.",
            tags=["test"],
            score=0.2,
            success=False,
            metadata={"uncertainty": 0.9},
        )
        goal = cognitive.motivation.add_goal(
            "Recheck a disputed claim", drive="uncertainty_reduction", priority=0.8
        )
        cognitive.motivation.set_goal_status(goal.goal_id, "active")
        cognitive.motivation.set_goal_status(goal.goal_id, "needs_verification")
        affect_before = cognitive.affect.state()
        drive_before = cognitive.motivation.state()

        for index in range(10_005):
            store.record_event("unrelated-noise", "stress-test", {"index": index})

    with Store(path) as reopened:
        cognitive = CognitiveSystem(AdaptiveCore(reopened))
        assert cognitive.affect.state() == affect_before
        assert cognitive.motivation.state() == drive_before
        goals = cognitive.motivation.goals(status="needs_verification")
        assert len(goals) == 1
        assert goals[0].goal_id == goal.goal_id


def test_goal_cannot_be_completed_by_a_bare_status_event() -> None:
    with Store(":memory:") as store:
        engine = MotivationEngine(store)
        goal = engine.add_goal("Review a disputed claim", drive="uncertainty_reduction")
        engine.set_goal_status(goal.goal_id, "active")
        engine.set_goal_status(goal.goal_id, "needs_verification")
        store.record_event("motivation-goal-status", "motivation", {
            "goal_id": goal.goal_id,
            "from": "needs_verification",
            "status": "completed",
            "verification_id": "not-a-real-verification-record",
        })
        restored = next(item for item in engine.goals() if item.goal_id == goal.goal_id)
        assert restored.status == "needs_verification"


def test_goal_completion_requires_nonempty_review_criteria_and_evidence() -> None:
    with Store(":memory:") as store:
        engine = MotivationEngine(store)
        goal = engine.add_goal("Verify a source claim", drive="uncertainty_reduction")
        engine.set_goal_status(goal.goal_id, "active")
        engine.set_goal_status(goal.goal_id, "needs_verification")
        with pytest.raises(ValueError, match="criteria"):
            engine.complete_goal(
                goal.goal_id, method="manual_review", reviewer="reviewer",
                criteria=[], evidence_refs=["https://example.org/evidence"],
            )
        with pytest.raises(ValueError, match="evidence_refs"):
            engine.complete_goal(
                goal.goal_id, method="manual_review", reviewer="reviewer",
                criteria=["Check the source"], evidence_refs=[],
            )
        assert next(item for item in engine.goals() if item.goal_id == goal.goal_id).status == "needs_verification"


def test_invalid_status_types_fail_with_validation_error() -> None:
    with Store(":memory:") as store:
        engine = MotivationEngine(store)
        goal = engine.add_goal("Validate API inputs", drive="competence")
        with pytest.raises(ValueError, match="unsupported goal status"):
            engine.set_goal_status(goal.goal_id, [])
