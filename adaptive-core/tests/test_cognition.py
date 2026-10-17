from __future__ import annotations

import pytest

from adaptive_core import (
    AdaptiveCore,
    CognitiveSystem,
    EvaluationCase,
    EvolutionManager,
    Store,
    StrategyConfig,
)
from adaptive_core.workspace import GlobalWorkspace, WorkspaceSignal


def signal(signal_id: str, priority: float, confidence: float = 1.0) -> WorkspaceSignal:
    return WorkspaceSignal(signal_id, "observation", "test", signal_id, priority, confidence)


def test_workspace_prioritizes_and_bounds_signals() -> None:
    workspace = GlobalWorkspace(capacity=2)
    assert workspace.publish(signal("low", 0.1))
    assert workspace.publish(signal("medium", 0.5))
    assert workspace.publish(signal("high", 0.9))
    assert [item.signal_id for item in workspace.snapshot()] == ["high", "medium"]
    assert not workspace.publish(signal("too-low", 0.01))


def test_broken_workspace_listener_cannot_break_publish() -> None:
    workspace = GlobalWorkspace(capacity=2)
    workspace.subscribe(lambda _: (_ for _ in ()).throw(RuntimeError("listener broken")))
    assert workspace.publish(signal("event", 0.8))
    assert workspace.snapshot()[0].signal_id == "event"
    assert workspace.listener_errors()[0]["error_type"] == "RuntimeError"


def test_episodic_memory_is_persistent_and_retrievable(tmp_path) -> None:
    db = tmp_path / "memory.sqlite3"
    with Store(db) as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("reasoning")
        cognitive = CognitiveSystem(core)
        cognitive.learn_from_outcome(
            task_type="reasoning",
            strategy_id=baseline.config.strategy_id,
            summary="Found a reasoning failure caused by missing edge-case validation.",
            tags=["reasoning", "edge-case"],
            score=0.3,
            success=False,
            metadata={"source": "unit-test"},
        )

    with Store(db) as reopened:
        results = reopened.search_episodes(
            "edge-case validation", task_type="reasoning", tags=["edge-case"]
        )
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "edge-case" in results[0]["tags"]


def test_self_model_reports_uncertainty_and_failure_limits() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("coding")
        cognitive = CognitiveSystem(core)
        store.record_event("task-execution-failed", "coding", {"error_type": "ToolError"})
        report = cognitive.self_model.report("coding")
        assert report.selected_strategy_id == baseline.config.strategy_id
        assert report.observation_count == 0
        assert report.evidence_strength == 0.0
        assert "task-execution-failed" in report.recent_failure_events
        assert report.limitations


def test_integrated_snapshot_contains_shared_state_memory_and_self_model() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("research")
        cognitive = CognitiveSystem(core, workspace_capacity=4)
        cognitive.perceive(
            signal_id="signal-1",
            kind="hypothesis",
            source="planner",
            content="Check whether independent validation finds a regression.",
            priority=0.9,
            confidence=0.8,
        )
        cognitive.learn_from_outcome(
            task_type="research",
            strategy_id=baseline.config.strategy_id,
            summary="Independent validation detected a regression.",
            tags=["validation", "regression"],
            score=0.5,
            success=False,
        )
        snapshot = cognitive.reflect("research", query="validation regression")
        assert snapshot.task_type == "research"
        assert snapshot.workspace
        assert snapshot.memories
        assert snapshot.self_model.observation_count == 1
        assert snapshot.motivation.competence > 0.42


def test_module_revision_is_proposed_and_gated() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        cognitive = CognitiveSystem(core)
        baseline = cognitive.register_module(
            "attention", "Rank salient signals using calibrated priority.", {"priority_scale": 1.0}
        )
        cases = [
            EvaluationCase(f"attention-{i}", "module-attention", f"signal-{i}", "rank")
            for i in range(1, 6)
        ]

        decision = cognitive.propose_module_revision(
            "attention",
            cases,
            proposer=lambda current, _: StrategyConfig(
                "attention-v2",
                "module-attention",
                "Use calibrated salience and preserve uncertainty.",
                version=current.version + 1,
                parameters={"priority_scale": 0.95},
            ),
            evaluator=lambda strategy, case: (
                0.9 if strategy.strategy_id == "attention-v2" else 0.6
            ),
        )
        assert decision.accepted
        assert store.get_strategy("attention-v2").status == "active"
        assert baseline.status == "active"
        # An unobserved promoted strategy must actually be tried, not hidden by old metrics.
        assert core.choose_strategy("module-attention").config.strategy_id == "attention-v2"


def test_rejected_module_revision_keeps_old_strategy_active() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        cognitive = CognitiveSystem(core)
        baseline = cognitive.register_module("memory", "Recall validated experiences.")
        cases = [
            EvaluationCase(f"memory-{i}", "module-memory", f"query-{i}", "recall")
            for i in range(1, 6)
        ]
        decision = cognitive.propose_module_revision(
            "memory",
            cases,
            proposer=lambda current, _: StrategyConfig(
                "memory-regression",
                "module-memory",
                "A worse memory strategy.",
                version=current.version + 1,
            ),
            evaluator=lambda strategy, case: (
                0.2 if strategy.strategy_id == "memory-regression" else 0.7
            ),
        )
        assert not decision.accepted
        assert core.choose_strategy("module-memory").config.strategy_id == baseline.config.strategy_id
        assert store.get_strategy("memory-regression").status == "candidate"


def test_duplicate_episode_is_not_overwritten() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        cognitive = CognitiveSystem(core)
        cognitive.memory.remember(
            "math", "First validated lesson.", episode_id="lesson-1",
            strategy_id=baseline.config.strategy_id, score=0.8, success=True,
        )
        with pytest.raises(ValueError, match="could not be inserted"):
            cognitive.memory.remember(
                "math", "Different lesson.", episode_id="lesson-1",
                strategy_id=baseline.config.strategy_id, score=0.2, success=False,
            )
        assert store.search_episodes(task_type="math")[0]["summary"] == "First validated lesson."


def test_untrusted_memory_metadata_cannot_contain_non_finite_values() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        cognitive = CognitiveSystem(core)
        with pytest.raises(ValueError):
            cognitive.memory.remember(
                "math", "Invalid score metadata.", episode_id="lesson-invalid",
                strategy_id=baseline.config.strategy_id, score=0.8, success=True,
                metadata={"bad": float("nan")},
            )
        assert store.search_episodes(task_type="math") == []


def test_self_model_failure_history_survives_unrelated_event_noise() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("reasoning")
        cognitive = CognitiveSystem(core)
        store.record_event("task-execution-failed", "reasoning", {"error_type": "KnownFailure"})
        # Another task's failures are in the same event streams and must not
        # crowd this task's failure telemetry out of the bounded query.
        for index in range(250):
            store.record_event("task-execution-failed", "another-task", {"index": index})

        report = cognitive.self_model.report("reasoning")
        assert "task-execution-failed" in report.recent_failure_events


@pytest.mark.parametrize("bad_kwargs, message", [
    ({"summary": ""}, "summary"),
    ({"summary": "x" * 10_001}, "summary"),
    ({"tags": "not-a-list"}, "tags"),
    ({"tags": ["valid", []]}, "each tag"),
    ({"metadata": []}, "metadata"),
    ({"metadata": {"uncertainty": float("nan")}}, "score"),
    ({"metadata": {"contradiction": 1.5}}, "score"),
    ({"metadata": {"novelty": "high"}}, "score"),
    ({"score": float("nan")}, "score"),
    ({"success": 1}, "success"),
])
def test_invalid_outcome_is_rejected_before_any_feedback_is_written(
    bad_kwargs: dict, message: str
) -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("reasoning")
        cognitive = CognitiveSystem(core)
        kwargs = {
            "task_type": "reasoning",
            "strategy_id": baseline.config.strategy_id,
            "summary": "A valid measured result.",
            "tags": ["checked"],
            "score": 0.8,
            "success": True,
            "metadata": {"uncertainty": 0.2, "contradiction": 0.0, "novelty": 0.1},
        }
        kwargs.update(bad_kwargs)

        with pytest.raises(ValueError, match=message):
            cognitive.learn_from_outcome(**kwargs)

        assert store.total_observations("reasoning") == 0
        assert store.search_episodes(task_type="reasoning") == []


def test_invalid_feedback_for_unknown_strategy_has_no_persisted_episode() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("reasoning")
        cognitive = CognitiveSystem(core)
        with pytest.raises(KeyError, match="unknown strategy"):
            cognitive.learn_from_outcome(
                task_type="reasoning",
                strategy_id="missing-strategy",
                summary="This must not be persisted as a memory.",
                score=0.8,
                success=True,
            )
        assert store.total_observations("reasoning") == 0
        assert store.search_episodes(task_type="reasoning") == []


def test_outcome_learning_rolls_back_all_persistent_changes_on_late_failure() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("atomic-feedback")
        cognitive = CognitiveSystem(core)
        affect_before = cognitive.affect.state()
        drive_before = cognitive.motivation.state()
        model_before = cognitive.affect.model_snapshot()

        def fail_late(**kwargs):
            raise RuntimeError("simulated late motivation failure")

        cognitive.motivation.update_outcome = fail_late
        with pytest.raises(RuntimeError, match="simulated late motivation failure"):
            cognitive.learn_from_outcome(
                task_type="atomic-feedback",
                strategy_id=baseline.config.strategy_id,
                summary="A feedback update with a simulated late failure.",
                score=0.3,
                success=False,
                metadata={"uncertainty": 0.8, "novelty": 0.3},
            )

        assert store.total_observations("atomic-feedback") == 0
        assert store.search_episodes(task_type="atomic-feedback") == []
        assert cognitive.affect.state() == affect_before
        assert cognitive.motivation.state() == drive_before
        assert cognitive.affect.model_snapshot() == model_before
        event_types = {event["event_type"] for event in store.recent_events(limit=100)}
        assert "episode-recorded" not in event_types
        assert "affect-state-updated" not in event_types
        assert "affect-model-trained" not in event_types
        assert "motivation-state-updated" not in event_types
        assert cognitive.workspace.snapshot() == []
