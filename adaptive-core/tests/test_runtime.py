from __future__ import annotations

import pytest

from adaptive_core import AdaptiveCore, Store
from adaptive_core.runtime import AdaptiveRunner, Assessment


def test_runner_closes_feedback_loop() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("summarize")
        runner = AdaptiveRunner(core)

        result = runner.run(
            "summarize",
            "short text",
            executor=lambda strategy, payload: f"{strategy.strategy_id}:{payload}",
            grader=lambda strategy, payload, output: Assessment(
                0.9, True, {"checks": ["length", "coverage"]}
            ),
        )

        assert result.strategy_id == baseline.config.strategy_id
        assert result.output == "baseline:summarize:short text"
        assert store.total_observations("summarize") == 1
        assert store.stats("summarize")[baseline.config.strategy_id]["avg_score"] == 0.9
        assert store.recent_events()[0]["event_type"] == "task-completed"


def test_broken_grader_does_not_poison_adaptation() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("coding")
        runner = AdaptiveRunner(core)

        with pytest.raises(RuntimeError, match="grader outage"):
            runner.run(
                "coding",
                "write a function",
                executor=lambda strategy, payload: "def answer(): return 1",
                grader=lambda *_: (_ for _ in ()).throw(RuntimeError("grader outage")),
            )

        assert store.total_observations("coding") == 0
        assert store.recent_events()[0]["event_type"] == "task-assessment-failed"


def test_executor_failure_is_logged_without_fake_score() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("coding")
        runner = AdaptiveRunner(core)

        def broken_executor(strategy, payload):
            raise OSError("tool unavailable")

        with pytest.raises(OSError):
            runner.run(
                "coding",
                "write a function",
                executor=broken_executor,
                grader=lambda *_: Assessment(1.0, True),
            )

        assert store.total_observations("coding") == 0
        assert store.recent_events()[0]["event_type"] == "task-execution-failed"


def test_success_score_rolls_back_if_completion_event_fails() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("atomic-runtime")
        runner = AdaptiveRunner(core)
        original_record_event = store.record_event

        def fail_completion_event(event_type, task_type, payload):
            if event_type == "task-completed":
                raise RuntimeError("completion audit unavailable")
            return original_record_event(event_type, task_type, payload)

        store.record_event = fail_completion_event
        with pytest.raises(RuntimeError, match="completion audit unavailable"):
            runner.run(
                "atomic-runtime",
                "test payload",
                executor=lambda strategy, payload: "ok",
                grader=lambda strategy, payload, output: Assessment(0.9, True),
            )

        assert store.total_observations("atomic-runtime") == 0
        assert not any(
            event["event_type"] == "task-completed"
            for event in store.events_for_task("atomic-runtime", limit=20)
        )
        assert store.get_strategy(baseline.config.strategy_id).status == "active"
