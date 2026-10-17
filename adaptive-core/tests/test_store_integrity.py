from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from adaptive_core import AdaptiveCore, EvaluationCase, EvolutionManager, Store, StrategyConfig


def _promote_second_strategy(core, baseline, strategy_id: str, task_type: str, instructions: str):
    candidate = core.register_candidate(
        StrategyConfig(
            strategy_id, task_type, instructions,
            version=baseline.config.version + 1,
        ),
        parent_id=baseline.config.strategy_id,
    )
    cases = [
        EvaluationCase(f"{strategy_id}-case-{i}", task_type, f"fixed-input-{i}", "expected")
        for i in range(1, 6)
    ]
    decision = EvolutionManager(core).evaluate_candidate(
        strategy_id,
        cases,
        lambda strategy, case: 0.95 if strategy.strategy_id == strategy_id else 0.60,
    )
    assert decision.accepted, decision.reason
    return core.store.get_strategy(strategy_id)


def test_event_store_rejects_non_object_and_non_json_payloads() -> None:
    with Store(":memory:") as store:
        before = len(store.recent_events())
        with pytest.raises(ValueError, match="dictionary"):
            store.record_event("bad-list", "test", ["not", "a", "mapping"])
        with pytest.raises(ValueError, match="JSON-serializable"):
            store.record_event("bad-object", "test", {"value": object()})
        with pytest.raises(ValueError, match="JSON-serializable"):
            store.record_event("bad-nan", "test", {"value": float("nan")})
        assert len(store.recent_events()) == before


def test_last_active_strategy_cannot_be_disabled_through_store_api() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("integrity")
        with pytest.raises(ValueError, match="last active strategy"):
            store.set_status(baseline.config.strategy_id, "disabled")
        assert store.get_strategy(baseline.config.strategy_id).status == "active"


def test_rollback_emits_auditable_transition_and_keeps_fallback(tmp_path) -> None:
    db = tmp_path / "rollback.sqlite3"
    with Store(db) as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("integrity")
        challenger = _promote_second_strategy(
            core, baseline, "integrity-challenger", "integrity", "Candidate."
        )
        EvolutionManager(core).rollback(challenger.config.strategy_id)
        assert store.get_strategy(challenger.config.strategy_id).status == "disabled"
        assert store.get_strategy(baseline.config.strategy_id).status == "active"
        recent = store.recent_events(limit=5)
        types = [event["event_type"] for event in recent]
        assert "strategy-status-changed" in types
        assert "strategy-rollback" in types


def test_concurrent_rollbacks_never_disable_every_active_strategy(tmp_path) -> None:
    db = tmp_path / "concurrent-rollback.sqlite3"
    with Store(db) as setup:
        core = AdaptiveCore(setup)
        first = core.create_baseline("concurrent-integrity")
        second = _promote_second_strategy(
            core, first, "concurrent-second", "concurrent-integrity", "Second strategy."
        )
        ids = [first.config.strategy_id, second.config.strategy_id]

    start = Barrier(2)

    def rollback_one(strategy_id: str) -> str:
        with Store(db) as store:
            start.wait(timeout=5)
            try:
                EvolutionManager(AdaptiveCore(store)).rollback(strategy_id)
                return "rolled_back"
            except ValueError as exc:
                assert "last active strategy" in str(exc)
                return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(rollback_one, ids))

    assert sorted(outcomes) == ["blocked", "rolled_back"]
    with Store(db) as reopened:
        active = reopened.list_strategies("concurrent-integrity", status="active")
        assert len(active) == 1


def test_feedback_is_not_recorded_for_disabled_strategies() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        first = core.create_baseline("feedback-integrity")
        second = _promote_second_strategy(
            core, first, "feedback-fallback", "feedback-integrity", "Fallback."
        )
        EvolutionManager(core).rollback(first.config.strategy_id)
        with pytest.raises(ValueError, match="only active"):
            core.record_result(first.config.strategy_id, 0.9, True)
        assert store.total_observations("feedback-integrity") == 0
        assert store.get_strategy(second.config.strategy_id).status == "active"


def test_nested_savepoint_rolls_back_only_inner_writes() -> None:
    with Store(":memory:") as store:
        with store.transaction():
            store.record_event("outer-before", "tx-test", {"phase": 1})
            with pytest.raises(RuntimeError, match="inner failure"):
                with store.transaction():
                    store.record_event("inner-should-rollback", "tx-test", {"phase": 2})
                    raise RuntimeError("inner failure")
            store.record_event("outer-after", "tx-test", {"phase": 3})

        events = store.events_for_task("tx-test", limit=20, newest_first=False)
        types = [event["event_type"] for event in events]
        assert types == ["outer-before", "outer-after"]


def test_outer_rollback_cancels_nested_strategy_and_audit_event() -> None:
    with Store(":memory:") as store:
        with pytest.raises(RuntimeError, match="abort outer transaction"):
            with store.transaction():
                store.create_strategy(
                    StrategyConfig("nested-strategy", "nested-tx", "Should roll back."),
                    status="active",
                )
                raise RuntimeError("abort outer transaction")

        with pytest.raises(KeyError, match="nested-strategy"):
            store.get_strategy("nested-strategy")
        assert store.list_strategies("nested-tx") == []
        assert store.events_for_task("nested-tx", limit=20) == []


def test_event_payload_size_limit_rejects_oversized_metadata() -> None:
    with Store(":memory:") as store:
        with pytest.raises(ValueError, match="1,000,000-byte limit"):
            store.record_event("oversized", "test", {"large": "x" * 1_000_001})
        assert store.events_for_task("test", limit=10) == []
