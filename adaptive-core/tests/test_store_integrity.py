from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from adaptive_core import AdaptiveCore, EvolutionManager, Store, StrategyConfig


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
        challenger = store.create_strategy(
            StrategyConfig("integrity-challenger", "integrity", "Candidate."),
            status="active",
            parent_id=baseline.config.strategy_id,
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
        second = setup.create_strategy(
            StrategyConfig("concurrent-second", "concurrent-integrity", "Second strategy."),
            status="active",
            parent_id=first.config.strategy_id,
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
        second = store.create_strategy(
            StrategyConfig("feedback-fallback", "feedback-integrity", "Fallback."),
            status="active",
            parent_id=first.config.strategy_id,
        )
        EvolutionManager(core).rollback(first.config.strategy_id)
        with pytest.raises(ValueError, match="only active"):
            core.record_result(first.config.strategy_id, 0.9, True)
        assert store.total_observations("feedback-integrity") == 0
        assert store.get_strategy(second.config.strategy_id).status == "active"
