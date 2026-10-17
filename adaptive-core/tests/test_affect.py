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
