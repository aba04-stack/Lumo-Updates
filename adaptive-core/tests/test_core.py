from __future__ import annotations

import pytest

from adaptive_core import AdaptiveCore, EvaluationCase, EvolutionManager, NoActiveStrategyError, Store, StrategyConfig


def promote_candidate_for_test(core, baseline, strategy_id: str, task_type: str, instructions: str):
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
        lambda strategy, case: 0.92 if strategy.strategy_id == strategy_id else 0.20,
    )
    assert decision.accepted, decision.reason
    return core.store.get_strategy(strategy_id)


def test_core_learns_from_measured_feedback_and_persists(tmp_path) -> None:
    db_path = tmp_path / "adaptive.sqlite3"
    store = Store(db_path)
    core = AdaptiveCore(store)
    baseline = core.create_baseline("math")
    challenger = promote_candidate_for_test(
        core, baseline, "challenger", "math", "Check the arithmetic twice."
    )

    for _ in range(30):
        core.record_result(baseline.config.strategy_id, 0.20, False)
        core.record_result(challenger.config.strategy_id, 0.92, True)

    assert core.choose_strategy("math").config.strategy_id == "challenger"
    store.close()

    with Store(db_path) as reopened:
        assert reopened.total_observations("math") == 60
        assert reopened.stats("math")["challenger"]["avg_score"] == pytest.approx(0.92)
        assert AdaptiveCore(reopened).choose_strategy("math").config.strategy_id == "challenger"


def test_candidate_is_not_used_before_promotion() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("coding")
        candidate = core.register_candidate(
            StrategyConfig("candidate-code", "coding", "Use smaller, testable changes.")
        )
        assert candidate.status == "candidate"
        assert core.choose_strategy("coding").config.strategy_id == baseline.config.strategy_id
        with pytest.raises(ValueError, match="only active"):
            core.record_result(candidate.config.strategy_id, 1.0, True)


def test_no_active_strategy_is_reported() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        with pytest.raises(NoActiveStrategyError):
            core.choose_strategy("unknown")


def test_invalid_scores_are_rejected() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        for bad_score in (-0.1, 1.1, float("nan"), float("inf"), True):
            with pytest.raises(ValueError):
                core.record_result(baseline.config.strategy_id, bad_score, True)


def test_disabling_best_strategy_keeps_fallback() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        better = promote_candidate_for_test(
            core, baseline, "better", "math", "Improved calculation strategy."
        )
        for _ in range(20):
            core.record_result(baseline.config.strategy_id, 0.25, False)
            core.record_result(better.config.strategy_id, 0.95, True)
        from adaptive_core import EvolutionManager

        EvolutionManager(core).rollback(better.config.strategy_id)
        assert core.choose_strategy("math").config.strategy_id == baseline.config.strategy_id
        with pytest.raises(ValueError, match="last active"):
            EvolutionManager(core).rollback(baseline.config.strategy_id)


def test_strategy_ids_are_immutable() -> None:
    with Store(":memory:") as store:
        config = StrategyConfig("same-id", "math", "Original")
        store.create_strategy(config, status="active")
        with pytest.raises(ValueError, match="could not be inserted"):
            store.create_strategy(config, status="candidate")


def test_second_active_strategy_must_pass_evaluation_gate() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("protected-activation")
        with pytest.raises(ValueError, match="pass fixed-case evaluation"):
            store.create_strategy(
                StrategyConfig("unchecked-active", "protected-activation", "Bypass evaluation."),
                status="active",
            )
        assert store.get_strategy(baseline.config.strategy_id).status == "active"
        assert store.list_strategies("protected-activation", status="candidate") == []
