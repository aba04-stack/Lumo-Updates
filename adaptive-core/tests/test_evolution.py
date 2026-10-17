from __future__ import annotations

import pytest

from adaptive_core import (
    AdaptiveCore,
    EvaluationCase,
    EvolutionManager,
    Store,
    StrategyConfig,
)


def build_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(f"case-{index}", "math", f"input-{index}", f"answer-{index}")
        for index in range(1, 6)
    ]


def make_candidate(core: AdaptiveCore, parent_id: str, strategy_id: str = "candidate-v1"):
    parent = core.store.get_strategy(parent_id)
    return core.register_candidate(
        StrategyConfig(
            strategy_id, "math", "Candidate configuration, not executable code.",
            version=parent.config.version + 1,
        ),
        parent_id=parent_id,
    )


def test_candidate_promotes_only_after_common_set_improves() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        candidate = make_candidate(core, baseline.config.strategy_id)
        cases = build_cases()
        scores = {
            baseline.config.strategy_id: {case.case_id: 0.55 for case in cases},
            candidate.config.strategy_id: {case.case_id: 0.82 for case in cases},
        }

        decision = EvolutionManager(core).evaluate_candidate(
            candidate.config.strategy_id,
            cases,
            lambda strategy, case: scores[strategy.strategy_id][case.case_id],
        )

        assert decision.accepted
        assert decision.cases_evaluated == 5
        assert decision.candidate_score > decision.baseline_score
        assert store.get_strategy(candidate.config.strategy_id).status == "active"


def test_large_regression_blocks_promotion_even_when_mean_improves() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        candidate = make_candidate(core, baseline.config.strategy_id)
        cases = build_cases()
        scores = {
            baseline.config.strategy_id: {case.case_id: 0.60 for case in cases},
            candidate.config.strategy_id: {
                cases[0].case_id: 0.20,
                cases[1].case_id: 0.90,
                cases[2].case_id: 0.90,
                cases[3].case_id: 0.90,
                cases[4].case_id: 0.90,
            },
        }

        decision = EvolutionManager(core).evaluate_candidate(
            candidate.config.strategy_id,
            cases,
            lambda strategy, case: scores[strategy.strategy_id][case.case_id],
        )

        assert not decision.accepted
        assert decision.max_case_regression == pytest.approx(0.40)
        assert store.get_strategy(candidate.config.strategy_id).status == "candidate"


def test_evaluator_failure_fails_closed() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        candidate = make_candidate(core, baseline.config.strategy_id)
        cases = build_cases()

        def broken_evaluator(strategy, case):
            if strategy.strategy_id == candidate.config.strategy_id:
                raise RuntimeError("evaluator unavailable")
            return 0.5

        decision = EvolutionManager(core).evaluate_candidate(
            candidate.config.strategy_id, cases, broken_evaluator
        )

        assert not decision.accepted
        assert "failed closed" in decision.reason
        assert store.get_strategy(candidate.config.strategy_id).status == "candidate"


def test_too_few_or_duplicate_cases_cannot_promote() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        candidate = make_candidate(core, baseline.config.strategy_id)
        manager = EvolutionManager(core)
        cases = build_cases()

        too_few = manager.evaluate_candidate(candidate.config.strategy_id, cases[:2], lambda *_: 1.0)
        assert not too_few.accepted

        duplicate_cases = [cases[0], cases[0], *cases[2:]]
        duplicate = manager.evaluate_candidate(
            candidate.config.strategy_id, duplicate_cases, lambda *_: 1.0
        )
        assert not duplicate.accepted
        assert store.get_strategy(candidate.config.strategy_id).status == "candidate"


def test_candidate_with_invalid_score_cannot_promote() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        candidate = make_candidate(core, baseline.config.strategy_id)
        decision = EvolutionManager(core).evaluate_candidate(
            candidate.config.strategy_id, build_cases(), lambda *_: float("nan")
        )
        assert not decision.accepted
        assert store.get_strategy(candidate.config.strategy_id).status == "candidate"


def test_proposal_feedback_keeps_task_history_despite_unrelated_event_noise() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("math")
        store.record_event("domain-failure-recorded", "math", {"reason": "wrong answer"})
        for index in range(250):
            store.record_event("unrelated-noise", "another-task", {"index": index})
        captured = []
        manager = EvolutionManager(core)
        cases = build_cases()

        decision = manager.propose_and_evaluate(
            "math",
            cases,
            proposer=lambda current, feedback: (
                captured.append(feedback) or StrategyConfig(
                    "candidate-after-noise",
                    "math",
                    "Check evidence and arithmetic before responding.",
                    version=current.version + 1,
                )
            ),
            evaluator=lambda strategy, case: (
                0.9 if strategy.strategy_id == "candidate-after-noise" else 0.6
            ),
        )

        assert decision.accepted
        assert captured
        assert "domain-failure-recorded" in {
            event["event_type"] for event in captured[0]["recent_event_types"]
        }


def test_proposal_audit_failure_rolls_back_candidate_creation() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("math")
        manager = EvolutionManager(core)
        record_event = store.record_event

        def fail_proposal_event(event_type, task_type, payload):
            if event_type == "candidate-proposed":
                raise RuntimeError("simulated audit failure")
            return record_event(event_type, task_type, payload)

        store.record_event = fail_proposal_event
        with pytest.raises(RuntimeError, match="simulated audit failure"):
            manager.propose_and_evaluate(
                "math",
                build_cases(),
                proposer=lambda current, _: StrategyConfig(
                    "candidate-audit-rollback",
                    "math",
                    "A candidate should be rolled back if its proposal audit fails.",
                    version=current.version + 1,
                ),
                evaluator=lambda *_: 0.9,
            )

        assert store.list_strategies("math", status="candidate") == []
        event_types = {event["event_type"] for event in store.events_for_task("math", limit=50)}
        assert "candidate-proposed" not in event_types
        assert not any(
            event["payload"].get("strategy_id") == "candidate-audit-rollback"
            for event in store.events_for_task("math", limit=50)
            if event["event_type"] == "strategy-created"
        )


def test_restore_audit_failure_keeps_strategy_disabled() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("restore-audit")
        candidate = core.register_candidate(
            StrategyConfig(
                "restore-candidate", "restore-audit", "Audited version.",
                version=baseline.config.version + 1,
            ),
            parent_id=baseline.config.strategy_id,
        )
        cases = [
            EvaluationCase(f"restore-case-{i}", "restore-audit", f"input-{i}", "expected")
            for i in range(1, 6)
        ]
        decision = EvolutionManager(core).evaluate_candidate(
            candidate.config.strategy_id, cases,
            lambda strategy, case: 0.9 if strategy.strategy_id == candidate.config.strategy_id else 0.6,
        )
        assert decision.accepted
        manager = EvolutionManager(core)
        manager.rollback(candidate.config.strategy_id)
        assert store.get_strategy(candidate.config.strategy_id).status == "disabled"

        record_event = store.record_event

        def fail_restore_event(event_type, task_type, payload):
            if event_type == "strategy-restored":
                raise RuntimeError("simulated restore audit failure")
            return record_event(event_type, task_type, payload)

        store.record_event = fail_restore_event
        with pytest.raises(RuntimeError, match="simulated restore audit failure"):
            manager.restore(candidate.config.strategy_id)

        assert store.get_strategy(candidate.config.strategy_id).status == "disabled"
