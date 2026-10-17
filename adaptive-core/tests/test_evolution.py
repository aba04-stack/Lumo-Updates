from __future__ import annotations

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
    return core.register_candidate(
        StrategyConfig(strategy_id, "math", "Candidate configuration, not executable code."),
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
        assert decision.max_case_regression == 0.40
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
