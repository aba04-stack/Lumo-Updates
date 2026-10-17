from __future__ import annotations

import pytest

from adaptive_core import AdaptiveCore, EvaluationCase, EvolutionManager, Store, StrategyConfig


def cases():
    return [
        EvaluationCase(f"proposal-{i}", "reasoning", f"input-{i}", f"target-{i}")
        for i in range(1, 6)
    ]


def test_proposed_revision_must_pass_gate_before_activation() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("reasoning")
        manager = EvolutionManager(core)
        fixed_cases = cases()

        def proposer(current, feedback):
            assert current.strategy_id == baseline.config.strategy_id
            assert feedback["task_type"] == "reasoning"
            assert feedback["guardrails"]["source_code_execution"] is False
            return StrategyConfig(
                "reasoning-v2",
                "reasoning",
                current.instructions + " Check edge cases and verify the result.",
                version=current.version + 1,
            )

        def evaluator(strategy, case):
            return 0.9 if strategy.strategy_id == "reasoning-v2" else 0.5

        decision = manager.propose_and_evaluate(
            "reasoning", fixed_cases, proposer=proposer, evaluator=evaluator
        )

        assert decision.accepted
        assert store.get_strategy("reasoning-v2").status == "active"
        assert any(
            event["event_type"] == "candidate-promoted"
            for event in store.recent_events()
        )


def test_invalid_proposal_is_rejected_before_registration() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        baseline = core.create_baseline("reasoning")
        manager = EvolutionManager(core)

        with pytest.raises(ValueError, match="version"):
            manager.propose_and_evaluate(
                "reasoning",
                cases(),
                proposer=lambda current, _: StrategyConfig(
                    "bad-version",
                    "reasoning",
                    current.instructions,
                    version=current.version,
                ),
                evaluator=lambda *_: 1.0,
            )

        assert len(store.list_strategies("reasoning")) == 1
        assert store.recent_events()[0]["event_type"] == "candidate-proposal-failed"
