from __future__ import annotations
from adaptive_core import AdaptiveCore, CognitiveSystem, Store

class FakeInternet:
    def __init__(self) -> None:
        self.documents = [{
            "title": "Теория обучения",
            "url": "https://ru.wikipedia.org/wiki/Теория_обучения",
            "content": ("Обучение помогает системе менять поведение на основе обратной связи. "
                        "Обучение требует проверки результата и повторного опыта."),
            "content_sha256": "a" * 64, "trust_level": "unverified_external_source", "language": "ru",
        }]
    def learn(self, query: str, *, language: str = "ru", limit: int = 2):
        return self.documents[:limit]
    def recall(self, query: str = "", *, limit: int = 5):
        return self.documents[:limit]

def test_autonomous_loop_runs_bounded_cycle_and_keeps_caveats() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("autonomous-research")
        cognitive = CognitiveSystem(core, internet=FakeInternet())
        loop = cognitive.autonomous_learning_loop(topics=["обучение"], pages_per_topic=1)
        result = loop.run(max_steps=1, page_budget=1)
        assert result["steps_completed"] == 1
        assert result["documents_found"] == 1
        step = result["results"][0]
        assert step["topic"] == "обучение"
        assert step["status"] == "source_found_unverified"
        assert step["retrieval_utility_is_truth_score"] is False
        assert "Keyword overlap" in step["caveat"]
        assert result["affect"]["curiosity"] >= 0.0
        assert result["affect_model_samples"] == 1
        assert result["results"][0]["goal_status"] == "needs_verification"
        assert result["results"][0]["claims_independently_verified"] is False
        assert step["claim_extraction_method"] == "sentence_heuristic_v1"
        assert step["claim_candidates"]
        assert all(claim["independently_verified"] is False for claim in step["claim_candidates"])
        assert step["potential_contradictions"] == []
        assert step["contradiction_proxy"] == 0.0
        assert result["motivation"]["uncertainty_reduction"] > 0.45
        assert len(result["goals_needing_verification"]) == 1
        assert result["goals_needing_verification"][0]["title"] == "обучение"
        # Finding a page must not be treated as answering or verifying the goal.
        assert loop._choose_topic(set()) != "обучение"
        assert store.search_episodes(task_type="autonomous-research")
        assert any(event["event_type"] == "autonomous-research-step" for event in store.recent_events())
        assert any(event["event_type"] == "affect-model-trained" for event in store.recent_events())
        assert any(event["event_type"] == "motivation-state-updated" for event in store.recent_events())
        assert any(event["event_type"] == "motivation-goal-status" for event in store.recent_events())

def test_autonomous_loop_rejects_unbounded_run_requests() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("autonomous-research")
        loop = CognitiveSystem(core, internet=FakeInternet()).autonomous_learning_loop(topics=["обучение"])
        try:
            loop.run(max_steps=10000)
        except ValueError as exc:
            assert "max_steps" in str(exc)
        else:
            raise AssertionError("unbounded request should be rejected")


def test_curiosity_can_prefer_a_newly_discovered_topic() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("autonomous-research")
        loop = CognitiveSystem(core, internet=FakeInternet()).autonomous_learning_loop(
            topics=["initial-topic"]
        )
        loop.queue_topic("related-topic", priority=0.35, reason="derived_from_new_material")
        # Under baseline affect, curiosity makes a newly discovered topic win.
        assert loop._choose_topic(set()) == "related-topic"

def test_irrelevant_documents_score_below_lexically_relevant_documents() -> None:
    class IrrelevantInternet(FakeInternet):
        def __init__(self) -> None:
            self.documents = [{
                "title": "Музыкальные инструменты",
                "url": "https://ru.wikipedia.org/wiki/Музыкальные_инструменты",
                "content": "Скрипка и виолончель относятся к струнным музыкальным инструментам.",
                "content_sha256": "b" * 64,
                "trust_level": "unverified_external_source",
                "language": "ru",
            }]

    with Store(":memory:") as relevant_store:
        core = AdaptiveCore(relevant_store)
        core.create_baseline("autonomous-research")
        relevant_loop = CognitiveSystem(core, internet=FakeInternet()).autonomous_learning_loop(
            topics=["обучение"], pages_per_topic=1
        )
        relevant = relevant_loop.run(max_steps=1, page_budget=1)["results"][0]

    with Store(":memory:") as irrelevant_store:
        core = AdaptiveCore(irrelevant_store)
        core.create_baseline("autonomous-research")
        irrelevant_loop = CognitiveSystem(core, internet=IrrelevantInternet()).autonomous_learning_loop(
            topics=["обучение"], pages_per_topic=1
        )
        irrelevant = irrelevant_loop.run(max_steps=1, page_budget=1)["results"][0]

    assert relevant["relevance_coverage_proxy"] > irrelevant["relevance_coverage_proxy"]
    assert relevant["retrieval_utility"] > irrelevant["retrieval_utility"]
    assert irrelevant["claims_independently_verified"] is False


def test_autonomous_loop_surfaces_but_does_not_overstate_contradiction_candidates() -> None:
    class ContradictoryInternet(FakeInternet):
        def __init__(self) -> None:
            self.documents = [
                {
                    "title": "Affirmative",
                    "url": "https://ru.wikipedia.org/wiki/Claim_A",
                    "content": "Earth is round.",
                    "content_sha256": "c" * 64,
                    "trust_level": "unverified_external_source",
                    "language": "en",
                },
                {
                    "title": "Negated",
                    "url": "https://ru.wikipedia.org/wiki/Claim_B",
                    "content": "Earth is not round.",
                    "content_sha256": "d" * 64,
                    "trust_level": "unverified_external_source",
                    "language": "en",
                },
            ]

    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("autonomous-research")
        loop = CognitiveSystem(core, internet=ContradictoryInternet()).autonomous_learning_loop(
            topics=["Earth round"], pages_per_topic=2
        )
        result = loop.run(max_steps=1, page_budget=2)["results"][0]

    assert len(result["claim_candidates"]) == 2
    assert len(result["potential_contradictions"]) == 1
    assert result["contradiction_proxy"] > 0.0
    assert all(item["independently_verified"] is False for item in result["potential_contradictions"])


def test_autonomous_loop_bounds_initial_topics_and_rejects_a_single_string() -> None:
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("autonomous-research")
        cognitive = CognitiveSystem(core, internet=FakeInternet())
        try:
            cognitive.autonomous_learning_loop(topics="not-a-topic-list")
        except ValueError as exc:
            assert "iterable of topic strings" in str(exc)
        else:
            raise AssertionError("a string must not be expanded into individual topics")

        def endless_topics():
            index = 0
            while True:
                index += 1
                yield f"topic-{index}"

        try:
            cognitive.autonomous_learning_loop(topics=endless_topics())
        except ValueError as exc:
            assert "at most 100 initial topics" in str(exc)
        else:
            raise AssertionError("unbounded topic generators must be rejected")
