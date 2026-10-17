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
        assert store.search_episodes(task_type="autonomous-research")
        assert any(event["event_type"] == "autonomous-research-step" for event in store.recent_events())

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
