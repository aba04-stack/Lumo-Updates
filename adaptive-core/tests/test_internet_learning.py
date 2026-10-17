from __future__ import annotations

import pytest

from adaptive_core import (
    AdaptiveCore,
    CognitiveSystem,
    InternetLearningError,
    Store,
    WebKnowledgeStore,
    WikipediaLearner,
)


def mock_wikipedia(learner: WikipediaLearner) -> None:
    def request_json(language, params):
        if params.get("list") == "search":
            return {
                "query": {
                    "search": [{"pageid": 42, "title": "Искусственный интеллект"}]
                }
            }
        return {
            "query": {
                "pages": {
                    "42": {
                        "pageid": 42,
                        "title": "Искусственный интеллект",
                        "fullurl": f"https://{language}.wikipedia.org/wiki/Искусственный_интеллект",
                        "extract": (
                            "Искусственный интеллект — область информатики. "
                            "Системы машинного обучения находят закономерности в данных. "
                            "Результаты требуют независимой проверки."
                        ),
                    }
                }
            }
        }

    learner._request_json = request_json


def test_wikipedia_learning_is_persistent_and_source_linked(tmp_path) -> None:
    path = tmp_path / "web.sqlite3"
    learner = WikipediaLearner(path)
    mock_wikipedia(learner)

    learned = learner.learn("искусственный интеллект", language="ru", limit=1)

    assert len(learned) == 1
    assert learned[0]["trust_level"] == "unverified_external_source"
    assert learned[0]["url"].startswith("https://ru.wikipedia.org/")
    assert len(learned[0]["content_sha256"]) == 64
    learner.close()

    with WebKnowledgeStore(path) as store:
        recalled = store.search("машинного обучения")
    assert len(recalled) == 1
    assert "независимой проверки" in recalled[0]["content"]


def test_wikipedia_network_access_is_explicit_and_cognitive_integration_works(tmp_path) -> None:
    learner = WikipediaLearner(tmp_path / "web.sqlite3")
    mock_wikipedia(learner)
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("research")
        cognitive = CognitiveSystem(core, internet=learner)
        learned = cognitive.learn_from_internet("искусственный интеллект", limit=1)
        snapshot = cognitive.reflect("research", web_query="машинного обучения")

        assert len(learned) == 1
        assert len(snapshot.web_documents) == 1
        assert any("UNVERIFIED EXTERNAL SOURCE" in item.content for item in snapshot.workspace)
        assert snapshot.web_documents[0]["url"] == learned[0]["url"]
    learner.close()


def test_invalid_language_and_query_are_rejected_without_network(tmp_path) -> None:
    learner = WikipediaLearner(tmp_path / "web.sqlite3")
    with pytest.raises(ValueError, match="query"):
        learner.learn(" ")
    with pytest.raises(ValueError, match="unsupported language"):
        learner.learn("test", language="../example")
    learner.close()


def test_store_rejects_non_wikipedia_urls_and_oversized_pages(tmp_path) -> None:
    with WebKnowledgeStore(tmp_path / "web.sqlite3") as store:
        with pytest.raises(ValueError, match="source URL"):
            store.save_document(
                title="Unsafe", url="https://example.com/page", language="ru",
                query="test", content="Untrusted content",
            )
        with pytest.raises(ValueError, match="content"):
            store.save_document(
                title="Too large",
                url="https://ru.wikipedia.org/wiki/Too_large",
                language="ru",
                query="test",
                content="x" * 20_001,
            )


def test_api_errors_are_reported_without_saving_fake_knowledge(tmp_path) -> None:
    learner = WikipediaLearner(tmp_path / "web.sqlite3")
    learner._request_json = lambda language, params: (_ for _ in ()).throw(
        InternetLearningError("network down")
    )
    with pytest.raises(InternetLearningError):
        learner.learn("some topic")
    assert learner.recall() == []
    learner.close()
