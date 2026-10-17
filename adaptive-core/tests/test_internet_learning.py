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


def test_malformed_search_payload_fails_closed(tmp_path) -> None:
    learner = WikipediaLearner(tmp_path / "web.sqlite3")
    learner._request_json = lambda language, params: {"query": []}
    with pytest.raises(InternetLearningError, match="search-result structure"):
        learner.learn("claim")
    assert learner.recall() == []
    learner.close()


def test_malformed_search_entries_are_skipped_without_crashing(tmp_path) -> None:
    learner = WikipediaLearner(tmp_path / "web.sqlite3")

    def request_json(language, params):
        if params.get("list") == "search":
            return {
                "query": {
                    "search": [
                        None,
                        42,
                        {"pageid": []},
                        {"pageid": "../bad"},
                        {"pageid": 42},
                        {"pageid": "42"},
                    ]
                }
            }
        return {
            "query": {
                "pages": {
                    "42": {
                        "title": "Valid page",
                        "fullurl": f"https://{language}.wikipedia.org/wiki/Valid_page",
                        "extract": "This is a valid retrieved page with sufficient text.",
                    }
                }
            }
        }

    learner._request_json = request_json
    result = learner.learn("valid page", limit=4)
    assert len(result) == 1
    assert result[0]["title"] == "Valid page"
    learner.close()


def test_bad_language_types_timeout_and_url_values_fail_as_validation_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="timeout"):
        WikipediaLearner(tmp_path / "bad-timeout.sqlite3", timeout=True)

    learner = WikipediaLearner(tmp_path / "web.sqlite3")
    with pytest.raises(ValueError, match="unsupported language"):
        learner.learn("topic", language=[])
    with pytest.raises(ValueError, match="unsupported Wikipedia language"):
        learner._request_json("../bad", {})
    with WebKnowledgeStore(tmp_path / "web2.sqlite3") as store:
        with pytest.raises(ValueError, match="source URL"):
            store.save_document(
                title="bad", url=None, language="ru", query="topic", content="some text"
            )
        with pytest.raises(ValueError, match="unsupported Wikipedia language"):
            store.save_document(
                title="bad", url="https://ru.wikipedia.org/wiki/Valid",
                language=[], query="topic", content="some text",
            )
    learner.close()


def test_web_knowledge_store_serializes_concurrent_writes_to_one_connection(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with WebKnowledgeStore(tmp_path / "parallel-web.sqlite3") as store:
        def save(index: int):
            return store.save_document(
                title=f"Page {index}",
                url=f"https://ru.wikipedia.org/wiki/Page_{index}",
                language="ru",
                query="parallel",
                content=f"This is source content for distinct page number {index}.",
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            saved = list(pool.map(save, range(16)))

        assert len(saved) == 16
        recalled = store.search("source content", limit=20)
        assert len(recalled) == 16
        assert {item["url"] for item in recalled} == {item["url"] for item in saved}
