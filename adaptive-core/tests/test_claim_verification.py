from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from adaptive_core import AdaptiveCore, ClaimVerificationEngine, CognitiveSystem, Store


CLAIM = {
    "claim_id": "claim-001",
    "text": "A sample claim used only to exercise the assessment boundary.",
    "source_url": "https://original.example/source",
}
EVIDENCE = [
    {
        "title": "Independent publisher A",
        "url": "https://publisher-a.org/article",
        "content": "Supporting or refuting evidence from publisher A.",
        "content_sha256": "a" * 64,
        "fetched_at": "2026-10-17T10:00:00+00:00",
    },
    {
        "title": "Independent publisher B",
        "url": "https://publisher-b.net/article",
        "content": "Supporting or refuting evidence from publisher B.",
        "content_sha256": "b" * 64,
        "fetched_at": "2026-10-17T10:01:00+00:00",
    },
]


class StubBackend:
    backend_id = "stub-semantic-reviewer"

    def __init__(self, response: Any = None, error: bool = False) -> None:
        self.response = response or {
            "status": "supports_claim",
            "confidence": 0.91,
            "summary": "Both supplied publisher texts appear to support the claim.",
            "evidence_urls": [EVIDENCE[0]["url"], EVIDENCE[1]["url"]],
        }
        self.error = error
        self.calls = 0

    def assess_claim(
        self, claim: Mapping[str, Any], evidence: tuple[Mapping[str, Any], ...]
    ) -> Mapping[str, Any]:
        self.calls += 1
        if self.error:
            raise RuntimeError("provider unavailable")
        assert claim["text"]
        assert all(item["url"].startswith("https://") for item in evidence)
        return self.response


def test_backend_assessment_is_cited_but_never_promoted_to_verified_truth() -> None:
    backend = StubBackend()
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "supports_claim"
    assert result["provider_status"] == "supports_claim"
    assert result["confidence"] == 0.91
    assert result["evidence_urls"] == [item["url"] for item in EVIDENCE]
    assert result["publisher_domains"] == ["publisher-a.org", "publisher-b.net"]
    assert result["backend_id"] == backend.backend_id
    assert result["claim_verified"] is False
    assert result["goal_completion_allowed"] is False
    assert result["review_method"] == "pluggable_semantic_backend_v1"


def test_too_few_independent_publishers_prevents_backend_call() -> None:
    backend = StubBackend()
    evidence = [
        {
            "title": "English Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Claim",
            "content": "Evidence from one publisher.",
        },
        {
            "title": "Russian Wikipedia",
            "url": "https://ru.wikipedia.org/wiki/Claim",
            "content": "A second subdomain from the same publisher.",
        },
    ]
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, evidence)

    assert result["status"] == "insufficient_evidence"
    assert result["reason"] == "fewer_than_two_independent_publishers"
    assert backend.calls == 0
    assert result["goal_completion_allowed"] is False


def test_claim_source_publisher_is_not_counted_as_independent_support() -> None:
    evidence = [
        EVIDENCE[0],
        EVIDENCE[1],
        {
            "title": "Publisher C",
            "url": "https://publisher-c.com/article",
            "content": "Another independent text.",
        },
    ]
    backend = StubBackend({
        "status": "supports_claim",
        "confidence": 0.9,
        "summary": "Two independent eligible publishers appear supportive.",
        "evidence_urls": [EVIDENCE[1]["url"], evidence[2]["url"]],
    })
    claim = {**CLAIM, "source_url": "https://publisher-a.org/original"}
    result = ClaimVerificationEngine(backend).assess_claim(claim, evidence)

    assert result["status"] == "supports_claim"
    assert "publisher-a.org" not in result["publisher_domains"]
    assert result["publisher_domains"] == ["publisher-b.net", "publisher-c.com"]
    assert result["evidence_urls"] == [EVIDENCE[1]["url"], evidence[2]["url"]]
    assert result["claim_verified"] is False


def test_unsupported_citation_fails_closed() -> None:
    backend = StubBackend({
        "status": "supports_claim",
        "confidence": 0.95,
        "summary": "A confident assertion with an invented citation.",
        "evidence_urls": ["https://not-provided.example/ghost"],
    })
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "needs_human_review"
    assert result["reason"] == "citation_not_in_supplied_evidence"
    assert result["evidence_urls"] == []
    assert result["claim_verified"] is False


def test_verdict_with_only_one_cited_publisher_becomes_insufficient() -> None:
    backend = StubBackend({
        "status": "contradicts_claim",
        "confidence": 0.99,
        "summary": "The model reports contrary evidence.",
        "evidence_urls": [EVIDENCE[0]["url"]],
    })
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "insufficient_evidence"
    assert result["provider_status"] == "contradicts_claim"
    assert result["reason"] == "insufficient_independent_citations"
    assert result["claim_verified"] is False


def test_low_confidence_model_verdict_requires_human_review() -> None:
    backend = StubBackend({
        "status": "supports_claim",
        "confidence": 0.54,
        "summary": "Some evidence appears relevant but is not decisive.",
        "evidence_urls": [EVIDENCE[0]["url"], EVIDENCE[1]["url"]],
    })
    result = ClaimVerificationEngine(backend, min_confidence=0.75).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "needs_human_review"
    assert result["reason"] == "below_minimum_model_confidence"
    assert result["provider_status"] == "supports_claim"
    assert result["goal_completion_allowed"] is False


def test_backend_exception_returns_review_state_instead_of_crashing() -> None:
    result = ClaimVerificationEngine(StubBackend(error=True)).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "needs_human_review"
    assert result["reason"] == "backend_error:RuntimeError"
    assert result["claim_verified"] is False


def test_invalid_backend_schema_returns_review_state() -> None:
    backend = StubBackend({
        "status": "verified_true",
        "confidence": 1.0,
        "summary": "Invalid status is not accepted.",
        "evidence_urls": [EVIDENCE[0]["url"], EVIDENCE[1]["url"]],
    })
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "needs_human_review"
    assert result["reason"] == "invalid_response_schema"


def test_evidence_with_non_https_or_userinfo_urls_is_ignored() -> None:
    backend = StubBackend()
    evidence = EVIDENCE + [
        {"title": "HTTP source", "url": "http://unsafe.example/article", "content": "ignored"},
        {"title": "Userinfo URL", "url": "https://person@unsafe.example/article", "content": "ignored"},
    ]
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, evidence)

    assert result["status"] == "supports_claim"
    assert all(url.startswith("https://") for url in result["evidence_urls"])


def test_cognitive_layer_exposes_the_fail_closed_assessment_boundary() -> None:
    backend = StubBackend()
    with Store(":memory:") as store:
        core = AdaptiveCore(store)
        core.create_baseline("semantic-claim-assessment")
        cognitive = CognitiveSystem(core, internet=FakeInternet())
        result = cognitive.assess_claim(CLAIM, EVIDENCE, backend=backend)
        assert result["status"] == "supports_claim"
        assert result["goal_completion_allowed"] is False
        assert result["claim_verified"] is False


class FakeInternet:
    def learn(self, query: str, *, language: str = "ru", limit: int = 5):
        return []

    def recall(self, query: str = "", *, limit: int = 5):
        return []


def test_out_of_range_confidence_is_rejected_without_crashing() -> None:
    backend = StubBackend({
        "status": "supports_claim",
        "confidence": 1.7,
        "summary": "Invalid confidence should never be trusted.",
        "evidence_urls": [EVIDENCE[0]["url"], EVIDENCE[1]["url"]],
    })
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "needs_human_review"
    assert result["reason"] == "invalid_response_schema"
    assert result["claim_verified"] is False


def test_non_string_status_fails_closed_without_type_error() -> None:
    backend = StubBackend({
        "status": ["supports_claim"],
        "confidence": 0.9,
        "summary": "Malformed status value.",
        "evidence_urls": [EVIDENCE[0]["url"], EVIDENCE[1]["url"]],
    })
    result = ClaimVerificationEngine(backend).assess_claim(CLAIM, EVIDENCE)

    assert result["status"] == "needs_human_review"
    assert result["reason"] == "invalid_response_schema"


def test_verifier_rejects_unbounded_evidence_iterable() -> None:
    consumed = 0

    def endless_evidence():
        nonlocal consumed
        while True:
            consumed += 1
            yield {
                "title": f"Source {consumed}",
                "url": f"https://publisher-{consumed}.example/article",
                "content": "Some evidence to assess.",
            }

    with pytest.raises(ValueError, match="at most 100 evidence items"):
        ClaimVerificationEngine(StubBackend()).assess_claim(CLAIM, endless_evidence())
    assert consumed == 101
