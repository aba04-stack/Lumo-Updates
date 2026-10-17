from __future__ import annotations
import pytest
from adaptive_core import InferenceRule, KnowledgeFact, ReasoningEngine

def test_symbolic_reasoning_records_trace_and_provenance() -> None:
    engine = ReasoningEngine()
    facts = [KnowledgeFact("rain", "yes", "weather_sensor", 0.9)]
    rules = [
        InferenceRule("rain-wets-ground", (("rain", "yes"),), ("ground", "wet"),
                      "Rain is modeled as wetting the ground.", 0.8),
        InferenceRule("wet-ground-slippery", (("ground", "wet"),), ("hazard", "slippery"),
                      "A simplified risk heuristic.", 0.9),
    ]
    result = engine.forward_chain(facts, rules)
    values = {(fact.predicate, fact.value) for fact in result.facts}
    assert ("ground", "wet") in values and ("hazard", "slippery") in values
    assert len(result.trace) == 2
    slippery = next(fact for fact in result.facts if fact.value == "slippery")
    assert slippery.inferred_by == "wet-ground-slippery"
    assert slippery.confidence == pytest.approx(0.9 * 0.8 * 0.9)
    assert result.conflicts == ()

def test_symbolic_reasoning_does_not_hide_conflicting_facts() -> None:
    result = ReasoningEngine().forward_chain([
        KnowledgeFact("door_locked", "yes", "sensor-a", 0.8),
        KnowledgeFact("door_locked", "no", "sensor-b", 0.7),
    ], [])
    assert len(result.conflicts) == 1
    assert "door_locked" in result.conflicts[0]

def test_review_does_not_treat_same_publisher_as_independent_confirmation() -> None:
    review = ReasoningEngine().review_documents("machine learning", [{
        "title": "Machine learning", "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "content": "Machine learning uses data to improve a model. Results need evaluation.",
    }])
    assert review.status == "single_publisher_unverified"
    assert review.source_urls == ("https://en.wikipedia.org/wiki/Machine_learning",)
    assert "do not independently corroborate" in review.caveat
    assert review.relevant_excerpts

def test_no_sources_is_explicitly_unknown() -> None:
    review = ReasoningEngine().review_documents("unknown topic", [])
    assert review.status == "no_evidence"
    assert review.source_urls == ()


def test_sentence_claim_candidates_preserve_provenance_and_mark_uncertainty() -> None:
    engine = ReasoningEngine()
    docs = [
        {
            "title": "A source",
            "url": "https://source-a.example/fact",
            "content": "Earth is round.",
            "fetched_at": "2026-10-17T10:00:00+00:00",
            "content_sha256": "a" * 64,
        },
        {
            "title": "Another source",
            "url": "https://source-b.example/fact",
            "content": "Earth is not round.",
            "fetched_at": "2026-10-17T10:01:00+00:00",
            "content_sha256": "b" * 64,
        },
    ]

    claims = engine.extract_claims(docs)
    assert len(claims) == 2
    assert claims[0]["source_url"] == docs[0]["url"]
    assert claims[0]["observed_at"] == docs[0]["fetched_at"]
    assert claims[0]["source_hash"] == docs[0]["content_sha256"]
    assert claims[0]["extraction_method"] == "sentence_heuristic_v1"
    assert claims[0]["independently_verified"] is False
    assert claims[0]["extraction_confidence"] < 0.5

    findings = engine.detect_potential_contradictions(claims)
    assert len(findings) == 1
    finding = findings[0]
    assert finding["kind"] == "opposite_polarity_lexical_match"
    assert finding["status"] == "candidate_requires_review"
    assert finding["independently_verified"] is False
    assert {claim["source_url"] for claim in finding["claims"]} == {
        docs[0]["url"], docs[1]["url"]
    }


def test_document_review_includes_claims_and_candidate_contradictions() -> None:
    review = ReasoningEngine().review_documents("Earth round", [
        {
            "title": "Affirmative",
            "url": "https://one.example/claim",
            "content": "Earth is round.",
        },
        {
            "title": "Negated",
            "url": "https://two.example/claim",
            "content": "Earth is not round.",
        },
    ])
    assert len(review.extracted_claims) == 2
    assert len(review.potential_contradictions) == 1
    assert review.potential_contradictions[0]["status"] == "candidate_requires_review"


def test_contradiction_candidate_requires_distinct_urls_opposite_polarity_and_overlap() -> None:
    engine = ReasoningEngine()
    claims = engine.extract_claims([
        {"title": "A", "url": "https://one.example/a", "content": "Earth is round."},
        {"title": "B", "url": "https://one.example/a", "content": "Earth is not round."},
        {"title": "C", "url": "https://two.example/c", "content": "Earth is blue."},
    ])
    findings = engine.detect_potential_contradictions(claims)
    assert findings == ()
