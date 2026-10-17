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


def test_same_unit_numeric_difference_becomes_review_candidate() -> None:
    engine = ReasoningEngine()
    claims = engine.extract_claims([
        {
            "title": "Source one",
            "url": "https://one.example/moons",
            "content": "Mars has 2 moons.",
        },
        {
            "title": "Source two",
            "url": "https://two.example/moons",
            "content": "Mars has 3 moons.",
        },
    ])
    assert [claim["numeric_mentions"][0]["value"] for claim in claims] == ["2", "3"]
    findings = engine.detect_potential_contradictions(claims)
    assert len(findings) == 1
    assert findings[0]["kind"] == "same_unit_numeric_mismatch"
    assert findings[0]["status"] == "candidate_requires_review"
    assert findings[0]["numeric_differences"][0]["first_value"] == "2"
    assert findings[0]["numeric_differences"][0]["second_value"] == "3"
    assert findings[0]["independently_verified"] is False


def test_equivalent_numeric_values_do_not_create_conflict() -> None:
    engine = ReasoningEngine()
    claims = engine.extract_claims([
        {
            "title": "Source one",
            "url": "https://one.example/height",
            "content": "The tower is 3 km tall.",
        },
        {
            "title": "Source two",
            "url": "https://two.example/height",
            "content": "The tower is 3 kilometers tall.",
        },
    ])
    assert claims[0]["numeric_mentions"][0]["unit"] == "km"
    assert claims[1]["numeric_mentions"][0]["unit"] == "km"
    assert engine.detect_potential_contradictions(claims) == ()


def test_different_units_are_not_compared_without_conversion() -> None:
    engine = ReasoningEngine()
    claims = engine.extract_claims([
        {
            "title": "Source one",
            "url": "https://one.example/route",
            "content": "The route is 3 km long.",
        },
        {
            "title": "Source two",
            "url": "https://two.example/route",
            "content": "The route is 3 miles long.",
        },
    ])
    assert claims[0]["numeric_mentions"][0]["unit"] == "km"
    assert claims[1]["numeric_mentions"][0]["unit"] == "mile"
    assert engine.detect_potential_contradictions(claims) == ()


def test_year_mismatch_is_flagged_but_remains_unverified() -> None:
    engine = ReasoningEngine()
    claims = engine.extract_claims([
        {
            "title": "Source one",
            "url": "https://one.example/bridge",
            "content": "The bridge opened in 1998.",
        },
        {
            "title": "Source two",
            "url": "https://two.example/bridge",
            "content": "The bridge opened in 2001.",
        },
    ])
    findings = engine.detect_potential_contradictions(claims)
    assert len(findings) == 1
    assert findings[0]["kind"] == "year_value_mismatch"
    assert findings[0]["independently_verified"] is False


def test_not_only_is_not_classified_as_negation() -> None:
    claims = ReasoningEngine().extract_claims([
        {
            "title": "Source one",
            "url": "https://one.example/learning",
            "content": "Learning is not only useful but also powerful.",
        }
    ])
    assert len(claims) == 1
    assert claims[0]["polarity"] == "affirmed"


def test_evidence_review_rejects_malformed_urls_and_groups_publishers_consistently() -> None:
    review = ReasoningEngine().review_documents("historical bridge", [
        {
            "title": "Unsafe credential URL",
            "url": "https://name@example.org/fake",
            "content": "The bridge opened in 1998.",
        },
        {
            "title": "Local host",
            "url": "https://127.0.0.1/bridge",
            "content": "The bridge opened in 1998.",
        },
        {
            "title": "Valid source",
            "url": "https://archive-one.example.co.uk/bridge",
            "content": "The bridge opened in 1998.",
        },
        {
            "title": "Invalid empty source",
            "url": "https://archive-two.example.co.uk/bridge",
            "content": "   ",
        },
    ])
    assert review.source_urls == ("https://archive-one.example.co.uk/bridge",)
    assert review.publisher_domains == ("example.co.uk",)
    assert review.status == "single_publisher_unverified"
    assert len(review.extracted_claims) == 1
    assert review.extracted_claims[0]["publisher_domain"] == "example.co.uk"


def test_review_rejects_unbounded_document_iterable_without_consuming_it_forever() -> None:
    consumed = 0

    def endless_documents():
        nonlocal consumed
        while True:
            consumed += 1
            yield {
                "title": "Test",
                "url": "https://source.example/article",
                "content": "A sentence long enough to look like a claim.",
            }

    with pytest.raises(ValueError, match="at most 100 documents"):
        ReasoningEngine().review_documents("test query", endless_documents())
    assert consumed == 101


def test_reasoner_bounds_fact_and_rule_iterables_before_materializing() -> None:
    engine = ReasoningEngine()
    facts_seen = 0

    def endless_facts():
        nonlocal facts_seen
        while True:
            facts_seen += 1
            yield KnowledgeFact("sensor", str(facts_seen), "test", 0.5)

    with pytest.raises(ValueError, match="at most 1000 facts"):
        engine.forward_chain(endless_facts(), [])
    assert facts_seen == 1001


def test_malformed_claim_records_are_ignored_by_contradiction_detector() -> None:
    engine = ReasoningEngine()
    malformed = [
        {
            "claim_id": "valid-ish",
            "source_url": "https://a.example/claim",
            "text": "The claim has a sufficiently long text.",
            "comparison_terms": ["claim", "sufficiently", "long"],
            "polarity": "affirmed",
        },
        {
            "claim_id": "bad-polarity",
            "source_url": "https://b.example/claim",
            "text": "The claim has a sufficiently long text.",
            "comparison_terms": ["claim", "sufficiently", "long"],
            "polarity": [],
        },
        {
            "claim_id": "bad-terms",
            "source_url": "https://c.example/claim",
            "text": "The claim has a sufficiently long text.",
            "comparison_terms": [["unhashable"]],
            "polarity": "negated",
        },
    ]
    assert engine.detect_potential_contradictions(malformed) == ()


def test_inference_rule_freezes_mutable_pairs_and_rejects_malformed_values() -> None:
    premises = [["sensor", "ready"]]
    conclusion = ["device", "on"]
    rule = InferenceRule("safe-rule", premises, conclusion, "A test rule.")
    premises[0][0] = "changed"
    conclusion[0] = "changed"
    assert rule.premises == (("sensor", "ready"),)
    assert rule.conclusion == ("device", "on")

    with pytest.raises(ValueError, match="rule_id"):
        InferenceRule([], (("sensor", "ready"),), ("device", "on"), "invalid id")
    with pytest.raises(ValueError, match="predicate-value pair"):
        InferenceRule("bad-premise", (("sensor",),), ("device", "on"), "invalid premise")
    with pytest.raises(ValueError, match="predicate and value"):
        InferenceRule("bad-conclusion", (("sensor", "ready"),), ["device"], "invalid conclusion")


def test_knowledge_fact_rejects_malformed_provenance_type() -> None:
    with pytest.raises(ValueError, match="inferred_by"):
        KnowledgeFact("p", "v", "source", inferred_by=["not", "an", "id"])


def test_reasoner_skips_oversized_external_documents() -> None:
    engine = ReasoningEngine()
    huge = {
        "title": "x" * 501,
        "url": "https://source.example/huge",
        "content": "A claim. " * 3_000,
    }
    assert engine.extract_claims([huge]) == ()
    review = engine.review_documents("claim", [huge])
    assert review.source_urls == ()
    assert review.extracted_claims == ()
