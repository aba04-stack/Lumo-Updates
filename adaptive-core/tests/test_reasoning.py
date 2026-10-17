from __future__ import annotations
import pytest
from adaptive_core import InferenceRule, KnowledgeFact, ReasoningEngine

def test_symbolic_reasoning_records_trace_and_provenance() -> None:
    engine = ReasoningEngine()
    facts = [KnowledgeFact("rain", "yes", "weather_sensor", 0.9)]
    rules = [
        InferenceRule("rain-wets-ground", (("rain", "yes"),), ("ground", "wet"),
                      "Rain is modeled as wetting the ground.", 0.8),
        InferenceRule("wet-ground-slippery", (("ground", "wet"),), ("ground", "slippery"),
                      "A simplified risk heuristic.", 0.9),
    ]
    result = engine.forward_chain(facts, rules)
    values = {(fact.predicate, fact.value) for fact in result.facts}
    assert ("ground", "wet") in values and ("ground", "slippery") in values
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
