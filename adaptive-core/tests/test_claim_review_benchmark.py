"""Small, synthetic regression benchmark for heuristic claim review.

This fixture validates known narrow cases only. Its scores are not an estimate of
real-world accuracy or semantic understanding.
"""
from __future__ import annotations

from adaptive_core import ReasoningEngine


CASES = (
    {
        "name": "negation mismatch",
        "documents": [
            {"title": "A", "url": "https://one.example/earth", "content": "Earth is round."},
            {"title": "B", "url": "https://two.example/earth", "content": "Earth is not round."},
        ],
        "expected_candidate": True,
        "expected_kind": "opposite_polarity_lexical_match",
    },
    {
        "name": "same-unit quantity mismatch",
        "documents": [
            {"title": "A", "url": "https://one.example/mars", "content": "Mars has 2 moons."},
            {"title": "B", "url": "https://two.example/mars", "content": "Mars has 3 moons."},
        ],
        "expected_candidate": True,
        "expected_kind": "same_unit_numeric_mismatch",
    },
    {
        "name": "year mismatch",
        "documents": [
            {"title": "A", "url": "https://one.example/bridge", "content": "The bridge opened in 1998."},
            {"title": "B", "url": "https://two.example/bridge", "content": "The bridge opened in 2001."},
        ],
        "expected_candidate": True,
        "expected_kind": "year_value_mismatch",
    },
    {
        "name": "same source is not independent comparison",
        "documents": [
            {
                "title": "One page",
                "url": "https://one.example/mars",
                "content": "Mars has 2 moons. Mars has 3 moons.",
            }
        ],
        "expected_candidate": False,
        "expected_kind": None,
    },
    {
        "name": "different units are not silently converted",
        "documents": [
            {"title": "A", "url": "https://one.example/route", "content": "The route is 3 km long."},
            {"title": "B", "url": "https://two.example/route", "content": "The route is 3 miles long."},
        ],
        "expected_candidate": False,
        "expected_kind": None,
    },
    {
        "name": "equivalent unit alias and value",
        "documents": [
            {"title": "A", "url": "https://one.example/tower", "content": "The tower is 3 km tall."},
            {"title": "B", "url": "https://two.example/tower", "content": "The tower is 3 kilometers tall."},
        ],
        "expected_candidate": False,
        "expected_kind": None,
    },
    {
        "name": "different subjects despite one shared word",
        "documents": [
            {"title": "A", "url": "https://one.example/square", "content": "The square is not round."},
            {"title": "B", "url": "https://two.example/earth", "content": "Earth is round."},
        ],
        "expected_candidate": False,
        "expected_kind": None,
    },
    {
        "name": "numerically different but lexically unrelated",
        "documents": [
            {"title": "A", "url": "https://one.example/mars", "content": "Mars has 2 moons."},
            {"title": "B", "url": "https://two.example/city", "content": "A city has 3 buildings."},
        ],
        "expected_candidate": False,
        "expected_kind": None,
    },
)


def test_synthetic_claim_review_benchmark() -> None:
    engine = ReasoningEngine()
    results = []
    for case in CASES:
        claims = engine.extract_claims(case["documents"])
        findings = engine.detect_potential_contradictions(claims)
        candidate = bool(findings)
        kind = findings[0]["kind"] if findings else None
        assert candidate is case["expected_candidate"], case["name"]
        assert kind == case["expected_kind"], case["name"]
        assert all(finding["independently_verified"] is False for finding in findings)
        results.append((case["expected_candidate"], candidate))

    true_positives = sum(expected and predicted for expected, predicted in results)
    false_positives = sum(not expected and predicted for expected, predicted in results)
    false_negatives = sum(expected and not predicted for expected, predicted in results)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)

    # These metrics describe only this tiny hand-authored fixture.
    assert len(CASES) == 8
    assert precision == 1.0
    assert recall == 1.0
