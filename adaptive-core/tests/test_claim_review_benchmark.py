"""Hand-labelled synthetic benchmark for the supported slice of claim review.

Results are regression diagnostics for these authored examples, not a measure of
real-world accuracy. Hard semantic paraphrases remain explicitly unscored and
are listed as known limitations instead of being hidden by a perfect score.
"""
from __future__ import annotations

from adaptive_core import ReasoningEngine


CASES = (
    {
        "name": "direct negation",
        "documents": [
            {"title": "A", "url": "https://one.example/earth", "content": "Earth is round."},
            {"title": "B", "url": "https://two.example/earth", "content": "Earth is not round."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "opposite_polarity_lexical_match",
    },
    {
        "name": "same-unit quantity mismatch",
        "documents": [
            {"title": "A", "url": "https://one.example/mars", "content": "Mars has 2 moons."},
            {"title": "B", "url": "https://two.example/mars", "content": "Mars has 3 moons."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "same_unit_numeric_mismatch",
    },
    {
        "name": "same-context year mismatch",
        "documents": [
            {"title": "A", "url": "https://one.example/bridge", "content": "The bridge opened in 1998."},
            {"title": "B", "url": "https://two.example/bridge", "content": "The bridge opened in 2001."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "year_value_mismatch",
    },
    {
        "name": "same source is not independent comparison",
        "documents": [
            {
                "title": "One page", "url": "https://one.example/mars",
                "content": "Mars has 2 moons. Mars has 3 moons.",
            }
        ],
        "relation": "contradiction", "scorable": False,
        "expected_candidate": False, "expected_kind": None,
        "known_limitation": "same-source comparisons are excluded by policy",
    },
    {
        "name": "different units are not silently converted",
        "documents": [
            {"title": "A", "url": "https://one.example/route", "content": "The route is 3 km long."},
            {"title": "B", "url": "https://two.example/route", "content": "The route is 3 miles long."},
        ],
        "relation": "unsupported_by_current_numeric_policy", "scorable": False,
        "expected_candidate": False, "expected_kind": None,
        "known_limitation": "unit conversion is intentionally not attempted",
    },
    {
        "name": "equivalent unit alias and value",
        "documents": [
            {"title": "A", "url": "https://one.example/tower", "content": "The tower is 3 km tall."},
            {"title": "B", "url": "https://two.example/tower", "content": "The tower is 3 kilometers tall."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "different subjects despite shared word",
        "documents": [
            {"title": "A", "url": "https://one.example/square", "content": "The square is not round."},
            {"title": "B", "url": "https://two.example/earth", "content": "Earth is round."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "numerically different but lexically unrelated",
        "documents": [
            {"title": "A", "url": "https://one.example/mars", "content": "Mars has 2 moons."},
            {"title": "B", "url": "https://two.example/city", "content": "A city has 3 buildings."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "existential claims can both be true",
        "documents": [
            {"title": "A", "url": "https://one.example/work", "content": "Some employees work remotely."},
            {"title": "B", "url": "https://two.example/work", "content": "Some employees do not work remotely."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "existential claims with plural nouns can coexist",
        "documents": [
            {"title": "A", "url": "https://one.example/cats", "content": "Some cats live indoors."},
            {"title": "B", "url": "https://two.example/cats", "content": "Some cats do not live indoors."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "different year contexts do not prove polarity conflict",
        "documents": [
            {"title": "A", "url": "https://one.example/paper", "content": "The researcher did not publish the paper in 2020."},
            {"title": "B", "url": "https://two.example/paper", "content": "The researcher did publish the paper in 2021."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "same predicate with direct opposite polarity",
        "documents": [
            {"title": "A", "url": "https://one.example/door", "content": "The door is locked."},
            {"title": "B", "url": "https://two.example/door", "content": "The door is not locked."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "opposite_polarity_lexical_match",
    },
    {
        "name": "numeric mismatch with singular/plural normalization",
        "documents": [
            {"title": "A", "url": "https://one.example/windows", "content": "The room has 2 windows."},
            {"title": "B", "url": "https://two.example/windows", "content": "The room has 3 windows."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "same_unit_numeric_mismatch",
    },
    {
        "name": "number words are treated as counts",
        "documents": [
            {"title": "A", "url": "https://one.example/jupiter", "content": "Jupiter has two moons."},
            {"title": "B", "url": "https://two.example/jupiter", "content": "Jupiter has three moons."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "same_unit_numeric_mismatch",
    },
    {
        "name": "no moons versus a moon",
        "documents": [
            {"title": "A", "url": "https://one.example/mercury", "content": "Mercury has no moons."},
            {"title": "B", "url": "https://two.example/mercury", "content": "Mercury has a moon."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "opposite_polarity_lexical_match",
    },
    {
        "name": "all versus not all",
        "documents": [
            {"title": "A", "url": "https://one.example/birds", "content": "All birds can fly."},
            {"title": "B", "url": "https://two.example/birds", "content": "Not all birds can fly."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "opposite_polarity_lexical_match",
    },
    {
        "name": "different years and opposite polarity may coexist",
        "documents": [
            {"title": "A", "url": "https://one.example/event", "content": "The event did not happen in 2020."},
            {"title": "B", "url": "https://two.example/event", "content": "The event did happen in 2021."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "Russian direct negation",
        "documents": [
            {"title": "A", "url": "https://one.example/earth-ru", "content": "Планета Земля круглая."},
            {"title": "B", "url": "https://two.example/earth-ru", "content": "Планета Земля не круглая."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "opposite_polarity_lexical_match",
    },
    {
        "name": "Russian number words are treated as counts",
        "documents": [
            {"title": "A", "url": "https://one.example/mars-ru", "content": "У Марса два спутника."},
            {"title": "B", "url": "https://two.example/mars-ru", "content": "У Марса три спутника."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "same_unit_numeric_mismatch",
    },
    {
        "name": "same unordered numeric list in different order",
        "documents": [
            {"title": "A", "url": "https://one.example/list", "content": "The list contains 2 and 3."},
            {"title": "B", "url": "https://two.example/list", "content": "The list contains 3 and 2."},
        ],
        "relation": "coexistence", "scorable": True,
        "expected_candidate": False, "expected_kind": None,
    },
    {
        "name": "swapped values attached to distinct event contexts",
        "documents": [
            {"title": "A", "url": "https://one.example/bridge-events", "content": "The bridge opened in 1998 and was renovated in 2001."},
            {"title": "B", "url": "https://two.example/bridge-events", "content": "The bridge opened in 2001 and was renovated in 1998."},
        ],
        "relation": "contradiction", "scorable": True,
        "expected_candidate": True, "expected_kind": "year_value_mismatch",
    },
    {
        "name": "paraphrase and synonym remain a known miss",
        "documents": [
            {"title": "A", "url": "https://one.example/mars", "content": "Mars has 2 moons."},
            {"title": "B", "url": "https://two.example/mars", "content": "Mars has three natural satellites."},
        ],
        "relation": "contradiction", "scorable": False,
        "expected_candidate": False, "expected_kind": None,
        "known_limitation": "the lexical overlap threshold does not map moons to natural satellites",
        "known_miss": True,
    },
)


def _evaluate_cases():
    engine = ReasoningEngine()
    rows = []
    for case in CASES:
        claims = engine.extract_claims(case["documents"])
        findings = engine.detect_potential_contradictions(claims)
        candidate = bool(findings)
        kind = findings[0]["kind"] if findings else None
        assert candidate is case["expected_candidate"], case["name"]
        assert kind == case["expected_kind"], case["name"]
        assert all(finding["independently_verified"] is False for finding in findings)
        rows.append({
            "name": case["name"],
            "relation": case["relation"],
            "scorable": case["scorable"],
            "known_miss": case.get("known_miss", False),
            "predicted_candidate": candidate,
            "findings": findings,
        })
    return rows


def _metrics(rows):
    scored = [row for row in rows if row["scorable"]]
    tp = sum(row["relation"] == "contradiction" and row["predicted_candidate"] for row in scored)
    fp = sum(row["relation"] != "contradiction" and row["predicted_candidate"] for row in scored)
    fn = sum(row["relation"] == "contradiction" and not row["predicted_candidate"] for row in scored)
    tn = sum(row["relation"] != "contradiction" and not row["predicted_candidate"] for row in scored)
    return {
        "scored_cases": len(scored),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "known_limitation_cases": sum(not case["scorable"] and "known_limitation" in case for case in CASES),
    }


def test_expanded_claim_review_benchmark() -> None:
    rows = _evaluate_cases()
    metrics = _metrics(rows)
    assert len(CASES) == 22
    # These are precision/recall values over 19 hand-labelled, in-scope synthetic
    # cases only. Three unscored cases remain visible as policy/coverage limits
    # and are not hidden inside the headline metrics.
    assert metrics["scored_cases"] == 19
    assert metrics["false_positives"] == 0, metrics
    assert metrics["false_negatives"] == 0, metrics
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["known_limitation_cases"] == 3
    assert metrics["true_positives"] == 11
    assert metrics["true_negatives"] == 8


def test_benchmark_reports_known_unscored_cases_instead_of_claiming_general_accuracy() -> None:
    rows = _evaluate_cases()
    known_misses = [row["name"] for row in rows if row["known_miss"]]
    assert known_misses == ["paraphrase and synonym remain a known miss"]
    assert all(row["predicted_candidate"] is False for row in rows if not row["scorable"])
