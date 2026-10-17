"""Explainable symbolic inference and source-limited evidence review."""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse
from .models import validate_score

def terms(text: str) -> set[str]:
    return {word.casefold() for word in re.findall(r"[^\W_]+", text, flags=re.UNICODE) if len(word) > 2}

@dataclass(frozen=True, slots=True)
class KnowledgeFact:
    predicate: str
    value: str
    source: str
    confidence: float = 0.5
    inferred_by: str | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.predicate, str) or not self.predicate.strip() or len(self.predicate) > 128:
            raise ValueError("predicate must contain 1-128 characters")
        if not isinstance(self.value, str) or not self.value.strip() or len(self.value) > 2000:
            raise ValueError("value must contain 1-2,000 characters")
        if not isinstance(self.source, str) or not self.source.strip() or len(self.source) > 2000:
            raise ValueError("source must contain 1-2,000 characters")
        object.__setattr__(self, "confidence", validate_score(self.confidence))

@dataclass(frozen=True, slots=True)
class InferenceRule:
    """A human-readable rule over exact predicate/value pairs."""
    rule_id: str
    premises: tuple[tuple[str, str], ...]
    conclusion: tuple[str, str]
    explanation: str
    reliability: float = 0.8
    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", self.rule_id):
            raise ValueError("rule_id must be a 1-128 character ASCII identifier")
        if not self.premises or len(self.premises) > 20:
            raise ValueError("a rule must have 1-20 premises")
        if any(not key.strip() or not value.strip() for key, value in self.premises):
            raise ValueError("rule premises must be non-empty")
        if len(self.conclusion) != 2 or not self.conclusion[0].strip() or not self.conclusion[1].strip():
            raise ValueError("rule conclusion must contain predicate and value")
        if not self.explanation.strip() or len(self.explanation) > 2000:
            raise ValueError("explanation must contain 1-2,000 characters")
        object.__setattr__(self, "reliability", validate_score(self.reliability))

@dataclass(frozen=True, slots=True)
class InferenceResult:
    facts: tuple[KnowledgeFact, ...]
    trace: tuple[str, ...]
    conflicts: tuple[str, ...]
    steps: int

_CLAIM_STOPWORDS = {
    # Function words are removed only for coarse lexical comparison; this is
    # deliberately not a linguistic or semantic parser.
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "does", "do", "did", "has", "have", "had", "at", "of", "to", "in",
    "on", "for", "and", "or", "but", "by", "with", "as", "from", "that",
    "this", "it", "its", "can", "could", "will", "would", "should",
    "не", "нет", "без", "это", "как", "что", "для", "при", "или", "так",
    "его", "ее", "они", "она", "мы", "вы", "быть", "был", "была", "были",
    "есть", "на", "из", "к", "по", "и", "а", "но", "же", "ли", "этот",
    "эта", "эти", "который", "которая", "которые", "может", "могут",
}
_NEGATION_PATTERNS = (
    r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b", r"\bcannot\b",
    r"\bcan't\b", r"\bdoesn't\b", r"\bdon't\b", r"\bdidn't\b",
    r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b",
    r"\bwon't\b", r"\bне\b", r"\bнет\b", r"\bникогда\b", r"\bбез\b",
    r"\bнельзя\b", r"\bне\s+является\b", r"\bне\s+имеет\b",
)

def _claim_comparison_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        if len(token) > 2
        and not token.isdigit()
        and token.casefold() not in _CLAIM_STOPWORDS
    }

def _claim_polarity(text: str) -> str:
    lowered = text.casefold()
    if any(re.search(pattern, lowered, flags=re.UNICODE) for pattern in _NEGATION_PATTERNS):
        return "negated"
    return "affirmed"

@dataclass(frozen=True, slots=True)
class EvidenceReview:
    query: str
    status: str
    source_urls: tuple[str, ...]
    publisher_domains: tuple[str, ...]
    relevant_excerpts: tuple[dict[str, Any], ...]
    shared_query_terms: tuple[str, ...]
    suggested_topics: tuple[str, ...]
    caveat: str
    extracted_claims: tuple[dict[str, Any], ...] = ()
    potential_contradictions: tuple[dict[str, Any], ...] = ()

class ReasoningEngine:
    """A transparent rule engine; lexical claim comparison is not semantic understanding."""
    def forward_chain(self, facts: Iterable[KnowledgeFact], rules: Iterable[InferenceRule], *, max_steps: int = 100) -> InferenceResult:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 1000:
            raise ValueError("max_steps must be an integer from 1 to 1,000")
        current, rule_list = list(facts), list(rules)
        if len(current) > 1000 or len(rule_list) > 500:
            raise ValueError("reasoning input exceeds configured limits")
        if any(not isinstance(fact, KnowledgeFact) for fact in current):
            raise TypeError("facts must contain KnowledgeFact objects")
        if any(not isinstance(rule, InferenceRule) for rule in rule_list):
            raise TypeError("rules must contain InferenceRule objects")
        known = {(fact.predicate, fact.value) for fact in current}
        trace, steps, changed = [], 0, True
        while changed and steps < max_steps:
            changed = False
            for rule in rule_list:
                matching = []
                for predicate, value in rule.premises:
                    candidates = [fact for fact in current if fact.predicate == predicate and fact.value == value]
                    if not candidates:
                        break
                    matching.append(max(candidates, key=lambda fact: fact.confidence))
                else:
                    predicate, value = rule.conclusion
                    pair = (predicate, value)
                    if pair in known:
                        continue
                    steps += 1
                    confidence = min(fact.confidence for fact in matching) * rule.reliability
                    current.append(KnowledgeFact(predicate, value, "rule:" + rule.rule_id, confidence, inferred_by=rule.rule_id))
                    known.add(pair)
                    trace.append(f"{rule.rule_id}: {', '.join(f'{f.predicate}={f.value}' for f in matching)} => {predicate}={value}; {rule.explanation}")
                    changed = True
                    if steps >= max_steps:
                        break
        by_predicate: dict[str, set[str]] = {}
        for fact in current:
            by_predicate.setdefault(fact.predicate, set()).add(fact.value)
        conflicts = tuple(f"Conflicting values for {key}: {', '.join(sorted(values))}" for key, values in sorted(by_predicate.items()) if len(values) > 1)
        return InferenceResult(tuple(current), tuple(trace), conflicts, steps)

    def extract_claims(
        self, documents: Iterable[dict[str, Any]], *, max_claims: int = 50
    ) -> tuple[dict[str, Any], ...]:
        """Extract sentence-level claim candidates with source provenance.

        This is a transparent sentence heuristic. It does not decide whether a
        sentence is factual, true, supported, or entailed by its source.
        """
        if isinstance(max_claims, bool) or not isinstance(max_claims, int) or not 1 <= max_claims <= 200:
            raise ValueError("max_claims must be an integer from 1 to 200")
        items = list(documents)
        if len(items) > 100:
            raise ValueError("at most 100 documents may be processed")
        claims: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            url, title, content = item.get("url"), item.get("title", ""), item.get("content", "")
            if not isinstance(url, str) or not isinstance(title, str) or not isinstance(content, str):
                continue
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None:
                continue
            publisher = parsed.hostname.lower()
            if not content.strip():
                continue
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", content)
                if sentence.strip()
            ]
            if not sentences:
                sentences = [content.strip()]
            for sentence in sentences:
                sentence = re.sub(r"\s+", " ", sentence).strip()
                if len(sentence) < 12 or len(sentence) > 800 or sentence.endswith("?"):
                    continue
                words = re.findall(r"[^\W_]+", sentence, flags=re.UNICODE)
                if len(words) < 3:
                    continue
                dedupe_key = (url, sentence.casefold())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                digest = hashlib.sha256(
                    (url + "\0" + sentence.casefold()).encode("utf-8")
                ).hexdigest()[:20]
                claims.append({
                    "claim_id": digest,
                    "text": sentence,
                    "title": title[:300],
                    "source_url": url,
                    "publisher_domain": publisher,
                    "observed_at": item.get("fetched_at") if isinstance(item.get("fetched_at"), str) else None,
                    "source_hash": item.get("content_sha256") if isinstance(item.get("content_sha256"), str) else None,
                    "comparison_terms": sorted(_claim_comparison_terms(sentence)),
                    "polarity": _claim_polarity(sentence),
                    "extraction_method": "sentence_heuristic_v1",
                    "extraction_confidence": 0.4,
                    "independently_verified": False,
                })
                if len(claims) >= max_claims:
                    return tuple(claims)
        return tuple(claims)

    def detect_potential_contradictions(
        self, claims: Iterable[dict[str, Any]], *, max_findings: int = 20
    ) -> tuple[dict[str, Any], ...]:
        """Find narrow, negation-based contradiction candidates across URLs.

        Similarity plus opposite polarity is only a review queue signal; it is not
        evidence that either claim is true or that the pair logically contradicts.
        """
        if isinstance(max_findings, bool) or not isinstance(max_findings, int) or not 1 <= max_findings <= 100:
            raise ValueError("max_findings must be an integer from 1 to 100")
        items = [claim for claim in claims if isinstance(claim, dict)]
        if len(items) > 200:
            raise ValueError("at most 200 claims may be compared")
        findings: list[dict[str, Any]] = []
        for index, first in enumerate(items):
            first_terms = set(first.get("comparison_terms", []))
            if len(first_terms) < 2:
                continue
            for second in items[index + 1:]:
                if first.get("source_url") == second.get("source_url"):
                    continue
                if first.get("polarity") == second.get("polarity"):
                    continue
                second_terms = set(second.get("comparison_terms", []))
                if len(second_terms) < 2:
                    continue
                union = first_terms | second_terms
                overlap = len(first_terms & second_terms) / max(1, len(union))
                if overlap < 0.80:
                    continue
                affirmed = first if first.get("polarity") == "affirmed" else second
                negated = second if first.get("polarity") == "affirmed" else first
                findings.append({
                    "finding_id": hashlib.sha256(
                        (str(first.get("claim_id")) + "\0" + str(second.get("claim_id"))).encode("utf-8")
                    ).hexdigest()[:20],
                    "kind": "opposite_polarity_lexical_match",
                    "status": "candidate_requires_review",
                    "claim_ids": [first.get("claim_id"), second.get("claim_id")],
                    "claims": [
                        {"text": affirmed.get("text"), "source_url": affirmed.get("source_url")},
                        {"text": negated.get("text"), "source_url": negated.get("source_url")},
                    ],
                    "shared_terms": sorted(first_terms & second_terms),
                    "lexical_overlap": round(overlap, 3),
                    "independently_verified": False,
                    "limitation": "Negation and lexical similarity are heuristic signals, not semantic contradiction proof.",
                })
                if len(findings) >= max_findings:
                    return tuple(findings)
        return tuple(findings)

    def review_documents(self, query: str, documents: Iterable[dict[str, Any]], *, max_excerpts: int = 5) -> EvidenceReview:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1-2,000 characters")
        if isinstance(max_excerpts, bool) or not isinstance(max_excerpts, int) or not 1 <= max_excerpts <= 20:
            raise ValueError("max_excerpts must be an integer from 1 to 20")
        items = list(documents)
        if len(items) > 100:
            raise ValueError("at most 100 documents may be reviewed")
        query_terms, excerpts, source_urls, publishers = terms(query), [], [], set()
        for item in items:
            if not isinstance(item, dict):
                continue
            url, title, content = item.get("url"), item.get("title", ""), item.get("content", "")
            if not isinstance(url, str) or not isinstance(title, str) or not isinstance(content, str):
                continue
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname:
                continue
            source_urls.append(url)
            labels = parsed.hostname.lower().split(".")
            publishers.add(".".join(labels[-2:]) if len(labels) >= 2 else labels[0])
            sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", content) if s.strip()]
            if not sentences and content.strip():
                sentences = [content.strip()]
            for sentence in sentences:
                overlap = len(query_terms.intersection(terms(title + " " + sentence))) if query_terms else 1
                if overlap:
                    excerpts.append((float(overlap), {"title": title[:500], "url": url, "text": sentence[:1200], "keyword_overlap": overlap}))
        excerpts.sort(key=lambda pair: pair[0], reverse=True)
        unique_urls, unique_publishers = tuple(dict.fromkeys(source_urls)), tuple(sorted(publishers))
        status = "no_evidence" if not unique_urls else ("single_publisher_unverified" if len(unique_publishers) == 1 else "multiple_publishers_unverified")
        caveat = "Keyword overlap is a retrieval heuristic, not proof of truth, causality, or logical entailment. External text remains unverified."
        if status == "single_publisher_unverified":
            caveat += " Retrieved pages come from one publisher and do not independently corroborate one another."
        claims = self.extract_claims(items)
        potential_contradictions = self.detect_potential_contradictions(claims)
        return EvidenceReview(
            query, status, unique_urls, unique_publishers,
            tuple(item for _, item in excerpts[:max_excerpts]),
            tuple(sorted(query_terms)), tuple(self.suggest_topics(items, excluded=(query,))),
            caveat, claims, potential_contradictions,
        )

    def suggest_topics(self, documents: Iterable[dict[str, Any]], *, excluded: Iterable[str] = (), limit: int = 4) -> list[str]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
            raise ValueError("limit must be an integer from 1 to 10")
        excluded_set = {entry.casefold().strip() for entry in excluded if isinstance(entry, str)}
        stopwords = {"что","это","как","для","при","или","так","его","ее","они","она","который","которая","которые","также","может","можно","более","менее","this","that","with","from","have","into","about","their","there","which","what","when","where","while","were","been","being","such","the","and","for","are","but","not","you","your","its","use","used"}
        candidates: dict[str, int] = {}
        for document in documents:
            if not isinstance(document, dict):
                continue
            title, content = document.get("title", ""), document.get("content", "")
            if isinstance(title, str):
                clean_title = title.strip()
                if 3 <= len(clean_title) <= 80 and clean_title.casefold() not in excluded_set:
                    candidates[clean_title] = candidates.get(clean_title, 0) + 5
            if not isinstance(content, str):
                continue
            counts: dict[str, int] = {}
            for word in re.findall(r"[^\W_]+", content.casefold(), flags=re.UNICODE):
                if len(word) >= 4 and word not in stopwords:
                    counts[word] = counts.get(word, 0) + 1
            for word, count in counts.items():
                if count >= 2 and word not in excluded_set:
                    candidates[word] = candidates.get(word, 0) + count
        ranked = sorted(candidates.items(), key=lambda pair: (pair[1], len(pair[0]), pair[0]), reverse=True)
        result = []
        for topic, _ in ranked:
            if topic.casefold() in excluded_set or any(topic.casefold() == value.casefold() for value in result):
                continue
            result.append(topic)
            if len(result) >= limit:
                break
        return result
