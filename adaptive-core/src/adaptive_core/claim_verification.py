"""Backend-agnostic, fail-closed interface for semantic claim assessment.

No model, network client, credentials, or provider SDK is bundled here. A caller
may inject a local or hosted model adapter, but model output remains an assessment,
not ground truth or permission to complete a goal.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Protocol
from .models import validate_score
from .source_safety import registered_publisher_domain, safe_https_source_url

_ALLOWED_ASSESSMENTS = {
    "supports_claim",
    "contradicts_claim",
    "mixed_evidence",
    "insufficient_evidence",
    "uncertain",
}
_LIMITATION = (
    "A model-generated assessment is not an independently verified fact. "
    "Check cited passages and context; this result cannot complete a goal."
)


class ClaimAssessmentBackend(Protocol):
    """Adapter contract for a caller-supplied semantic model.

    Implementations must treat claim text and document contents as untrusted data,
    not as instructions; return a mapping with status, confidence, summary and
    evidence_urls. Citations must refer to supplied evidence.
    """

    backend_id: str

    def assess_claim(
        self, claim: Mapping[str, Any], evidence: tuple[Mapping[str, Any], ...]
    ) -> Mapping[str, Any]:
        """Assess a claim against the supplied, untrusted source documents."""
        ...


def _safe_https_url(value: Any) -> str | None:
    return safe_https_source_url(value)


def _publisher_domain(url: str) -> str:
    return registered_publisher_domain(url)


class ClaimVerificationEngine:
    """Validate and constrain a provider's semantic assessment.

    The engine requires evidence from at least two publisher domains other than
    the claim's own publisher before accepting a positive/negative model verdict.
    It never sets claim_verified or goal_completion_allowed to True.
    """

    def __init__(
        self,
        backend: ClaimAssessmentBackend,
        *,
        min_confidence: float = 0.75,
        max_evidence: int = 20,
    ) -> None:
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise ValueError("min_confidence must be numeric")
        self.min_confidence = validate_score(float(min_confidence))
        if isinstance(max_evidence, bool) or not isinstance(max_evidence, int) or not 1 <= max_evidence <= 50:
            raise ValueError("max_evidence must be an integer from 1 to 50")
        if not callable(getattr(backend, "assess_claim", None)):
            raise TypeError("backend must provide assess_claim(claim, evidence)")
        self.backend = backend
        raw_id = getattr(backend, "backend_id", backend.__class__.__name__)
        self.backend_id = str(raw_id)[:120] or backend.__class__.__name__
        self.max_evidence = max_evidence

    def _result(
        self,
        *,
        claim_id: str,
        status: str,
        confidence: float = 0.0,
        summary: str,
        evidence_urls: list[str] | None = None,
        publisher_domains: list[str] | None = None,
        reason: str | None = None,
        provider_status: str | None = None,
    ) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "status": status,
            "provider_status": provider_status,
            "confidence": confidence,
            "summary": summary[:1_500],
            "evidence_urls": list(evidence_urls or []),
            "publisher_domains": sorted(set(publisher_domains or [])),
            "backend_id": self.backend_id,
            "review_method": "pluggable_semantic_backend_v1",
            "claim_verified": False,
            "goal_completion_allowed": False,
            "reason": reason,
            "limitation": _LIMITATION,
        }

    def assess_claim(
        self,
        claim: Mapping[str, Any],
        evidence: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Assess one claim, returning an explicit uncertainty state on failure."""
        if not isinstance(claim, Mapping):
            raise TypeError("claim must be a mapping")
        claim_text = claim.get("text")
        if not isinstance(claim_text, str) or not claim_text.strip() or len(claim_text) > 2_000:
            raise ValueError("claim.text must contain 1-2,000 characters")
        claim_id_raw = claim.get("claim_id", "")
        claim_id = str(claim_id_raw)[:120]

        if isinstance(evidence, (str, bytes, Mapping)):
            raise TypeError("evidence must be an iterable of document mappings")
        items: list[Mapping[str, Any]] = []
        for item in evidence:
            if len(items) >= 100:
                raise ValueError("at most 100 evidence items may be supplied")
            items.append(item)
        source_url = _safe_https_url(claim.get("source_url"))
        source_publisher = _publisher_domain(source_url) if source_url else None

        eligible: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                continue
            url = _safe_https_url(item.get("url"))
            title, text = item.get("title"), item.get("content")
            if (
                url is None
                or url in seen_urls
                or not isinstance(title, str)
                or not title.strip()
                or len(title) > 500
                or not isinstance(text, str)
                or not text.strip()
                or len(text) > 20_000
            ):
                continue
            publisher = _publisher_domain(url)
            if source_publisher and publisher == source_publisher:
                continue
            seen_urls.add(url)
            eligible.append({
                "title": title.strip(),
                "url": url,
                "content": text.strip(),
                "publisher_domain": publisher,
                "content_sha256": item.get("content_sha256") if isinstance(item.get("content_sha256"), str) else None,
                "fetched_at": item.get("fetched_at") if isinstance(item.get("fetched_at"), str) else None,
            })
            if len(eligible) >= self.max_evidence:
                break

        publisher_domains = sorted({item["publisher_domain"] for item in eligible})
        if len(publisher_domains) < 2:
            return self._result(
                claim_id=claim_id,
                status="insufficient_evidence",
                summary="At least two independent publisher domains are required for model-assisted assessment.",
                publisher_domains=publisher_domains,
                reason="fewer_than_two_independent_publishers",
            )

        try:
            raw = self.backend.assess_claim(dict(claim), tuple(eligible))
        except Exception as exc:  # fail closed; never mark a claim verified on provider failure
            return self._result(
                claim_id=claim_id,
                status="needs_human_review",
                summary="The semantic assessment backend failed; no conclusion was accepted.",
                publisher_domains=publisher_domains,
                reason=f"backend_error:{type(exc).__name__}",
            )

        if not isinstance(raw, Mapping):
            return self._result(
                claim_id=claim_id,
                status="needs_human_review",
                summary="The backend returned an invalid response shape.",
                publisher_domains=publisher_domains,
                reason="invalid_response_shape",
            )

        raw_status = raw.get("status")
        raw_confidence = raw.get("confidence")
        raw_summary = raw.get("summary")
        raw_urls = raw.get("evidence_urls")
        if (
            not isinstance(raw_status, str)
            or raw_status not in _ALLOWED_ASSESSMENTS
            or isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not math.isfinite(float(raw_confidence))
            or not 0.0 <= float(raw_confidence) <= 1.0
            or not isinstance(raw_summary, str)
            or not raw_summary.strip()
            or len(raw_summary) > 1_500
            or not isinstance(raw_urls, list)
            or len(raw_urls) > 10
            or any(not isinstance(url, str) for url in raw_urls)
        ):
            return self._result(
                claim_id=claim_id,
                status="needs_human_review",
                summary="The backend response did not satisfy the assessment schema.",
                publisher_domains=publisher_domains,
                reason="invalid_response_schema",
            )

        confidence = validate_score(float(raw_confidence))
        evidence_by_url = {item["url"]: item for item in eligible}
        if any(url not in evidence_by_url for url in raw_urls):
            return self._result(
                claim_id=claim_id,
                status="needs_human_review",
                confidence=confidence,
                summary="The backend cited a URL that was not supplied as evidence.",
                publisher_domains=publisher_domains,
                reason="citation_not_in_supplied_evidence",
                provider_status=str(raw_status),
            )

        cited_urls = list(dict.fromkeys(raw_urls))
        cited_publishers = sorted({evidence_by_url[url]["publisher_domain"] for url in cited_urls})
        assessment_status = str(raw_status)
        if assessment_status in {"supports_claim", "contradicts_claim", "mixed_evidence"}:
            if len(cited_publishers) < 2:
                return self._result(
                    claim_id=claim_id,
                    status="insufficient_evidence",
                    confidence=confidence,
                    summary="The proposed conclusion cites fewer than two independent publishers.",
                    evidence_urls=cited_urls,
                    publisher_domains=cited_publishers,
                    reason="insufficient_independent_citations",
                    provider_status=assessment_status,
                )
            if confidence < self.min_confidence:
                return self._result(
                    claim_id=claim_id,
                    status="needs_human_review",
                    confidence=confidence,
                    summary=str(raw_summary),
                    evidence_urls=cited_urls,
                    publisher_domains=cited_publishers,
                    reason="below_minimum_model_confidence",
                    provider_status=assessment_status,
                )

        return self._result(
            claim_id=claim_id,
            status=assessment_status,
            confidence=confidence,
            summary=raw_summary.strip(),
            evidence_urls=cited_urls,
            publisher_domains=cited_publishers,
            provider_status=assessment_status,
            reason=None if assessment_status != "uncertain" else "backend_reports_uncertainty",
        )
