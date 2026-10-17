"""Bounded, self-directed research cycles with persistent memory and feedback."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Iterable
from .cognition import CognitiveSystem
from .internet_learning import InternetLearningError
from .models import validate_score
from .reasoning import ReasoningEngine, terms

_ALLOWED_LANGUAGES = {"ar","de","en","es","fr","it","ja","kk","pt","ru","tr","uk","zh"}
_DEFAULT_TOPICS = ("научный метод", "рассуждение", "эмоции", "обучение")

class AutonomousLearningLoop:
    """Bounded algorithmic exploration, not an LLM or neural-weight trainer.
    Network access occurs only when run_step/run is explicitly called.
    """
    def __init__(self, cognitive: CognitiveSystem, *, topics: Iterable[str] | None = None, language: str = "ru", pages_per_topic: int = 2) -> None:
        if isinstance(pages_per_topic, bool) or not isinstance(pages_per_topic, int) or not 1 <= pages_per_topic <= 3:
            raise ValueError("pages_per_topic must be an integer from 1 to 3")
        if not isinstance(language, str) or language not in _ALLOWED_LANGUAGES:
            raise ValueError("unsupported language")
        self.cognitive, self.store = cognitive, cognitive.store
        self.reasoner: ReasoningEngine = cognitive.reasoner
        self.language, self.pages_per_topic = language, pages_per_topic
        has_queue = any(event["event_type"] == "autonomous-topic-queued" for event in self._events())
        if topics is None:
            initial = list(_DEFAULT_TOPICS if not has_queue else ())
        else:
            if isinstance(topics, (str, bytes)):
                raise ValueError("topics must be an iterable of topic strings, not a single string")
            initial = []
            for topic in topics:
                if len(initial) >= 100:
                    raise ValueError("at most 100 initial topics may be queued")
                initial.append(topic)
        for topic in initial:
            self.queue_topic(topic, priority=0.6, reason="initial_research_goal")

    def _events(self) -> list[dict[str, Any]]:
        # Retrieve only the streams this loop needs so unrelated history cannot
        # push queued topics and research outcomes out of the global recent window.
        return self.store.events_of_types(
            ["autonomous-topic-queued", "autonomous-research-step"],
            limit=None,
            newest_first=True,
        )

    @staticmethod
    def _topic(payload: dict[str, Any]) -> str | None:
        topic = payload.get("topic")
        return topic.strip() if isinstance(topic, str) and topic.strip() else None

    def queue_topic(
        self, topic: str, *, priority: float = 0.5,
        reason: str = "curiosity", drive: str | None = None,
    ) -> bool:
        if not isinstance(topic, str) or not topic.strip() or len(topic) > 180:
            raise ValueError("topic must contain 1-180 characters")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 100:
            raise ValueError("reason must contain 1-100 characters")
        priority = validate_score(priority)
        normalized = topic.strip()
        selected_drive = drive or self.cognitive.motivation.drive_for_reason(reason)
        # Validate the drive before writing either the goal or the queue event.
        self.cognitive.motivation.goal_score(selected_drive, priority)
        known = {value.casefold() for event in self._events() if event["event_type"] == "autonomous-topic-queued" if (value := self._topic(event["payload"])) is not None}
        if normalized.casefold() in known:
            return False
        with self.store.transaction():
            self.cognitive.motivation.add_goal(
                normalized, drive=selected_drive, priority=priority, rationale=reason
            )
            self.store.record_event("autonomous-topic-queued", "autonomous-research", {
                "topic": normalized, "priority": priority, "reason": reason, "drive": selected_drive
            })
        return True

    def _history(self) -> tuple[list[dict[str, Any]], set[str], dict[str, int], set[str]]:
        events = self._events()
        queued, attempts = [], {}
        for event in reversed(events):
            payload = event["payload"]
            if event["event_type"] == "autonomous-topic-queued":
                topic = self._topic(payload)
                if topic is not None:
                    reason = str(payload.get("reason", "curiosity"))
                    queued.append({
                        "topic": topic,
                        "priority": float(payload.get("priority", 0.5)),
                        "reason": reason,
                        "drive": str(payload.get("drive") or self.cognitive.motivation.drive_for_reason(reason)),
                    })
            elif event["event_type"] == "autonomous-research-step":
                topic = payload.get("topic")
                if isinstance(topic, str) and topic:
                    key = topic.casefold()
                    attempts[key] = attempts.get(key, 0) + 1

        # A retrieved page is not proof that a goal was achieved. Use the
        # persistent goal ledger to decide whether a topic is complete or pending.
        status_by_goal = {
            (goal.title.casefold(), goal.drive): goal.status
            for goal in self.cognitive.motivation.goals()
        }
        successful, pending_verification = set(), set()
        for item in queued:
            key = item["topic"].casefold()
            drive = item.get("drive") or self.cognitive.motivation.drive_for_reason(item.get("reason", "curiosity"))
            status = status_by_goal.get((key, drive))
            if status == "completed":
                successful.add(key)
            elif status == "needs_verification":
                pending_verification.add(key)
        return queued, successful, attempts, pending_verification

    def _choose_topic(self, session_attempted: set[str]) -> str | None:
        queued, successful, attempts, pending_verification = self._history()
        state, candidates = self.cognitive.affect.state(), {}
        for item in queued:
            topic, key = item["topic"], item["topic"].casefold()
            if key in successful or key in pending_verification or key in session_attempted or attempts.get(key, 0) >= 2:
                continue
            reason = item.get("reason", "curiosity")
            drive = item.get("drive") or self.cognitive.motivation.drive_for_reason(reason)
            # Migrate topics created by older versions into the persistent goal ledger.
            self.cognitive.motivation.add_goal(
                topic, drive=drive, priority=item["priority"], rationale=reason
            )
            score = self.cognitive.motivation.goal_score(drive, item["priority"])
            # Learned affect still modulates exploration, while drive pressure can
            # shift priority toward competence, coherence, or reducing uncertainty.
            if reason == "derived_from_new_material":
                score += 0.5 * state.curiosity
            else:
                score += 0.05 * state.curiosity
            if reason in {"initial_research_goal", "revisit_existing_knowledge"}:
                score -= 0.2 * state.frustration
            if attempts.get(key, 0):
                score -= 0.2 * attempts[key] * (1.0 + state.frustration)
            candidates[topic] = max(candidates.get(topic, -1.0), score)
        return max(candidates, key=candidates.get) if candidates else None

    def _derive_topics(self, documents: list[dict[str, Any]], query: str) -> list[str]:
        suggestions = self.reasoner.suggest_topics(documents, excluded=(query,), limit=4)
        events = self._events()
        queued = {topic.casefold() for event in events if event["event_type"] == "autonomous-topic-queued" if (topic := self._topic(event["payload"])) is not None}
        attempted = {str(event["payload"].get("topic", "")).casefold() for event in events if event["event_type"] == "autonomous-research-step"}
        added = []
        for topic in suggestions:
            if topic.casefold() in queued or topic.casefold() in attempted:
                continue
            if self.queue_topic(topic, priority=0.35, reason="derived_from_new_material"):
                added.append(topic)
        return added

    def run_step(self, *, page_limit: int | None = None) -> dict[str, Any]:
        effective_limit = self.pages_per_topic if page_limit is None else page_limit
        if isinstance(effective_limit, bool) or not isinstance(effective_limit, int) or not 1 <= effective_limit <= 3:
            raise ValueError("page_limit must be an integer from 1 to 3")
        return self._run_one(effective_limit, set())

    def _run_one(self, page_limit: int, session_attempted: set[str]) -> dict[str, Any]:
        query = self._choose_topic(session_attempted)
        if query is None:
            cached = self.cognitive.internet.recall("", limit=20)
            for topic in self.reasoner.suggest_topics(cached, limit=4):
                self.queue_topic(topic, priority=0.25, reason="revisit_existing_knowledge")
            query = self._choose_topic(session_attempted)
        if query is None:
            return {"status": "idle", "reason": "no_unattempted_topics", "documents_found": 0}
        session_attempted.add(query.casefold())
        queue_record = next((
            event["payload"] for event in self._events()
            if event["event_type"] == "autonomous-topic-queued"
            and (queued_topic := self._topic(event["payload"])) is not None
            and queued_topic.casefold() == query.casefold()
        ), {})
        goal_reason = str(queue_record.get("reason", "curiosity"))
        goal_drive = str(queue_record.get("drive") or self.cognitive.motivation.drive_for_reason(goal_reason))
        goal = self.cognitive.motivation.add_goal(
            query, drive=goal_drive,
            priority=validate_score(queue_record.get("priority", 0.5)),
            rationale=goal_reason,
        )
        goal = self.cognitive.motivation.set_goal_status(goal.goal_id, "active")
        prior_documents = self.cognitive.internet.recall("", limit=20)
        known_terms: set[str] = set()
        for prior in prior_documents:
            known_terms.update(terms(str(prior.get("title", "")) + " " + str(prior.get("content", ""))))
        self.cognitive.affect.update("task-started", intensity=0.7)
        error, documents = None, []
        try:
            documents = self.cognitive.learn_from_internet(query, language=self.language, limit=page_limit)
        except InternetLearningError as exc:
            error = str(exc)[:240]
        review = self.reasoner.review_documents(query, documents)
        claim_candidates = list(review.extracted_claims)
        contradiction_candidates = list(review.potential_contradictions)
        # These are cautious review-queue signals, not verified logical conflicts.
        contradiction_signal = validate_score(min(1.0, len(contradiction_candidates) / 3.0))
        discovered_terms: set[str] = set()
        for document in documents:
            discovered_terms.update(terms(str(document.get("title", "")) + " " + str(document.get("content", ""))))
        discovered_terms -= terms(query)
        new_terms = discovered_terms - known_terms
        page_yield = validate_score(min(1.0, len(documents) / max(1, page_limit)))
        query_term_count = max(1, len(terms(query)))
        best_keyword_overlap = max(
            (int(excerpt.get("keyword_overlap", 0)) for excerpt in review.relevant_excerpts),
            default=0,
        )
        relevance_coverage = validate_score(min(1.0, best_keyword_overlap / query_term_count))
        novelty_signal = validate_score(min(1.0, len(new_terms) / 10.0))
        # Multi-signal retrieval proxy: page count alone cannot dominate.
        # Keyword overlap is lexical relevance, not semantic correctness.
        retrieval_utility = validate_score(
            0.50 * relevance_coverage + 0.30 * novelty_signal + 0.20 * page_yield
        ) if documents else 0.0
        next_topics = self._derive_topics(documents, query) if documents else []
        status = "source_found_unverified" if documents else "no_source_found"
        summary = f"Research query: {query}. Retrieved {len(documents)} page(s). Evidence status: {review.status}. Relevance coverage proxy: {relevance_coverage:.3f}. Potential related terms: {', '.join(sorted(discovered_terms)[:12]) or 'none'}."
        # Research memory, appraisal, drive state, goal status and step telemetry
        # form one persistent unit. A failed final write must not leave an apparent
        # completed research step with only some of its state committed.
        with self.store.transaction():
            self.cognitive.memory.remember(
                "autonomous-research", summary, tags=["research", "internet", "source-review", review.status],
                strategy_id=None, score=retrieval_utility, success=bool(documents),
                metadata={
                    "query": query, "status": status, "sources": list(review.source_urls),
                    "publisher_domains": list(review.publisher_domains),
                    "page_yield": page_yield, "relevance_coverage_proxy": relevance_coverage,
                    "novelty_proxy": novelty_signal, "retrieval_utility": retrieval_utility,
                    "retrieval_utility_is_truth_score": False,
                    "claims_independently_verified": False,
                    "claim_extraction_method": "sentence_heuristic_v1",
                    "claim_candidates": claim_candidates,
                    "potential_contradictions": contradiction_candidates,
                    "contradiction_proxy": contradiction_signal,
                    "suggested_topics": next_topics, "error": error,
                },
            )
            # These are observable proxies, not a truth score. Since fetched pages are
            # unverified, uncertainty remains high even when lexical coverage is strong.
            uncertainty_signal = 1.0 if not documents else 1.0 - 0.35 * relevance_coverage
            succeeded = bool(documents) and error is None
            self.cognitive.affect.appraise(
                succeeded=succeeded,
                progress=retrieval_utility,
                uncertainty=uncertainty_signal,
                contradiction=contradiction_signal,
                novelty=novelty_signal,
                feedback=retrieval_utility if documents else -0.5,
            )
            drive_state = self.cognitive.motivation.update_outcome(
                succeeded=succeeded,
                progress=retrieval_utility,
                uncertainty=uncertainty_signal,
                contradiction=contradiction_signal,
                novelty=novelty_signal,
            )
            # Completing a retrieval pass is not the same as verifying the research
            # question. A separate verifier must mark a goal completed later.
            goal = self.cognitive.motivation.set_goal_status(
                goal.goal_id, "needs_verification" if documents else "deferred"
            )
            self.store.record_event("autonomous-research-step", "autonomous-research", {
                "topic": query, "documents_found": len(documents), "sources": list(review.source_urls),
                "evidence_status": review.status, "goal_status": goal.status,
                "retrieval_utility": retrieval_utility,
                "page_yield": page_yield, "relevance_coverage_proxy": relevance_coverage,
                "novelty_proxy": novelty_signal, "uncertainty_proxy": uncertainty_signal,
                "claims_independently_verified": False,
                "discovered_terms": sorted(discovered_terms)[:30],
                "new_terms": sorted(new_terms)[:30], "suggested_topics": next_topics, "error": error,
            })
        return {
            "status": status, "topic": query, "documents_found": len(documents),
            "source_urls": list(review.source_urls), "evidence_status": review.status,
            "relevant_excerpts": list(review.relevant_excerpts), "caveat": review.caveat,
            "discovered_terms": sorted(discovered_terms)[:30], "new_terms": sorted(new_terms)[:30],
            "next_topics": next_topics, "page_yield": page_yield,
            "relevance_coverage_proxy": relevance_coverage, "novelty_proxy": novelty_signal,
            "epistemic_uncertainty_proxy": uncertainty_signal,
            "retrieval_utility": retrieval_utility, "retrieval_utility_is_truth_score": False,
            "claims_independently_verified": False, "error": error,
            "claim_extraction_method": "sentence_heuristic_v1",
            "claim_candidates": claim_candidates,
            "potential_contradictions": contradiction_candidates,
            "contradiction_proxy": contradiction_signal,
            "goal_id": goal.goal_id, "goal_status": goal.status, "drive": goal.drive,
            "drive_state": asdict(drive_state),
        }

    def run(self, *, max_steps: int = 3, page_budget: int = 6) -> dict[str, Any]:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 10:
            raise ValueError("max_steps must be an integer from 1 to 10")
        if isinstance(page_budget, bool) or not isinstance(page_budget, int) or not 1 <= page_budget <= 30:
            raise ValueError("page_budget must be an integer from 1 to 30")
        results, session_attempted, remaining = [], set(), page_budget
        for _ in range(max_steps):
            if remaining <= 0:
                break
            request_limit = min(self.pages_per_topic, remaining)
            result = self._run_one(request_limit, session_attempted)
            if result["status"] == "idle":
                break
            results.append(result)
            remaining -= request_limit
        return {
            "steps_completed": len(results), "pages_requested_budget": page_budget,
            "pages_budget_remaining": remaining, "documents_found": sum(item["documents_found"] for item in results),
            "results": results, "affect": asdict(self.cognitive.affect.state()),
            "affect_model_samples": self.cognitive.affect.model_snapshot()["samples"],
            "motivation": asdict(self.cognitive.motivation.state()),
            "goals_needing_verification": [
                {
                    "goal_id": goal.goal_id, "title": goal.title, "drive": goal.drive,
                    "priority": goal.priority, "rationale": goal.rationale, "status": goal.status,
                }
                for goal in self.cognitive.motivation.goals(status="needs_verification")
            ],
        }
