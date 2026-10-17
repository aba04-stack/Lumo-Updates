"""Persistent internal drive pressures and inspectable goal lifecycle.

Drive values represent computational priorities/need pressure. They are not
claims about biological needs, human-like desires, or subjective experience.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .models import validate_score
from .store import Store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_DRIVES = ("curiosity", "competence", "coherence", "progress", "uncertainty_reduction")
_BASELINE = {
    "curiosity": 0.52,
    "competence": 0.42,
    "coherence": 0.48,
    "progress": 0.50,
    "uncertainty_reduction": 0.45,
}
_GOAL_STATUSES = {"queued", "active", "needs_verification", "completed", "deferred"}
_TRANSITIONS = {
    "queued": {"active", "deferred"},
    "active": {"queued", "needs_verification", "deferred"},
    "needs_verification": {"active", "deferred"},
    "deferred": {"queued", "active", "needs_verification"},
    "completed": set(),
}
_REASON_DRIVE = {
    "derived_from_new_material": "curiosity",
    "initial_research_goal": "progress",
    "revisit_existing_knowledge": "coherence",
    "verify_uncertainty": "uncertainty_reduction",
    "improve_skill": "competence",
    "curiosity": "curiosity",
}


@dataclass(frozen=True, slots=True)
class DriveState:
    """Bounded pressure values for internal computational drives."""
    curiosity: float = _BASELINE["curiosity"]
    competence: float = _BASELINE["competence"]
    coherence: float = _BASELINE["coherence"]
    progress: float = _BASELINE["progress"]
    uncertainty_reduction: float = _BASELINE["uncertainty_reduction"]
    updated_at: str = "initial"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def strongest(self) -> str:
        return max(_DRIVES, key=lambda name: getattr(self, name))


@dataclass(frozen=True, slots=True)
class MotivationalGoal:
    """A persistent goal with an explicit drive, priority, and lifecycle."""
    goal_id: str
    title: str
    drive: str
    priority: float
    rationale: str
    status: str
    created_at: str
    updated_at: str


class MotivationEngine:
    """Maintain bounded drive pressure and persistent, auditable goals."""

    def __init__(self, store: Store, *, homeostasis: float = 0.04) -> None:
        if isinstance(homeostasis, bool) or not isinstance(homeostasis, (int, float)) or not math.isfinite(homeostasis) or not 0 <= homeostasis <= 1:
            raise ValueError("homeostasis must be a finite number in [0, 1]")
        self.store = store
        self.homeostasis = float(homeostasis)

    def _events(self) -> list[dict[str, Any]]:
        return self.store.recent_events(limit=10_000)

    def state(self) -> DriveState:
        event = self.store.latest_event("motivation-state-updated")
        if event is None:
            return DriveState()
        payload = event["payload"].get("state", {})
        try:
            values = {name: validate_score(payload[name]) for name in _DRIVES}
            updated_at = payload.get("updated_at", event["created_at"])
            if isinstance(updated_at, str):
                return DriveState(**values, updated_at=updated_at)
        except (KeyError, ValueError):
            pass
        return DriveState()

    def update_outcome(
        self,
        *,
        succeeded: bool,
        progress: float,
        uncertainty: float = 0.0,
        contradiction: float = 0.0,
        novelty: float = 0.0,
    ) -> DriveState:
        """Update drive pressure after a measurable result.

        Signals must come from available telemetry. A missing contradiction
        detector should send 0; the engine does not invent contradictions.
        """
        if not isinstance(succeeded, bool):
            raise ValueError("succeeded must be a bool")
        progress, uncertainty, contradiction, novelty = map(
            validate_score, (progress, uncertainty, contradiction, novelty)
        )
        previous = self.state()
        deltas = {
            "curiosity": 0.12 * novelty + 0.08 * uncertainty - 0.10 * progress * (1.0 - novelty),
            "competence": 0.15 * (not succeeded) + 0.10 * (0.5 - progress) - 0.18 * progress * succeeded,
            "coherence": 0.18 * contradiction + 0.05 * uncertainty - 0.12 * progress * (1.0 - contradiction),
            "progress": 0.16 * (1.0 - progress) - 0.16 * progress,
            "uncertainty_reduction": 0.20 * uncertainty + 0.18 * contradiction - 0.15 * progress * (1.0 - uncertainty),
        }
        values: dict[str, float] = {}
        for name, baseline in _BASELINE.items():
            current = getattr(previous, name)
            drifted = current + (baseline - current) * self.homeostasis
            values[name] = min(1.0, max(0.0, drifted + deltas[name]))
        state = DriveState(**values, updated_at=_now())
        self.store.record_event("motivation-state-updated", "motivation", {
            "succeeded": succeeded,
            "signals": {
                "progress": progress, "uncertainty": uncertainty,
                "contradiction": contradiction, "novelty": novelty,
            },
            "state": asdict(state),
        })
        return state

    @staticmethod
    def drive_for_reason(reason: str) -> str:
        return _REASON_DRIVE.get(reason, "progress")

    @staticmethod
    def _goal_id(title: str, drive: str) -> str:
        raw = f"{drive.casefold()}\n{title.strip().casefold()}".encode("utf-8")
        return "goal-" + hashlib.sha256(raw).hexdigest()[:24]

    def goals(self, *, status: str | None = None) -> list[MotivationalGoal]:
        if status is not None and (not isinstance(status, str) or status not in _GOAL_STATUSES):
            raise ValueError(f"unsupported goal status: {status}")
        records: dict[str, dict[str, Any]] = {}
        events = self.store.events_of_types(
            ["motivation-goal-created", "motivation-goal-status", "motivation-goal-verification-recorded"],
            limit=None,
            newest_first=False,
        )
        verification_records: dict[str, str] = {}
        for event in events:
            payload = event["payload"]
            if not isinstance(payload, dict):
                continue
            kind = event["event_type"]
            if kind == "motivation-goal-created":
                goal_id = payload.get("goal_id")
                if not isinstance(goal_id, str):
                    continue
                records.setdefault(goal_id, {
                    "goal_id": goal_id,
                    "title": payload.get("title", ""),
                    "drive": payload.get("drive", "progress"),
                    "priority": payload.get("priority", 0.5),
                    "rationale": payload.get("rationale", ""),
                    "status": "queued",
                    "created_at": event["created_at"],
                    "updated_at": event["created_at"],
                })
            elif kind == "motivation-goal-verification-recorded":
                goal_id, verification_id = payload.get("goal_id"), payload.get("verification_id")
                if (
                    isinstance(goal_id, str)
                    and goal_id in records
                    and isinstance(verification_id, str) and 1 <= len(verification_id) <= 120
                    and isinstance(payload.get("method"), str)
                    and payload["method"] in {"manual_review", "deterministic_test", "formal_proof"}
                    and isinstance(payload.get("reviewer"), str)
                    and 1 <= len(payload["reviewer"].strip()) <= 128
                    and isinstance(payload.get("criteria"), list)
                    and 1 <= len(payload["criteria"]) <= 20
                    and all(isinstance(value, str) and 1 <= len(value.strip()) <= 300 for value in payload["criteria"])
                    and isinstance(payload.get("evidence_refs"), list)
                    and 1 <= len(payload["evidence_refs"]) <= 20
                    and all(isinstance(value, str) and 1 <= len(value.strip()) <= 2_000 for value in payload["evidence_refs"])
                ):
                    verification_records[verification_id] = goal_id
            elif kind == "motivation-goal-status":
                goal_id, requested = payload.get("goal_id"), payload.get("status")
                if (
                    not isinstance(goal_id, str)
                    or goal_id not in records
                    or not isinstance(requested, str)
                    or requested not in _GOAL_STATUSES
                ):
                    continue
                current = records[goal_id]["status"]
                if requested == current:
                    continue
                # Replay the same transition constraints as the public API. A raw
                # status event cannot bypass them by being written to the event log.
                if payload.get("from") != current:
                    continue
                if requested == "completed":
                    verification_id = payload.get("verification_id")
                    allowed = (
                        current == "needs_verification"
                        and isinstance(verification_id, str)
                        and verification_records.get(verification_id) == goal_id
                    )
                else:
                    allowed = requested in _TRANSITIONS.get(current, set())
                if allowed:
                    records[goal_id]["status"] = requested
                    records[goal_id]["updated_at"] = event["created_at"]
        result: list[MotivationalGoal] = []
        for item in records.values():
            try:
                goal = MotivationalGoal(
                    goal_id=str(item["goal_id"]),
                    title=str(item["title"]),
                    drive=str(item["drive"]),
                    priority=validate_score(item["priority"]),
                    rationale=str(item["rationale"]),
                    status=str(item["status"]),
                    created_at=str(item["created_at"]),
                    updated_at=str(item["updated_at"]),
                )
            except ValueError:
                continue
            if status is None or goal.status == status:
                result.append(goal)
        result.sort(key=lambda item: (item["updated_at"] if isinstance(item, dict) else item.updated_at, item.goal_id))
        return result

    def add_goal(
        self,
        title: str,
        *,
        drive: str,
        priority: float = 0.5,
        rationale: str = "curiosity",
    ) -> MotivationalGoal:
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 180:
            raise ValueError("goal title must contain 1-180 characters")
        if not isinstance(drive, str) or drive not in _DRIVES:
            raise ValueError(f"unsupported drive: {drive}")
        priority = validate_score(priority)
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 300:
            raise ValueError("rationale must contain 1-300 characters")
        normalized = title.strip()
        goal_id = self._goal_id(normalized, drive)
        existing = next((goal for goal in self.goals() if goal.goal_id == goal_id), None)
        if existing is not None:
            return existing
        self.store.record_event("motivation-goal-created", "motivation", {
            "goal_id": goal_id, "title": normalized, "drive": drive,
            "priority": priority, "rationale": rationale.strip(),
        })
        return next(goal for goal in self.goals() if goal.goal_id == goal_id)

    def complete_goal(
        self,
        goal_id: str,
        *,
        method: str,
        reviewer: str,
        criteria: list[str] | tuple[str, ...],
        evidence_refs: list[str] | tuple[str, ...],
    ) -> MotivationalGoal:
        """Complete a goal only with an explicit, auditable verification record.

        This records that a named reviewer/process checked the listed criteria
        against the supplied references. It does not authenticate that reviewer
        or independently prove that a cited source is true.
        """
        goal = next((item for item in self.goals() if item.goal_id == goal_id), None)
        if goal is None:
            raise KeyError(f"unknown goal: {goal_id}")
        if goal.status != "needs_verification":
            raise ValueError("goal must be in needs_verification before completion")
        if not isinstance(method, str) or method not in {"manual_review", "deterministic_test", "formal_proof"}:
            raise ValueError("method must be manual_review, deterministic_test, or formal_proof")
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer.strip()) > 128:
            raise ValueError("reviewer must contain 1-128 characters")
        if isinstance(criteria, str) or not isinstance(criteria, (list, tuple)) or not 1 <= len(criteria) <= 20:
            raise ValueError("criteria must contain 1-20 explicit review criteria")
        normalized_criteria: list[str] = []
        for criterion in criteria:
            if not isinstance(criterion, str) or not criterion.strip() or len(criterion.strip()) > 300:
                raise ValueError("each criterion must contain 1-300 characters")
            normalized_criteria.append(criterion.strip())
        if isinstance(evidence_refs, str) or not isinstance(evidence_refs, (list, tuple)) or not 1 <= len(evidence_refs) <= 20:
            raise ValueError("evidence_refs must contain 1-20 references")
        normalized_refs: list[str] = []
        for reference in evidence_refs:
            if not isinstance(reference, str) or not reference.strip() or len(reference.strip()) > 2_000:
                raise ValueError("each evidence reference must contain 1-2,000 characters")
            normalized_refs.append(reference.strip())
        # Avoid storing duplicate sources/criteria without discarding their order.
        normalized_criteria = list(dict.fromkeys(normalized_criteria))
        normalized_refs = list(dict.fromkeys(normalized_refs))
        verification_id = "verification-" + uuid.uuid4().hex
        self.store.record_event("motivation-goal-verification-recorded", "motivation", {
            "goal_id": goal_id,
            "verification_id": verification_id,
            "method": method,
            "reviewer": reviewer.strip(),
            "criteria": normalized_criteria,
            "evidence_refs": normalized_refs,
            "claim_verified": False,
            "reviewer_authenticated": False,
            "meaning": "An auditable review record, not independent proof of truth.",
        })
        # A failure between these two writes leaves the goal uncompleted; replay
        # ignores a completion without a matching verification record.
        self.store.record_event("motivation-goal-status", "motivation", {
            "goal_id": goal_id,
            "from": goal.status,
            "status": "completed",
            "verification_id": verification_id,
        })
        return next(item for item in self.goals() if item.goal_id == goal_id)

    def set_goal_status(self, goal_id: str, status: str) -> MotivationalGoal:
        if not isinstance(goal_id, str) or not goal_id:
            raise ValueError("goal_id must be non-empty")
        if not isinstance(status, str) or status not in _GOAL_STATUSES:
            raise ValueError(f"unsupported goal status: {status}")
        if status == "completed":
            existing_completed = next((goal for goal in self.goals() if goal.goal_id == goal_id), None)
            if existing_completed is not None and existing_completed.status == "completed":
                return existing_completed
            raise ValueError("use complete_goal() with an explicit verification record")
        existing = next((goal for goal in self.goals() if goal.goal_id == goal_id), None)
        if existing is None:
            raise KeyError(f"unknown goal: {goal_id}")
        if status != existing.status and status not in _TRANSITIONS[existing.status]:
            raise ValueError(f"invalid goal transition: {existing.status} -> {status}")
        if status != existing.status:
            self.store.record_event("motivation-goal-status", "motivation", {
                "goal_id": goal_id, "from": existing.status, "status": status,
            })
        return next(goal for goal in self.goals() if goal.goal_id == goal_id)

    def goal_score(self, drive: str, priority: float) -> float:
        """Blend user/topic priority with current drive pressure."""
        if drive not in _DRIVES:
            raise ValueError(f"unsupported drive: {drive}")
        priority = validate_score(priority)
        pressure = getattr(self.state(), drive)
        return priority + 0.5 * (pressure - _BASELINE[drive])
