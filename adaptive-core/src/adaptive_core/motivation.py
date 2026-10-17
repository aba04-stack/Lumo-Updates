"""Persistent internal drive pressures and inspectable goal lifecycle.

Drive values represent computational priorities/need pressure. They are not
claims about biological needs, human-like desires, or subjective experience.
"""
from __future__ import annotations

import hashlib
import math
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
    "active": {"queued", "needs_verification", "completed", "deferred"},
    "needs_verification": {"active", "completed", "deferred"},
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
        if status is not None and status not in _GOAL_STATUSES:
            raise ValueError(f"unsupported goal status: {status}")
        records: dict[str, dict[str, Any]] = {}
        events = self.store.events_of_types(
            ["motivation-goal-created", "motivation-goal-status"],
            limit=None,
            newest_first=False,
        )
        for event in events:
            payload = event["payload"]
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
            elif kind == "motivation-goal-status":
                goal_id = payload.get("goal_id")
                if goal_id in records and payload.get("status") in _GOAL_STATUSES:
                    records[goal_id]["status"] = payload["status"]
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
        if drive not in _DRIVES:
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

    def set_goal_status(self, goal_id: str, status: str) -> MotivationalGoal:
        if not isinstance(goal_id, str) or not goal_id:
            raise ValueError("goal_id must be non-empty")
        if status not in _GOAL_STATUSES:
            raise ValueError(f"unsupported goal status: {status}")
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
