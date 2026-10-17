"""Appraisal-driven affect-like internal state; not a claim of subjective feeling."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from .models import validate_score
from .store import Store

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

_BASELINE = {"curiosity": 0.55, "satisfaction": 0.30, "frustration": 0.10, "caution": 0.35, "confidence": 0.25, "engagement": 0.45}
_DELTAS = {
    "task-started": {"curiosity": 0.04, "engagement": 0.08},
    "knowledge-found": {"satisfaction": 0.06, "curiosity": 0.04, "engagement": 0.03},
    "search-empty": {"frustration": 0.12, "curiosity": 0.03, "caution": 0.05},
    "task-succeeded": {"satisfaction": 0.15, "frustration": -0.15, "confidence": 0.08, "engagement": 0.05},
    "task-failed": {"frustration": 0.18, "caution": 0.12, "confidence": -0.08, "curiosity": 0.05},
    "contradiction-detected": {"caution": 0.20, "curiosity": 0.12, "confidence": -0.10},
    "uncertainty-detected": {"curiosity": 0.12, "caution": 0.07},
    "information-gain": {"satisfaction": 0.08, "curiosity": 0.07, "engagement": 0.04},
    "no-progress": {"frustration": 0.10, "curiosity": -0.07, "caution": 0.04},
    "test-passed": {"satisfaction": 0.10, "confidence": 0.04, "frustration": -0.08},
    "test-failed": {"frustration": 0.15, "caution": 0.12, "confidence": -0.08},
}

@dataclass(frozen=True, slots=True)
class AffectState:
    """Bounded, interpretable control variables influenced by observed events."""
    curiosity: float = _BASELINE["curiosity"]
    satisfaction: float = _BASELINE["satisfaction"]
    frustration: float = _BASELINE["frustration"]
    caution: float = _BASELINE["caution"]
    confidence: float = _BASELINE["confidence"]
    engagement: float = _BASELINE["engagement"]
    updated_at: str = "initial"

class AffectEngine:
    """Persist event-driven affect-like state used to modulate exploration."""
    def __init__(self, store: Store, *, decay: float = 0.02) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 <= decay <= 1:
            raise ValueError("decay must be a number in [0, 1]")
        self.store, self.decay = store, float(decay)

    def state(self) -> AffectState:
        for event in self.store.recent_events(limit=1000):
            if event["event_type"] == "affect-state-updated":
                payload = event["payload"].get("state", {})
                try:
                    values = {name: validate_score(payload[name]) for name in _BASELINE}
                    updated_at = payload.get("updated_at", event["created_at"])
                    if not isinstance(updated_at, str):
                        break
                    return AffectState(**values, updated_at=updated_at)
                except (KeyError, ValueError):
                    break
        return AffectState()

    def update(self, event_type: str, *, intensity: float = 1.0) -> AffectState:
        if event_type not in _DELTAS:
            raise ValueError(f"unsupported appraisal event: {event_type}")
        intensity = validate_score(intensity)
        previous = self.state()
        values = {}
        for name, baseline in _BASELINE.items():
            current = getattr(previous, name)
            decayed = current + (baseline - current) * self.decay
            values[name] = min(1.0, max(0.0, decayed + _DELTAS[event_type].get(name, 0.0) * intensity))
        state = AffectState(**values, updated_at=_now())
        self.store.record_event("affect-state-updated", "system", {
            "trigger": event_type, "intensity": intensity, "state": asdict(state)
        })
        return state

    def appraise(self, *, succeeded: bool, progress: float = 0.0, uncertainty: float = 0.0, contradiction: float = 0.0) -> AffectState:
        """Choose a bounded state update from measured appraisal signals."""
        if not isinstance(succeeded, bool):
            raise ValueError("succeeded must be a bool")
        progress, uncertainty, contradiction = map(validate_score, (progress, uncertainty, contradiction))
        if contradiction >= 0.5:
            return self.update("contradiction-detected", intensity=contradiction)
        if uncertainty >= 0.7:
            return self.update("uncertainty-detected", intensity=uncertainty)
        if not succeeded:
            return self.update("task-failed")
        if progress >= 0.5:
            return self.update("information-gain", intensity=progress)
        return self.update("task-succeeded")
