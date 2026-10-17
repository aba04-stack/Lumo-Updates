"""Bounded global workspace for sharing high-value signals across modules."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import validate_identifier, validate_score


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class WorkspaceSignal:
    signal_id: str
    kind: str
    source: str
    content: str
    priority: float = 0.5
    confidence: float = 0.5
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        validate_identifier(self.signal_id, "signal_id")
        validate_identifier(self.kind, "kind")
        validate_identifier(self.source, "source")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string")
        if len(self.content) > 12_000:
            raise ValueError("content exceeds 12,000 characters")
        object.__setattr__(self, "priority", validate_score(self.priority))
        object.__setattr__(self, "confidence", validate_score(self.confidence))
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at must be a non-empty string")

    @property
    def attention_score(self) -> float:
        """Priority-weighted confidence; a routing heuristic, not a truth score."""
        return self.priority * self.confidence


SignalListener = Callable[[WorkspaceSignal], None]


class GlobalWorkspace:
    """A small, bounded attention buffer with failure-isolated subscribers."""

    def __init__(self, capacity: int = 32) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._lock = threading.RLock()
        self._signals: dict[str, WorkspaceSignal] = {}
        self._listeners: dict[int, SignalListener] = {}
        self._next_listener_id = 1
        self._listener_errors: list[dict[str, str]] = []

    def subscribe(self, listener: SignalListener) -> int:
        if not callable(listener):
            raise ValueError("listener must be callable")
        with self._lock:
            token = self._next_listener_id
            self._next_listener_id += 1
            self._listeners[token] = listener
            return token

    def unsubscribe(self, token: int) -> bool:
        with self._lock:
            return self._listeners.pop(token, None) is not None

    def publish(self, signal: WorkspaceSignal) -> bool:
        """Keep a signal when there is capacity or it outranks the weakest item."""
        if not isinstance(signal, WorkspaceSignal):
            raise TypeError("signal must be a WorkspaceSignal")
        with self._lock:
            if signal.signal_id in self._signals:
                raise ValueError(f"duplicate signal_id: {signal.signal_id}")
            if len(self._signals) >= self.capacity:
                weakest = min(
                    self._signals.values(),
                    key=lambda item: (item.attention_score, item.created_at, item.signal_id),
                )
                if (signal.attention_score, signal.created_at, signal.signal_id) <= (
                    weakest.attention_score, weakest.created_at, weakest.signal_id
                ):
                    return False
                del self._signals[weakest.signal_id]
            self._signals[signal.signal_id] = signal
            listeners = tuple(self._listeners.items())

        errors: list[dict[str, str]] = []
        for token, listener in listeners:
            try:
                listener(signal)
            except Exception as exc:
                errors.append({"listener_id": str(token), "error_type": type(exc).__name__})
        if errors:
            with self._lock:
                self._listener_errors.extend(errors)
                self._listener_errors = self._listener_errors[-100:]
        return True

    def snapshot(self, limit: int | None = None) -> list[WorkspaceSignal]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be a positive integer or None")
        with self._lock:
            signals = list(self._signals.values())
        signals.sort(
            key=lambda item: (item.attention_score, item.created_at, item.signal_id),
            reverse=True,
        )
        return signals if limit is None else signals[:limit]

    def drain(self) -> list[WorkspaceSignal]:
        """Atomically retrieve and clear the transient workspace."""
        with self._lock:
            signals = list(self._signals.values())
            self._signals.clear()
        signals.sort(
            key=lambda item: (item.attention_score, item.created_at, item.signal_id),
            reverse=True,
        )
        return signals

    def listener_errors(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(item) for item in self._listener_errors]
