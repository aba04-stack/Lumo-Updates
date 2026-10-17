"""Explicit, bounded interface to persistent episodic memory."""

from __future__ import annotations

import uuid
from typing import Any

from .store import Store


class EpisodicMemory:
    """Stores concise, user-approved summaries rather than silently archiving raw inputs."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def remember(
        self, task_type: str, summary: str, *, tags: list[str] | tuple[str, ...] = (),
        strategy_id: str | None, score: float, success: bool,
        metadata: dict[str, Any] | None = None, episode_id: str | None = None,
    ) -> dict[str, Any]:
        if episode_id is None:
            episode_id = f"episode:{uuid.uuid4().hex}"
        return self.store.remember_episode(
            episode_id, task_type, summary, tags=tags, strategy_id=strategy_id,
            score=score, success=success, metadata=metadata,
        )

    def recall(
        self, query: str = "", *, task_type: str | None = None,
        tags: list[str] | tuple[str, ...] = (), limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self.store.search_episodes(query, task_type=task_type, tags=tags, limit=limit)

    def recent(self, *, task_type: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        return self.store.search_episodes("", task_type=task_type, limit=limit)
