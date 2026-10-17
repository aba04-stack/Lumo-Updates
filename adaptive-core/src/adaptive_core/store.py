"""Transactional SQLite storage and append-only event history."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from statistics import fmean
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import StoredStrategy, StrategyConfig, validate_identifier, validate_score


_STATUSES = {"candidate", "active", "disabled"}
_TRANSITIONS = {
    "candidate": {"active", "disabled"},
    "active": {"disabled"},
    "disabled": {"active"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-serializable") from exc


class Store:
    """Thread-safe local store. Strategy records are immutable after insertion."""

    def __init__(self, db_path: str | Path = "adaptive-core.sqlite3") -> None:
        raw_path = str(db_path)
        if raw_path != ":memory:":
            Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(Path(raw_path).expanduser())
        self._lock = threading.RLock()
        self._savepoint_counter = 0
        self._connection = sqlite3.connect(
            raw_path, timeout=10.0, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            if raw_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version > 0),
                    instructions TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'disabled')),
                    parent_id TEXT REFERENCES strategies(strategy_id),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_strategies_task_status
                    ON strategies(task_type, status, created_at);
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    strategy_id TEXT NOT NULL REFERENCES strategies(strategy_id),
                    score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
                    success INTEGER NOT NULL CHECK(success IN (0, 1)),
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observations_task_strategy
                    ON observations(task_type, strategy_id);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    task_type TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_type_id ON events(event_type, id);
                CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_type, id);
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    strategy_id TEXT REFERENCES strategies(strategy_id),
                    score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
                    success INTEGER NOT NULL CHECK(success IN (0, 1)),
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_task_time
                    ON episodes(task_type, created_at);
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Compose writes atomically, using savepoints for nested store operations.

        The lock remains held for the complete transaction, so another thread
        cannot accidentally join this connection's in-flight transaction.
        """
        with self._lock:
            nested = self._connection.in_transaction
            savepoint: str | None = None
            if nested:
                self._savepoint_counter += 1
                savepoint = f"adaptive_sp_{self._savepoint_counter}"
                self._connection.execute(f"SAVEPOINT {savepoint}")
            else:
                self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                if savepoint is not None:
                    self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                elif self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            else:
                try:
                    if savepoint is not None:
                        self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                    else:
                        self._connection.execute("COMMIT")
                except BaseException:
                    if savepoint is not None and self._connection.in_transaction:
                        self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                    elif self._connection.in_transaction:
                        self._connection.execute("ROLLBACK")
                    raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> StoredStrategy:
        parameters = json.loads(row["parameters_json"])
        config = StrategyConfig(
            strategy_id=row["strategy_id"],
            task_type=row["task_type"],
            version=row["version"],
            instructions=row["instructions"],
            parameters=parameters,
        )
        return StoredStrategy(
            config=config,
            status=row["status"],
            parent_id=row["parent_id"],
            created_at=row["created_at"],
        )

    def create_strategy(
        self,
        config: StrategyConfig,
        *,
        status: str = "candidate",
        parent_id: str | None = None,
    ) -> StoredStrategy:
        if not isinstance(config, StrategyConfig):
            raise TypeError("config must be a StrategyConfig")
        if not isinstance(status, str) or status not in {"candidate", "active"}:
            raise ValueError("new strategies must start as candidate or active")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError("parent_id must be a non-empty string or None")
            parent = self.get_strategy(parent_id)
            if parent.config.task_type != config.task_type:
                raise ValueError("parent and child strategies must have the same task_type")
        created_at = _utc_now()
        try:
            with self.transaction():
                self._connection.execute(
                    """
                    INSERT INTO strategies
                    (strategy_id, task_type, version, instructions, parameters_json,
                     status, parent_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        config.strategy_id, config.task_type, config.version,
                        config.instructions, _json_dump(config.parameters), status,
                        parent_id, created_at,
                    ),
                )
                self.record_event(
                    "strategy-created", config.task_type,
                    {"strategy_id": config.strategy_id, "status": status, "parent_id": parent_id},
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"strategy could not be inserted: {exc}") from exc
        return self.get_strategy(config.strategy_id)
    def get_strategy(self, strategy_id: str) -> StoredStrategy:
        validate_identifier(strategy_id, "strategy_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown strategy: {strategy_id}")
        return self._from_row(row)

    def list_strategies(
        self, task_type: str, *, status: str | None = None
    ) -> list[StoredStrategy]:
        validate_identifier(task_type, "task_type")
        if status is not None and (not isinstance(status, str) or status not in _STATUSES):
            raise ValueError(f"unsupported status: {status}")
        query = "SELECT * FROM strategies WHERE task_type = ?"
        args: tuple[Any, ...] = (task_type,)
        if status is not None:
            query += " AND status = ?"
            args += (status,)
        query += " ORDER BY created_at ASC, strategy_id ASC"
        with self._lock:
            rows = self._connection.execute(query, args).fetchall()
        return [self._from_row(row) for row in rows]

    def set_status(
        self, strategy_id: str, new_status: str, *, expected_status: str | None = None,
    ) -> StoredStrategy:
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("strategy_id must be a non-empty string")
        if not isinstance(new_status, str) or new_status not in _STATUSES:
            raise ValueError(f"unsupported status: {new_status}")
        if expected_status is not None and (
            not isinstance(expected_status, str) or expected_status not in _STATUSES
        ):
            raise ValueError("expected_status must be a supported status or None")
        with self.transaction():
            row = self._connection.execute(
                "SELECT task_type, status, version FROM strategies WHERE strategy_id = ?",
                (strategy_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown strategy: {strategy_id}")
            old_status = row["status"]
            if expected_status is not None and old_status != expected_status:
                raise RuntimeError(
                    f"status changed concurrently: expected {expected_status}, found {old_status}"
                )
            if old_status != new_status and new_status not in _TRANSITIONS[old_status]:
                raise ValueError(f"invalid status transition: {old_status} -> {new_status}")
            if old_status == "candidate" and new_status == "active":
                raise ValueError("candidate activation requires a successful fixed-case evaluation")
            if old_status == "active" and new_status == "disabled":
                remaining = self._connection.execute(
                    "SELECT COUNT(*) AS n FROM strategies WHERE task_type = ? AND status = 'active'",
                    (row["task_type"],),
                ).fetchone()["n"]
                if int(remaining) <= 1:
                    raise ValueError("cannot disable the last active strategy; no safe fallback would remain")
            if old_status != new_status:
                self._connection.execute(
                    "UPDATE strategies SET status = ? WHERE strategy_id = ?",
                    (new_status, strategy_id),
                )
                self.record_event(
                    "strategy-status-changed", row["task_type"],
                    {"strategy_id": strategy_id, "from": old_status, "to": new_status},
                )
        return self.get_strategy(strategy_id)
    def _promote_candidate_from_evaluation(
        self, candidate_id: str, baseline_id: str, *,
        case_ids: list[str] | tuple[str, ...],
        baseline_scores: list[float] | tuple[float, ...],
        candidate_scores: list[float] | tuple[float, ...],
        minimum_gain: float, max_case_regression: float,
    ) -> StoredStrategy:
        """Atomically promote a candidate only when its per-case evaluation passes."""
        validate_identifier(candidate_id, "candidate_id")
        validate_identifier(baseline_id, "baseline_id")
        minimum_gain, max_case_regression = validate_score(minimum_gain), validate_score(max_case_regression)
        if isinstance(case_ids, (str, bytes)) or not isinstance(case_ids, (list, tuple)):
            raise ValueError("case_ids must be a list/tuple")
        if isinstance(baseline_scores, (str, bytes)) or not isinstance(baseline_scores, (list, tuple)):
            raise ValueError("baseline_scores must be a list/tuple")
        if isinstance(candidate_scores, (str, bytes)) or not isinstance(candidate_scores, (list, tuple)):
            raise ValueError("candidate_scores must be a list/tuple")
        if not 2 <= len(case_ids) <= 10_000:
            raise ValueError("promotion requires 2-10,000 fixed evaluation cases")
        for case_id in case_ids:
            validate_identifier(case_id, "case_id")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_ids must be unique")
        if len(baseline_scores) != len(case_ids) or len(candidate_scores) != len(case_ids):
            raise ValueError("each evaluation case must have baseline and candidate scores")
        base_values, trial_values = (
            [validate_score(value) for value in baseline_scores],
            [validate_score(value) for value in candidate_scores],
        )
        baseline_mean, candidate_mean = fmean(base_values), fmean(trial_values)
        gain = candidate_mean - baseline_mean
        case_regressions = [
            max(0.0, base - trial)
            for base, trial in zip(base_values, trial_values, strict=True)
        ]
        maximum_regression = max(case_regressions, default=0.0)
        if gain < minimum_gain or maximum_regression > max_case_regression:
            raise ValueError("candidate did not pass the promotion quality gate")

        with self.transaction():
            candidate = self._connection.execute(
                "SELECT task_type, status, parent_id, version FROM strategies WHERE strategy_id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise KeyError(f"unknown strategy: {candidate_id}")
            if candidate["status"] != "candidate":
                raise ValueError("only a candidate strategy can be promoted")
            baseline = self._connection.execute(
                "SELECT task_type, status, version FROM strategies WHERE strategy_id = ?",
                (baseline_id,),
            ).fetchone()
            if baseline is None:
                raise KeyError(f"unknown baseline strategy: {baseline_id}")
            if baseline["status"] != "active":
                raise ValueError("evaluation baseline must remain active")
            if baseline["task_type"] != candidate["task_type"]:
                raise ValueError("baseline and candidate task_type must match")
            if baseline_id == candidate_id:
                raise ValueError("candidate cannot be its own baseline")
            if candidate["parent_id"] != baseline_id:
                raise ValueError("candidate lineage does not match the evaluated baseline")
            if int(candidate["version"]) <= int(baseline["version"]):
                raise ValueError("candidate version must be greater than the evaluated baseline")
            self._connection.execute(
                "UPDATE strategies SET status = 'active' WHERE strategy_id = ? AND status = 'candidate'",
                (candidate_id,),
            )
            self.record_event(
                "strategy-status-changed", candidate["task_type"],
                {"strategy_id": candidate_id, "from": "candidate", "to": "active"},
            )
            self.record_event("candidate-promoted", candidate["task_type"], {
                "candidate_id": candidate_id, "baseline_id": baseline_id,
                "candidate_score": candidate_mean, "baseline_score": baseline_mean,
                "gain": gain, "minimum_gain": minimum_gain,
                "max_case_regression": maximum_regression,
                "max_case_regression_limit": max_case_regression,
                "cases": len(case_ids),
                "case_results": [
                    {"case_id": case_id, "baseline_score": base, "candidate_score": trial, "regression": regression}
                    for case_id, base, trial, regression in zip(
                        case_ids, base_values, trial_values, case_regressions, strict=True
                    )
                ],
                "accepted": True,
                "reason": f"promoted: mean gain {gain:.3f}, max case regression {maximum_regression:.3f}",
            })
        return self.get_strategy(candidate_id)
    def disable_strategy_with_fallback(self, strategy_id: str) -> StoredStrategy:
        """Atomically disable an active strategy only when another active fallback remains."""
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("strategy_id must be a non-empty string")
        with self.transaction():
            row = self._connection.execute(
                "SELECT task_type, status FROM strategies WHERE strategy_id = ?",
                (strategy_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown strategy: {strategy_id}")
            if row["status"] != "active":
                raise ValueError("only an active strategy can be rolled back")
            remaining = self._connection.execute(
                "SELECT COUNT(*) AS n FROM strategies WHERE task_type = ? AND status = 'active'",
                (row["task_type"],),
            ).fetchone()["n"]
            if int(remaining) <= 1:
                raise ValueError("cannot disable the last active strategy; no safe fallback would remain")
            self._connection.execute(
                "UPDATE strategies SET status = 'disabled' WHERE strategy_id = ? AND status = 'active'",
                (strategy_id,),
            )
            self.record_event(
                "strategy-status-changed", row["task_type"],
                {"strategy_id": strategy_id, "from": "active", "to": "disabled"},
            )
            self.record_event("strategy-rollback", row["task_type"], {
                "disabled_strategy_id": strategy_id,
                "fallbacks": [
                    item["strategy_id"] for item in self._connection.execute(
                        "SELECT strategy_id FROM strategies WHERE task_type = ? AND status = 'active' ORDER BY strategy_id",
                        (row["task_type"],),
                    ).fetchall()
                ],
            })
        return self.get_strategy(strategy_id)
    def record_outcome(
        self, task_type: str, strategy_id: str, score: float, success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        validate_identifier(task_type, "task_type")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("strategy_id must be a non-empty string")
        score = validate_score(score)
        if not isinstance(success, bool):
            raise ValueError("success must be a bool")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary or None")
        metadata_json = _json_dump(metadata or {})
        with self.transaction():
            strategy = self._connection.execute(
                "SELECT task_type, status FROM strategies WHERE strategy_id = ?",
                (strategy_id,),
            ).fetchone()
            if strategy is None:
                raise KeyError(f"unknown strategy: {strategy_id}")
            if strategy["task_type"] != task_type:
                raise ValueError("task_type does not match the strategy")
            if strategy["status"] != "active":
                raise ValueError("only active strategies may influence live adaptation")
            cursor = self._connection.execute(
                """
                INSERT INTO observations
                (task_type, strategy_id, score, success, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_type, strategy_id, score, int(success), metadata_json, _utc_now()),
            )
            observation_id = int(cursor.lastrowid)
        return observation_id
    def stats(self, task_type: str) -> dict[str, dict[str, float]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT strategy_id, COUNT(*) AS n, AVG(score) AS avg_score,
                       AVG(success) AS success_rate
                FROM observations
                WHERE task_type = ?
                GROUP BY strategy_id
                """,
                (task_type,),
            ).fetchall()
        return {
            row["strategy_id"]: {
                "count": float(row["n"]),
                "avg_score": float(row["avg_score"]),
                "success_rate": float(row["success_rate"]),
            }
            for row in rows
        }

    def total_observations(self, task_type: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM observations WHERE task_type = ?", (task_type,)
            ).fetchone()
        return int(row["n"])

    def record_event(
        self, event_type: str, task_type: str | None, payload: dict[str, Any]
    ) -> None:
        if not isinstance(event_type, str) or not event_type.strip() or len(event_type) > 128:
            raise ValueError("event_type must contain 1-128 characters")
        if task_type is not None and (
            not isinstance(task_type, str) or not task_type.strip() or len(task_type) > 128
        ):
            raise ValueError("task_type must be a non-empty string of at most 128 characters or None")
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a dictionary")
        payload_json = _json_dump(payload)
        with self._lock:
            self._connection.execute(
                "INSERT INTO events(event_type, task_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (event_type.strip(), task_type, payload_json, _utc_now()),
            )

    def remember_episode(
        self, episode_id: str, task_type: str, summary: str, *,
        tags: list[str] | tuple[str, ...] = (), strategy_id: str | None = None,
        score: float, success: bool, metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a bounded experience summary; raw inputs are intentionally not required."""
        validate_identifier(episode_id, "episode_id")
        validate_identifier(task_type, "task_type")
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 10_000:
            raise ValueError("summary must contain 1-10,000 characters")
        score = validate_score(score)
        if not isinstance(success, bool):
            raise ValueError("success must be a bool")
        if isinstance(tags, str) or not isinstance(tags, (list, tuple)) or len(tags) > 32:
            raise ValueError("tags must be a list/tuple of at most 32 strings")
        normalized_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip() or len(tag) > 64:
                raise ValueError("each tag must contain 1-64 characters")
            normalized = tag.strip().casefold()
            if normalized not in seen_tags:
                normalized_tags.append(normalized)
                seen_tags.add(normalized)
        if strategy_id is not None:
            strategy = self.get_strategy(strategy_id)
            if strategy.config.task_type != task_type:
                raise ValueError("task_type does not match the strategy")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary or None")
        tags_json, metadata_json, created_at = (
            _json_dump(normalized_tags), _json_dump(metadata or {}), _utc_now()
        )
        try:
            with self.transaction():
                self._connection.execute(
                    """
                    INSERT INTO episodes
                    (episode_id, task_type, summary, tags_json, strategy_id, score,
                     success, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (episode_id, task_type, summary.strip(), tags_json, strategy_id,
                     score, int(success), metadata_json, created_at),
                )
                self._connection.execute(
                    """INSERT INTO events(event_type, task_type, payload_json, created_at)
                       VALUES (?, ?, ?, ?)""",
                    ("episode-recorded", task_type,
                     _json_dump({"episode_id": episode_id, "strategy_id": strategy_id,
                                 "score": score, "success": success, "tags": normalized_tags}),
                     created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"episode could not be inserted: {exc}") from exc
        return {
            "episode_id": episode_id, "task_type": task_type, "summary": summary.strip(),
            "tags": normalized_tags, "strategy_id": strategy_id, "score": score,
            "success": success, "metadata": json.loads(metadata_json), "created_at": created_at,
        }
    def search_episodes(
        self, query: str = "", *, task_type: str | None = None,
        tags: list[str] | tuple[str, ...] = (), limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Recall relevant recent episodes by lexical and tag overlap."""
        if not isinstance(query, str) or len(query) > 2_000:
            raise ValueError("query must be a string no longer than 2,000 characters")
        if task_type is not None:
            validate_identifier(task_type, "task_type")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if isinstance(tags, str) or not isinstance(tags, (list, tuple)) or len(tags) > 32:
            raise ValueError("tags must be a list/tuple of at most 32 strings")
        if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError("tags must contain only non-empty strings")
        wanted_tags = {tag.strip().casefold() for tag in tags}
        query_terms = {term.casefold() for term in re.findall(r"[^\W_]+", query, flags=re.UNICODE) if len(term) > 2}
        sql = "SELECT * FROM episodes"
        clauses: list[str] = []
        args: list[Any] = []
        if task_type is not None:
            clauses.append("task_type = ?")
            args.append(task_type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid DESC LIMIT 1000"
        with self._lock:
            rows = self._connection.execute(sql, args).fetchall()
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for recency_index, row in enumerate(rows):
            episode_tags = json.loads(row["tags_json"])
            summary_terms = {term.casefold() for term in re.findall(r"[^\W_]+", row["summary"], flags=re.UNICODE) if len(term) > 2}
            tag_hits = len(wanted_tags.intersection(episode_tags))
            term_hits = len(query_terms.intersection(summary_terms))
            relevance = 1.0 if not query_terms and not wanted_tags else float(term_hits) + 2.0 * tag_hits
            if relevance <= 0:
                continue
            item = {
                "episode_id": row["episode_id"], "task_type": row["task_type"],
                "summary": row["summary"], "tags": episode_tags, "strategy_id": row["strategy_id"],
                "score": float(row["score"]), "success": bool(row["success"]),
                "metadata": json.loads(row["metadata_json"]), "created_at": row["created_at"],
                "relevance": relevance,
            }
            ranked.append((relevance, -recency_index, item))
        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in ranked[:limit]]

    @staticmethod
    def _event_record(row: sqlite3.Row) -> dict[str, Any]:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {"_invalid_payload": True, "_payload_error": "invalid_json"}
        if not isinstance(payload, dict):
            payload = {
                "_invalid_payload": True,
                "_payload_error": "payload_not_object",
                "_payload_type": type(payload).__name__,
            }
        return {
            "id": int(row["id"]),
            "event_type": row["event_type"],
            "task_type": row["task_type"],
            "payload": payload,
            "created_at": row["created_at"],
        }

    def latest_event(self, event_type: str) -> dict[str, Any] | None:
        """Fetch the newest event of one type without a global history window."""
        if not isinstance(event_type, str) or not event_type or len(event_type) > 128:
            raise ValueError("event_type must contain 1-128 characters")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY id DESC LIMIT 1",
                (event_type,),
            ).fetchone()
        return self._event_record(row) if row is not None else None

    def events_of_types(
        self,
        event_types: list[str] | tuple[str, ...],
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        """Query selected event streams; optionally return all matching history."""
        if (
            isinstance(event_types, (str, bytes))
            or not isinstance(event_types, (list, tuple))
            or not 1 <= len(event_types) <= 32
        ):
            raise ValueError("event_types must be a list/tuple of 1-32 event names")
        if any(not isinstance(value, str) or not value or len(value) > 128 for value in event_types):
            raise ValueError("each event type must contain 1-128 characters")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100_000
        ):
            raise ValueError("limit must be None or an integer from 1 to 100,000")
        placeholders = ",".join("?" for _ in event_types)
        direction = "DESC" if newest_first else "ASC"
        query = f"SELECT * FROM events WHERE event_type IN ({placeholders}) ORDER BY id {direction}"
        args: tuple[Any, ...] = tuple(event_types)
        if limit is not None:
            query += " LIMIT ?"
            args += (limit,)
        with self._lock:
            rows = self._connection.execute(query, args).fetchall()
        return [self._event_record(row) for row in rows]

    def events_for_task(
        self,
        task_type: str,
        *,
        limit: int = 100,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Query one task's event stream without unrelated-task history noise."""
        validate_identifier(task_type, "task_type")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100_000:
            raise ValueError("limit must be an integer from 1 to 100,000")
        if not isinstance(newest_first, bool):
            raise ValueError("newest_first must be a bool")
        direction = "DESC" if newest_first else "ASC"
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM events WHERE task_type = ? ORDER BY id {direction} LIMIT ?",
                (task_type, limit),
            ).fetchall()
        return [self._event_record(row) for row in rows]

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be an integer from 1 to 10,000")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._event_record(row) for row in rows]
