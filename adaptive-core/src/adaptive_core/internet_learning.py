"""Bounded Wikipedia connector and persistent, source-linked knowledge base.

This module stores retrieved reference material for later retrieval (RAG). It does
not train model weights, execute page contents, or treat external text as instructions.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


_ALLOWED_LANGUAGES = frozenset(
    {"ar", "de", "en", "es", "fr", "it", "ja", "kk", "pt", "ru", "tr", "uk", "zh"}
)
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_DOCUMENT_CHARS = 20_000


class InternetLearningError(RuntimeError):
    """Raised when a public source cannot be fetched or parsed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _terms(text: str) -> set[str]:
    return {
        term.casefold()
        for term in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        if len(term) > 2
    }


class WebKnowledgeStore:
    """Independent SQLite store for unverified internet sources."""

    def __init__(self, db_path: str | Path = "adaptive-web-knowledge.sqlite3") -> None:
        raw_path = str(db_path)
        if raw_path != ":memory:":
            Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(Path(raw_path).expanduser())
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            raw_path, timeout=10.0, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 10000")
        if raw_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS web_documents (
                source_url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                language TEXT NOT NULL,
                query_text TEXT NOT NULL,
                content TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_web_documents_fetched_at "
            "ON web_documents(fetched_at)"
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> WebKnowledgeStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _as_document(row: sqlite3.Row, relevance: float = 0.0) -> dict[str, Any]:
        return {
            "title": row["title"],
            "url": row["source_url"],
            "language": row["language"],
            "query": row["query_text"],
            "content": row["content"],
            "content_sha256": row["content_sha256"],
            "fetched_at": row["fetched_at"],
            "source_type": "wikipedia",
            "trust_level": "unverified_external_source",
            "relevance": relevance,
        }

    def save_document(
        self, *, title: str, url: str, language: str, query: str, content: str
    ) -> dict[str, Any]:
        if not isinstance(language, str) or language not in _ALLOWED_LANGUAGES:
            raise ValueError("unsupported Wikipedia language")
        if not isinstance(url, str) or not url or len(url) > 2_000:
            raise ValueError("source URL must be a non-empty string of at most 2,000 characters")
        try:
            parsed = urlparse(url)
            if (
                parsed.scheme != "https"
                or parsed.hostname != f"{language}.wikipedia.org"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port is not None
            ):
                raise ValueError("source URL must be HTTPS on the expected Wikipedia host")
        except ValueError as exc:
            if "source URL" in str(exc):
                raise
            raise ValueError("source URL must be HTTPS on the expected Wikipedia host") from exc
        if not isinstance(title, str) or not title.strip() or len(title) > 500:
            raise ValueError("title must contain 1-500 characters")
        if not isinstance(query, str) or not query.strip() or len(query) > 2_000:
            raise ValueError("query must contain 1-2,000 characters")
        if (
            not isinstance(content, str)
            or not content.strip()
            or len(content) > _MAX_DOCUMENT_CHARS
        ):
            raise ValueError("content must contain 1-20,000 characters")
        normalized_content = content.strip()
        digest = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
        fetched_at = _utc_now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO web_documents
                    (source_url, title, language, query_text, content, content_sha256, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_url) DO UPDATE SET
                    title=excluded.title,
                    language=excluded.language,
                    query_text=excluded.query_text,
                    content=excluded.content,
                    content_sha256=excluded.content_sha256,
                    fetched_at=excluded.fetched_at
                """,
                (url, title.strip(), language, query.strip(), normalized_content, digest, fetched_at),
            )
            row = self._connection.execute(
                "SELECT * FROM web_documents WHERE source_url = ?", (url,)
            ).fetchone()
        if row is None:
            raise RuntimeError("saved web document could not be read back")
        return self._as_document(row)

    def search(self, query: str = "", *, limit: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or len(query) > 2_000:
            raise ValueError("query must be a string no longer than 2,000 characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer from 1 to 20")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM web_documents ORDER BY fetched_at DESC, rowid DESC LIMIT 1000"
            ).fetchall()
        query_terms = _terms(query)
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for recency, row in enumerate(rows):
            searchable = f"{row['title']} {row['content']}"
            hits = len(query_terms.intersection(_terms(searchable)))
            if query_terms and hits == 0:
                continue
            relevance = float(hits) if query_terms else 1.0
            ranked.append((relevance, -recency, self._as_document(row, relevance)))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item for _, _, item in ranked[:limit]]


class WikipediaLearner:
    """Search Wikipedia and save bounded extracts for later grounded retrieval.

    Network access occurs only when learn() is called explicitly. Source extracts are
    unverified evidence and must not be treated as executable instructions or truth.
    """

    def __init__(
        self,
        db_path: str | Path = "adaptive-web-knowledge.sqlite3",
        *,
        store: WebKnowledgeStore | None = None,
        timeout: float = 8.0,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 1.0 <= timeout <= 30.0
        ):
            raise ValueError("timeout must be between 1 and 30 seconds")
        self.store = store or WebKnowledgeStore(db_path)
        self.timeout = timeout

    def close(self) -> None:
        self.store.close()

    def _request_json(self, language: str, params: dict[str, str]) -> dict[str, Any]:
        if not isinstance(language, str) or language not in _ALLOWED_LANGUAGES:
            raise ValueError("unsupported Wikipedia language")
        if not isinstance(params, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in params.items()):
            raise ValueError("params must be a dictionary of string keys and values")
        host = f"{language}.wikipedia.org"
        url = f"https://{host}/w/api.php?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "AdaptiveCoreLab/0.11.1 (educational research; "
                    "https://github.com/aba04-stack/Lumo-Updates)"
                ),
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                try:
                    final_url = urlparse(response.geturl())
                    final_host = final_url.hostname
                    final_port = final_url.port
                except (TypeError, ValueError) as exc:
                    raise InternetLearningError("Wikipedia redirected to an invalid URL") from exc
                if (
                    final_url.scheme != "https"
                    or final_host != host
                    or final_url.username is not None
                    or final_url.password is not None
                    or final_port is not None
                ):
                    raise InternetLearningError("Wikipedia redirected to an unexpected host or authority")
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    raise InternetLearningError("Wikipedia response exceeded the size limit")
                if response.status != 200:
                    raise InternetLearningError(f"Wikipedia returned HTTP {response.status}")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise InternetLearningError(f"Wikipedia request failed: {type(exc).__name__}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InternetLearningError("Wikipedia returned invalid JSON") from exc
        if not isinstance(payload, dict) or "error" in payload:
            raise InternetLearningError("Wikipedia returned an API error")
        return payload

    def learn(self, query: str, *, language: str = "ru", limit: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or len(query) > 2_000:
            raise ValueError("query must contain 1-2,000 characters")
        query = query.strip()
        if not isinstance(language, str) or language not in _ALLOWED_LANGUAGES:
            raise ValueError(f"unsupported language: {language}")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
            raise ValueError("limit must be an integer from 1 to 10")

        result = self._request_json(
            language,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
                "format": "json",
                "utf8": "1",
            },
        )
        if not isinstance(result, dict):
            raise InternetLearningError("Wikipedia returned an unexpected search-result structure")
        query_result = result.get("query")
        if not isinstance(query_result, dict) or not isinstance(query_result.get("search"), list):
            raise InternetLearningError("Wikipedia returned an unexpected search-result structure")
        page_ids: list[str] = []
        for item in query_result["search"]:
            if not isinstance(item, dict):
                continue
            page_id = item.get("pageid")
            if isinstance(page_id, bool) or not isinstance(page_id, (str, int)):
                continue
            normalized_id = str(page_id)
            if not normalized_id.isdigit() or int(normalized_id) <= 0:
                continue
            if normalized_id not in page_ids:
                page_ids.append(normalized_id)
            if len(page_ids) >= limit:
                break
        if not page_ids:
            return []

        details = self._request_json(
            language,
            {
                "action": "query",
                "prop": "extracts|info",
                "pageids": "|".join(page_ids),
                "explaintext": "1",
                "exsectionformat": "plain",
                "exchars": "15000",
                "inprop": "url",
                "format": "json",
                "utf8": "1",
            },
        )
        if not isinstance(details, dict):
            raise InternetLearningError("Wikipedia returned an unexpected page-detail structure")
        details_query = details.get("query")
        if not isinstance(details_query, dict):
            raise InternetLearningError("Wikipedia returned an unexpected page-detail structure")
        pages = details_query.get("pages")
        if isinstance(pages, dict):
            page_items = list(pages.values())
        elif isinstance(pages, list):
            page_items = pages
        else:
            raise InternetLearningError("Wikipedia returned an unexpected page structure")

        documents: list[dict[str, Any]] = []
        for page in page_items:
            if not isinstance(page, dict):
                continue
            title = page.get("title")
            url = page.get("fullurl")
            extract = page.get("extract")
            if not isinstance(title, str) or not isinstance(url, str) or not isinstance(extract, str):
                continue
            content = re.sub(r"\n{3,}", "\n\n", extract).strip()[:_MAX_DOCUMENT_CHARS]
            if not content:
                continue
            try:
                document = self.store.save_document(
                    title=title, url=url, language=language, query=query, content=content
                )
            except ValueError:
                # Unexpected/malformed URLs and invalid source records are not ingested.
                continue
            documents.append(document)
        return documents

    def recall(self, query: str = "", *, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve previously stored material without making a network request."""
        return self.store.search(query, limit=limit)
