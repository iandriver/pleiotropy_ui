"""Tiny SQLite cache wrapping the Open Targets GraphQL client.

GraphQL responses are deterministic for a (query, variables) pair within a
release. During dev and demo we hit the same genes dozens of times — caching
those bytes keeps the live demo snappy and the API polite.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import ot_pleiotropy as ot

_LOCK = threading.Lock()
_DEFAULT_DB = Path(os.environ.get("PLEIO_CACHE_DB", Path.home() / ".pleio_cache.sqlite"))


def _key(query: str, variables: dict | None) -> str:
    blob = json.dumps({"q": query, "v": variables or {}}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS gql (
            k TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            data TEXT NOT NULL
        )"""
    )
    return conn


class GQLCache:
    """Wraps :func:`ot.gql` with a persistent on-disk cache.

    Use ``install()`` at app startup to monkey-patch ``ot.gql`` so every caller
    (including ``ot_pleiotropy.fetch_target`` etc.) is transparently cached.
    """

    def __init__(self, db_path: Path = _DEFAULT_DB, ttl_seconds: float | None = None):
        self.db_path = db_path
        self.ttl = ttl_seconds  # None = forever
        self.conn = _connect(db_path)
        self.hits = 0
        self.misses = 0
        self._original_gql = ot.gql

    def get(self, query: str, variables: dict | None) -> dict | None:
        k = _key(query, variables)
        with _LOCK:
            row = self.conn.execute("SELECT ts, data FROM gql WHERE k = ?", (k,)).fetchone()
        if row is None:
            return None
        ts, data = row
        if self.ttl is not None and (time.time() - ts) > self.ttl:
            return None
        return json.loads(data)

    def put(self, query: str, variables: dict | None, data: dict) -> None:
        k = _key(query, variables)
        with _LOCK:
            self.conn.execute(
                "INSERT OR REPLACE INTO gql (k, ts, data) VALUES (?, ?, ?)",
                (k, time.time(), json.dumps(data)),
            )
            self.conn.commit()

    def cached_gql(self, query: str, variables: dict | None = None, **kw: Any) -> dict:
        cached = self.get(query, variables)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        data = self._original_gql(query, variables, **kw)
        self.put(query, variables, data)
        return data

    def install(self) -> "GQLCache":
        """Monkey-patch ot.gql so all module-level callers use the cache."""
        ot.gql = self.cached_gql  # type: ignore[assignment]
        return self

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def clear(self) -> None:
        with _LOCK:
            self.conn.execute("DELETE FROM gql")
            self.conn.commit()


_singleton: GQLCache | None = None


def get_cache() -> GQLCache:
    global _singleton
    if _singleton is None:
        _singleton = GQLCache().install()
    return _singleton
