# SPDX-License-Identifier: AGPL-3.0-or-later
"""SQLite-backed, inspectable memory store (MVP-1 baseline).

This is the *engineering* baseline from ``docs/architecture.md`` §2.2, not the
proposed neural memory slots. Everything is a plain row you can read, audit and
revoke:

* three memory kinds: ``working`` / ``episodic`` / ``semantic``;
* provenance fields (``source``, ``source_ref``) on every record;
* soft deletion with a reason (``archive`` / ``revoke``), plus hard purge;
* consolidation that links a summary back to the records it replaced;
* keyword retrieval with a deterministic, explainable score.

The score is deliberately simple and documented:

    coverage  = matched query tokens / total query tokens
    recency   = exp(-age_days / 30)
    score     = 0.55 * coverage + 0.20 * recency
              + 0.15 * importance + 0.10 * confidence

No embeddings, no vector index, no server. If this baseline cannot be beaten by
a stronger retriever, the proposal has bigger problems than its memory design.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

KINDS: tuple[str, ...] = ("working", "episodic", "semantic")
STATUSES: tuple[str, ...] = ("active", "archived", "revoked")
FORGET_MODES: tuple[str, ...] = ("archive", "revoke")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('working', 'episodic', 'semantic')),
    title        TEXT NOT NULL DEFAULT '',
    content      TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'unknown',
    source_ref   TEXT NOT NULL DEFAULT '',
    confidence   REAL NOT NULL DEFAULT 0.5,
    importance   REAL NOT NULL DEFAULT 0.5,
    tags         TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active', 'archived', 'revoked')),
    supersedes   TEXT NOT NULL DEFAULT '',
    note         TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    accessed_at  TEXT,
    access_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS memories_kind_idx    ON memories (kind);
CREATE INDEX IF NOT EXISTS memories_status_idx  ON memories (status);
CREATE INDEX IF NOT EXISTS memories_created_idx ON memories (created_at DESC);
"""

_WORD_RE = re.compile(r"[0-9a-zA-Z_]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def utcnow() -> str:
    """Current UTC time as a sortable ISO-8601 string (second resolution)."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a number, got {value!r}") from exc
    if math.isnan(number) or math.isinf(number):
        raise ValueError("expected a finite number")
    return max(low, min(high, number))


def tokenize(text: str) -> list[str]:
    """Split mixed Latin/CJK text into comparable tokens.

    Latin runs of two or more characters are kept whole. CJK text has no spaces,
    so each character plus each adjacent character pair (bigram) becomes a token.
    This keeps retrieval honest for Chinese input without pulling in a tokenizer
    dependency.
    """

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = [w for w in _WORD_RE.findall(normalized) if len(w) >= 2]

    cjk = _CJK_RE.findall(normalized)
    if cjk:
        tokens.extend(cjk)
        tokens.extend("".join(pair) for pair in zip(cjk, cjk[1:]))

    # de-duplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


@dataclass(frozen=True)
class Memory:
    """One stored memory record."""

    id: str
    kind: str
    content: str
    source: str
    source_ref: str
    confidence: float
    importance: float
    tags: tuple[str, ...]
    status: str
    supersedes: tuple[str, ...]
    note: str
    created_at: str
    updated_at: str
    accessed_at: str | None
    access_count: int
    title: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "source_ref": self.source_ref,
            "confidence": self.confidence,
            "importance": self.importance,
            "tags": list(self.tags),
            "status": self.status,
            "supersedes": list(self.supersedes),
            "note": self.note,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
        }

    def summary(self, limit: int = 80) -> str:
        text = " ".join(self.content.split())
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return f"[{self.kind}/{self.status}] {text}"


@dataclass(frozen=True)
class SearchHit:
    """A memory plus why it matched."""

    memory: Memory
    score: float
    matched_tokens: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "matched_tokens": list(self.matched_tokens),
            "memory": self.memory.to_dict(),
        }


class MemoryError_(Exception):
    """Base error for this module (named to avoid shadowing the builtin)."""


class MemoryNotFound(MemoryError_):
    pass


class MemoryStore:
    """A small, auditable memory store backed by SQLite.

    Pass ``":memory:"`` for an ephemeral store (tests, demos) or a file path for
    a persistent one. The store is safe to close and reopen.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- write path -------------------------------------------------------

    def write(
        self,
        content: str,
        *,
        kind: str = "episodic",
        title: str = "",
        source: str = "unknown",
        source_ref: str = "",
        confidence: float = 0.5,
        importance: float = 0.5,
        tags: Sequence[str] = (),
        supersedes: Sequence[str] = (),
    ) -> Memory:
        """Store one memory and return it.

        ``source``/``source_ref`` are required by policy: a memory whose origin
        cannot be stated should not be written. The defaults exist so that
        experiments can be quick, but callers are expected to pass real values.
        """

        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        text = (content or "").strip()
        if not text:
            raise ValueError("content must not be empty")
        if not source or not source.strip():
            raise ValueError("source must not be empty")

        timestamp = utcnow()
        record = Memory(
            id=uuid.uuid4().hex[:16],
            kind=kind,
            title=title.strip(),
            content=text,
            source=source.strip(),
            source_ref=source_ref.strip(),
            confidence=_clamp(confidence),
            importance=_clamp(importance),
            tags=tuple(sorted({t.strip() for t in tags if t.strip()})),
            status="active",
            supersedes=tuple(dict.fromkeys(supersedes)),
            note="",
            created_at=timestamp,
            updated_at=timestamp,
            accessed_at=None,
            access_count=0,
        )
        self._insert(record)
        return record

    def _insert(self, record: Memory) -> None:
        self._conn.execute(
            """
            INSERT INTO memories (id, kind, title, content, source, source_ref,
                                  confidence, importance, tags, status, supersedes,
                                  note, created_at, updated_at, accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.kind,
                record.title,
                record.content,
                record.source,
                record.source_ref,
                record.confidence,
                record.importance,
                ",".join(record.tags),
                record.status,
                ",".join(record.supersedes),
                record.note,
                record.created_at,
                record.updated_at,
                record.accessed_at,
                record.access_count,
            ),
        )
        self._conn.commit()

    # -- read path --------------------------------------------------------

    def get(self, memory_id: str, *, touch: bool = False) -> Memory:
        """Fetch one memory by id. ``touch`` records an access."""

        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise MemoryNotFound(f"no memory with id {memory_id!r}")
        if touch:
            now = utcnow()
            self._conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (now, memory_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row)

    def list(
        self,
        *,
        kind: str | None = None,
        status: str | None = "active",
        limit: int = 50,
    ) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE 1 = 1"
        params: list[object] = []
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        limit: int = 5,
        include_archived: bool = False,
        exclude_sources: Sequence[str] = (),
        touch: bool = False,
    ) -> list[SearchHit]:
        """Keyword retrieval with an explainable score.

        Returns at most ``limit`` hits sorted by score. ``revoked`` records are
        never returned; ``archived`` ones are excluded unless requested.

        ``exclude_sources`` exists for a specific failure mode: an agent that
        writes its own answers to memory will happily retrieve them later and
        treat them as independent evidence. Callers should usually exclude their
        own writing source (the baseline agent excludes ``"agent"``).
        """

        tokens = tokenize(query)
        if not tokens:
            return []

        statuses = ("active", "archived") if include_archived else ("active",)
        placeholders = ",".join("?" for _ in statuses)
        sql = f"SELECT * FROM memories WHERE status IN ({placeholders})"
        params: list[object] = list(statuses)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        blocked = [source for source in exclude_sources if source]
        if blocked:
            sql += " AND source NOT IN (" + ",".join("?" for _ in blocked) + ")"
            params.extend(blocked)

        # cheap pre-filter: any token must appear in content/title/tags
        like_clauses = []
        for token in tokens[:12]:
            like_clauses.append("(content LIKE ? OR title LIKE ? OR tags LIKE ?)")
            params.extend([f"%{token}%", f"%{token}%", f"%{token}%"])
        sql += " AND (" + " OR ".join(like_clauses) + ")"

        rows = self._conn.execute(sql, tuple(params)).fetchall()
        now = datetime.now(timezone.utc)
        hits: list[SearchHit] = []
        for row in rows:
            memory = self._row_to_memory(row)
            haystack = f"{memory.title}\n{memory.content}\n{','.join(memory.tags)}".lower()
            matched = tuple(t for t in tokens if t in haystack)
            if not matched:
                continue
            coverage = len(matched) / len(tokens)
            score = (
                0.55 * coverage
                + 0.20 * self._recency(memory.created_at, now)
                + 0.15 * memory.importance
                + 0.10 * memory.confidence
            )
            hits.append(SearchHit(memory=memory, score=score, matched_tokens=matched))

        hits.sort(key=lambda hit: (hit.score, hit.memory.created_at), reverse=True)
        hits = hits[: max(1, int(limit))]

        if touch and hits:
            ids = [(utcnow(), hit.memory.id) for hit in hits]
            self._conn.executemany(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                ids,
            )
            self._conn.commit()
        return hits

    @staticmethod
    def _recency(created_at: str, now: datetime) -> float:
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return 0.0
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        return math.exp(-age_days / 30.0)

    # -- update / forget / consolidate ------------------------------------

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        title: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        tags: Sequence[str] | None = None,
        note: str | None = None,
    ) -> Memory:
        """Revise a record in place. Revision is auditable only via ``note``.

        The baseline keeps no full revision history on purpose: history is the
        harder problem (see H1 in ``docs/architecture.md``) and pretending to
        solve it in the baseline would hide the gap.
        """

        current = self.get(memory_id)
        fields: dict[str, object] = {}
        if content is not None:
            text = content.strip()
            if not text:
                raise ValueError("content must not be empty")
            fields["content"] = text
        if title is not None:
            fields["title"] = title.strip()
        if importance is not None:
            fields["importance"] = _clamp(importance)
        if confidence is not None:
            fields["confidence"] = _clamp(confidence)
        if tags is not None:
            fields["tags"] = ",".join(sorted({t.strip() for t in tags if t.strip()}))
        if note is not None:
            fields["note"] = note.strip()
        if not fields:
            return current

        fields["updated_at"] = utcnow()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        self._conn.execute(
            f"UPDATE memories SET {assignments} WHERE id = ?",
            (*fields.values(), memory_id),
        )
        self._conn.commit()
        return self.get(memory_id)

    def forget(
        self,
        memory_id: str,
        *,
        mode: str = "archive",
        reason: str = "",
    ) -> Memory:
        """Stop a memory from being recalled, while keeping it auditable.

        ``archive`` hides it from default retrieval; ``revoke`` additionally
        marks it as retracted (e.g. source turned out to be wrong). Both keep the
        row so the decision can be reviewed. Use :meth:`purge` to erase it.
        """

        if mode not in FORGET_MODES:
            raise ValueError(f"mode must be one of {FORGET_MODES}, got {mode!r}")
        if not reason.strip():
            raise ValueError("a forget reason is required for auditability")
        self.get(memory_id)  # raises if missing
        status = {"archive": "archived", "revoke": "revoked"}[mode]
        self._conn.execute(
            "UPDATE memories SET status = ?, note = ?, updated_at = ? WHERE id = ?",
            (status, reason.strip(), utcnow(), memory_id),
        )
        self._conn.commit()
        return self.get(memory_id)

    def purge(self, memory_id: str) -> None:
        """Hard-delete a record. Irreversible; prefer :meth:`forget`."""

        self.get(memory_id)
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def consolidate(
        self,
        memory_ids: Iterable[str],
        summary: str,
        *,
        kind: str = "semantic",
        title: str = "",
        source: str = "consolidation",
        source_ref: str = "",
        confidence: float = 0.7,
        importance: float = 0.6,
    ) -> Memory:
        """Fold several memories into one summary and archive the originals.

        The summary keeps ``supersedes`` = the source ids so the chain stays
        recoverable. Originals are archived (not purged) with a reason.
        """

        ids = [memory_id for memory_id in dict.fromkeys(memory_ids) if memory_id]
        if not ids:
            raise ValueError("consolidate() needs at least one memory id")
        for memory_id in ids:
            self.get(memory_id)

        merged_source_ref = source_ref.strip()
        if not merged_source_ref:
            merged_source_ref = "consolidate:" + ",".join(ids)

        merged = self.write(
            summary,
            kind=kind,
            title=title,
            source=source,
            source_ref=merged_source_ref,
            confidence=confidence,
            importance=importance,
            supersedes=ids,
        )
        reason = f"consolidated into {merged.id}"
        for memory_id in ids:
            self._conn.execute(
                "UPDATE memories SET status = 'archived', note = ?, updated_at = ? WHERE id = ?",
                (reason, utcnow(), memory_id),
            )
        self._conn.commit()
        return merged

    # -- stats ------------------------------------------------------------

    def stats(self) -> dict:
        by_kind = {
            row["kind"]: row["n"]
            for row in self._conn.execute(
                "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind"
            )
        }
        by_status = {
            row["status"]: row["n"]
            for row in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
            )
        }
        total = sum(by_status.values())
        return {
            "path": self.path,
            "total": total,
            "active": by_status.get("active", 0),
            "archived": by_status.get("archived", 0),
            "revoked": by_status.get("revoked", 0),
            "by_kind": {kind: by_kind.get(kind, 0) for kind in KINDS},
        }

    # -- internals --------------------------------------------------------

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            source=row["source"],
            source_ref=row["source_ref"],
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            tags=tuple(t for t in row["tags"].split(",") if t),
            status=row["status"],
            supersedes=tuple(s for s in row["supersedes"].split(",") if s),
            note=row["note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            accessed_at=row["accessed_at"],
            access_count=int(row["access_count"]),
        )
