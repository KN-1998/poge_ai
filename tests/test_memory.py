# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the SQLite memory store."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(SRC))

from poge_ai.memory import (  # noqa: E402
    MemoryNotFound,
    MemoryStore,
    tokenize,
)


class TokenizeTests(unittest.TestCase):
    def test_latin_words_are_kept(self) -> None:
        self.assertIn("memory", tokenize("Memory store"))

    def test_cjk_produces_chars_and_bigrams(self) -> None:
        tokens = tokenize("回答偏好")
        self.assertIn("回", tokens)
        self.assertIn("回答", tokens)
        self.assertIn("偏好", tokens)

    def test_single_letters_are_dropped(self) -> None:
        self.assertNotIn("a", tokenize("a memory b"))

    def test_tokens_are_deduplicated_in_order(self) -> None:
        tokens = tokenize("memory memory 记忆 记忆")
        self.assertEqual(len(tokens), len(set(tokens)))


class WriteAndReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)

    def test_write_returns_typed_record(self) -> None:
        memory = self.store.write(
            "用户偏好可复现的实验步骤",
            kind="semantic",
            title="偏好",
            source="user",
            source_ref="conversation#1",
            confidence=0.9,
            importance=0.8,
            tags=["preference", "preference", "workflow"],
        )
        self.assertEqual(memory.kind, "semantic")
        self.assertEqual(memory.status, "active")
        self.assertEqual(memory.tags, ("preference", "workflow"))
        self.assertEqual(memory.confidence, 0.9)

    def test_empty_content_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.write("   ", source="user")

    def test_missing_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.write("something", source="")

    def test_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.write("x", kind="epic", source="user")

    def test_scores_are_clamped(self) -> None:
        memory = self.store.write("x", source="user", confidence=5.0, importance=-2.0)
        self.assertEqual(memory.confidence, 1.0)
        self.assertEqual(memory.importance, 0.0)

    def test_get_missing_raises(self) -> None:
        with self.assertRaises(MemoryNotFound):
            self.store.get("no-such-id")

    def test_get_touch_records_access(self) -> None:
        memory = self.store.write("x", source="user")
        updated = self.store.get(memory.id, touch=True)
        self.assertEqual(updated.access_count, 1)
        self.assertIsNotNone(updated.accessed_at)

    def test_list_filters_by_kind_and_status(self) -> None:
        self.store.write("working item", kind="working", source="test")
        self.store.write("semantic item", kind="semantic", source="test")
        working = self.store.list(kind="working")
        self.assertEqual(len(working), 1)
        self.assertEqual(working[0].content, "working item")


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)

    def test_chinese_query_hits_relevant_memory(self) -> None:
        target = self.store.write("用户偏好中文回答，并要求给出可复现的实验步骤", kind="semantic", source="user")
        self.store.write("服务器重启后需要重建索引", kind="semantic", source="ops")

        hits = self.store.search("回答偏好")

        self.assertTrue(hits)
        self.assertEqual(hits[0].memory.id, target.id)
        self.assertGreater(hits[0].score, 0.0)
        self.assertTrue(hits[0].matched_tokens)

    def test_limit_is_respected(self) -> None:
        for index in range(5):
            self.store.write(f"memory about database number {index}", source="test")
        self.assertLessEqual(len(self.store.search("database", limit=2)), 2)

    def test_empty_query_returns_nothing(self) -> None:
        self.store.write("anything", source="test")
        self.assertEqual(self.store.search("   "), [])

    def test_revoked_and_archived_are_excluded_by_default(self) -> None:
        active = self.store.write("deployment checklist", source="test")
        archived = self.store.write("deployment checklist old", source="test")
        revoked = self.store.write("deployment checklist wrong", source="test")
        self.store.forget(archived.id, mode="archive", reason="superseded")
        self.store.forget(revoked.id, mode="revoke", reason="source was wrong")

        ids = [hit.memory.id for hit in self.store.search("deployment checklist", limit=10)]
        self.assertIn(active.id, ids)
        self.assertNotIn(archived.id, ids)
        self.assertNotIn(revoked.id, ids)

    def test_include_archived_opt_in(self) -> None:
        memory = self.store.write("legacy migration notes", source="test")
        self.store.forget(memory.id, mode="archive", reason="done")
        hits = self.store.search("migration", include_archived=True)
        self.assertEqual([hit.memory.id for hit in hits], [memory.id])

    def test_exclude_sources_filters_self_written_memory(self) -> None:
        kept = self.store.write("deployment notes", source="user")
        self.store.write("deployment notes generated by the agent", source="agent")

        hits = self.store.search("deployment notes", exclude_sources=("agent",))

        self.assertEqual([hit.memory.id for hit in hits], [kept.id])

    def test_ranking_prefers_token_coverage(self) -> None:
        weak = self.store.write("实验", source="test")
        strong = self.store.write("实验步骤 可复现", source="test")
        hits = self.store.search("可复现实验步骤")
        ranked = [hit.memory.id for hit in hits]
        self.assertLess(ranked.index(strong.id), ranked.index(weak.id))


class ForgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)
        self.memory = self.store.write("temporary fact", source="test")

    def test_forget_requires_reason(self) -> None:
        with self.assertRaises(ValueError):
            self.store.forget(self.memory.id, reason="  ")

    def test_forget_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            self.store.forget(self.memory.id, mode="delete", reason="why not")

    def test_archive_keeps_row(self) -> None:
        archived = self.store.forget(self.memory.id, mode="archive", reason="stale")
        self.assertEqual(archived.status, "archived")
        self.assertEqual(archived.note, "stale")
        self.assertEqual(self.store.get(self.memory.id).status, "archived")

    def test_purge_removes_row(self) -> None:
        self.store.purge(self.memory.id)
        with self.assertRaises(MemoryNotFound):
            self.store.get(self.memory.id)


class ConsolidateTests(unittest.TestCase):
    def test_consolidation_links_and_archives_sources(self) -> None:
        store = MemoryStore(":memory:")
        self.addCleanup(store.close)
        first = store.write("实验需要固定随机种子", source="user")
        second = store.write("实验需要记录硬件型号", source="user")

        summary = store.consolidate([first.id, second.id], "实验规范：固定种子并记录硬件", title="实验规范")

        self.assertEqual(summary.kind, "semantic")
        self.assertEqual(set(summary.supersedes), {first.id, second.id})
        self.assertEqual(store.get(first.id).status, "archived")
        self.assertEqual(store.get(second.id).status, "archived")
        hits = store.search("实验规范")
        self.assertEqual(hits[0].memory.id, summary.id)

    def test_consolidate_needs_ids(self) -> None:
        store = MemoryStore(":memory:")
        self.addCleanup(store.close)
        with self.assertRaises(ValueError):
            store.consolidate([], "nothing")


class UpdateAndStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)

    def test_update_changes_fields(self) -> None:
        memory = self.store.write("draft", source="test")
        updated = self.store.update(memory.id, content="final", importance=0.9, tags=["done"])
        self.assertEqual(updated.content, "final")
        self.assertEqual(updated.importance, 0.9)
        self.assertEqual(updated.tags, ("done",))
        self.assertNotEqual(updated.updated_at, "")

    def test_update_without_fields_is_noop(self) -> None:
        memory = self.store.write("draft", source="test")
        self.assertEqual(self.store.update(memory.id).updated_at, memory.updated_at)

    def test_stats_counts(self) -> None:
        self.store.write("a", kind="working", source="test")
        self.store.write("b", kind="semantic", source="test")
        forgotten = self.store.write("c", kind="semantic", source="test")
        self.store.forget(forgotten.id, mode="revoke", reason="wrong")

        stats = self.store.stats()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["active"], 2)
        self.assertEqual(stats["revoked"], 1)
        self.assertEqual(stats["by_kind"]["semantic"], 2)


class PersistenceTests(unittest.TestCase):
    def test_reopen_keeps_data(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            with MemoryStore(path) as store:
                memory_id = store.write("persisted", source="test").id
            with MemoryStore(path) as store:
                self.assertEqual(store.get(memory_id).content, "persisted")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
