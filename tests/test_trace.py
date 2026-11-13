# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for trace redaction and persistence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(SRC))

from poge_ai.trace import TraceRecorder, redact  # noqa: E402


class RedactionTests(unittest.TestCase):
    def test_secret_keys_are_masked(self) -> None:
        cleaned = redact({"api_key": "sk-123", "Authorization": "Bearer x", "model": "m"})
        self.assertEqual(cleaned["api_key"], "***")
        self.assertEqual(cleaned["Authorization"], "***")
        self.assertEqual(cleaned["model"], "m")

    def test_nested_structures_are_masked(self) -> None:
        cleaned = redact({"tool": {"args": {"password": "hunter2"}, "name": "db"}})
        self.assertEqual(cleaned["tool"]["args"]["password"], "***")
        self.assertEqual(cleaned["tool"]["name"], "db")

    def test_lists_are_masked(self) -> None:
        cleaned = redact([{"token": "abc"}, "plain"])
        self.assertEqual(cleaned[0]["token"], "***")
        self.assertEqual(cleaned[1], "plain")

    def test_long_strings_are_truncated(self) -> None:
        cleaned = redact("x" * 1000)
        self.assertLessEqual(len(cleaned), 240)
        self.assertTrue(cleaned.endswith("…"))

    def test_scalars_pass_through(self) -> None:
        self.assertEqual(redact(7), 7)
        self.assertIsNone(redact(None))
        self.assertTrue(redact(True))


class RecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "trace.jsonl"

    def test_events_are_appended_as_jsonl(self) -> None:
        with TraceRecorder(self.path, run_id="run-1") as tracer:
            tracer.record(step=0, action="recall", query="q", api_key="secret")
            tracer.record(step=1, action="route", action_name="tool")

        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["run_id"], "run-1")
        self.assertEqual(first["action"], "recall")
        self.assertEqual(first["api_key"], "***")
        self.assertNotIn("secret", self.path.read_text(encoding="utf-8"))

    def test_in_memory_recorder_returns_events(self) -> None:
        tracer = TraceRecorder()
        tracer.record(step=0, action="x")
        self.assertEqual(len(tracer.events()), 1)
        self.assertEqual(tracer.events()[0]["action"], "x")

    def test_read_skips_malformed_lines(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n', encoding="utf-8")
        events = TraceRecorder.read(self.path)
        self.assertEqual([event["ok"] for event in events], [1, 2])

    def test_read_respects_limit(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text("\n".join(json.dumps({"i": i}) for i in range(5)) + "\n", encoding="utf-8")
        events = TraceRecorder.read(self.path, limit=2)
        self.assertEqual([event["i"] for event in events], [3, 4])

    def test_read_missing_file_returns_empty(self) -> None:
        self.assertEqual(TraceRecorder.read(Path(self.tmp.name) / "nope.jsonl"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
