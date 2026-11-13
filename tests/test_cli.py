# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the command-line interface."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(SRC))

from poge_ai.cli import main  # noqa: E402


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = str(self.root / "cli.db")
        self.trace = str(self.root / "trace.jsonl")

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--db", self.db, "--trace", self.trace, "--root", str(self.root), *args])
        return code, out.getvalue(), err.getvalue()

    def test_ask_json(self) -> None:
        code, out, _ = self.run_cli("--json", "ask", "3 * (4 + 5)")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["decision"]["action"], "tool")
        self.assertEqual(payload["tool_result"]["value"], 27)

    def test_ask_text_output(self) -> None:
        code, out, _ = self.run_cli("ask", "3 * (4 + 5)")
        self.assertEqual(code, 0)
        self.assertIn("动作：tool", out)
        self.assertIn("轨迹：", out)

    def test_remember_then_recall(self) -> None:
        code, out, _ = self.run_cli(
            "remember", "用户偏好中文回答", "--kind", "semantic", "--source", "user"
        )
        self.assertEqual(code, 0)
        self.assertIn("已写入", out)

        code, out, _ = self.run_cli("recall", "回答偏好")
        self.assertEqual(code, 0)
        self.assertIn("命中", out)
        self.assertIn("用户偏好中文回答", out)

    def test_recall_without_hits(self) -> None:
        code, out, _ = self.run_cli("recall", "nothing here")
        self.assertEqual(code, 0)
        self.assertIn("没有命中", out)

    def test_stats(self) -> None:
        self.run_cli("remember", "one fact", "--source", "user")
        code, out, _ = self.run_cli("--json", "stats")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["total"], 1)

    def test_tools_listing(self) -> None:
        code, out, _ = self.run_cli("--json", "tools")
        self.assertEqual(code, 0)
        names = [tool["name"] for tool in json.loads(out)["tools"]]
        self.assertIn("calc", names)
        self.assertIn("read_text", names)

    def test_forget_requires_existing_memory(self) -> None:
        code, _, err = self.run_cli("forget", "deadbeef", "--reason", "test")
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_forget_archives(self) -> None:
        _, out, _ = self.run_cli("--json", "remember", "temporary fact", "--source", "user")
        memory_id = json.loads(out)["id"]
        code, out, _ = self.run_cli("--json", "forget", memory_id, "--reason", "stale")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "archived")

    def test_trace_command_reads_file(self) -> None:
        self.run_cli("ask", "3 * 9")
        code, out, _ = self.run_cli("--json", "trace", "--limit", "5")
        self.assertEqual(code, 0)
        events = json.loads(out)["events"]
        self.assertTrue(events)
        self.assertIn("route", {event["action"] for event in events})

    def test_openai_planner_without_env_fails_cleanly(self) -> None:
        import os

        saved = {key: os.environ.pop(key, None) for key in ("POGE_AI_BASE_URL", "POGE_AI_MODEL")}
        try:
            code, _, err = self.run_cli("ask", "hello there", "--planner", "openai")
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.assertEqual(code, 2)
        self.assertIn("POGE_AI_BASE_URL", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
