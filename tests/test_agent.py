# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end tests for the MVP-1 agent loop."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(SRC))

from poge_ai.agent import Agent  # noqa: E402
from poge_ai.memory import MemoryStore  # noqa: E402
from poge_ai.planner import OfflinePlanner, PlannerError  # noqa: E402
from poge_ai.tools import ToolRegistry  # noqa: E402
from poge_ai.trace import TraceRecorder  # noqa: E402


class FailingPlanner:
    name = "failing"

    def complete(self, prompt: str, *, context=()) -> str:
        raise PlannerError("backend is down")


class AgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "notes.md").write_text("# notes\nhello", encoding="utf-8")
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)
        self.tracer = TraceRecorder(self.root / "trace.jsonl", run_id="test-run")
        self.tools = ToolRegistry(self.root)

    def make_agent(self, planner=None, **kwargs) -> Agent:
        return Agent(
            store=self.store,
            tools=self.tools,
            planner=planner or OfflinePlanner(),
            tracer=self.tracer,
            **kwargs,
        )

    def test_arithmetic_task_uses_calculator(self) -> None:
        result = self.make_agent().run("3 * (4 + 5) 等于多少")

        self.assertEqual(result.decision.action, "tool")
        self.assertIsNotNone(result.tool_result)
        self.assertTrue(result.tool_result.ok)
        self.assertEqual(result.tool_result.value, 27)
        self.assertIn("27", result.answer)
        self.assertEqual(result.run_id, "test-run")

    def test_steps_are_recorded_in_order(self) -> None:
        result = self.make_agent().run("3 * (4 + 5)")
        actions = [step.action for step in result.steps]
        self.assertEqual(actions[0], "recall")
        self.assertEqual(actions[1], "route")
        self.assertIn("generate", actions)
        self.assertIn("remember", actions)
        self.assertEqual([step.index for step in result.steps], list(range(len(result.steps))))

    def test_trace_contains_route_and_tool_events(self) -> None:
        self.make_agent().run("3 * (4 + 5)")
        actions = [event["action"] for event in self.tracer.events()]
        self.assertIn("recall", actions)
        self.assertIn("route", actions)
        self.assertIn("tool", actions)
        self.assertTrue((self.root / "trace.jsonl").exists())

    def test_interaction_is_written_to_memory(self) -> None:
        agent = self.make_agent()
        agent.run("请把需求拆成三步")
        self.assertGreaterEqual(self.store.stats()["total"], 1)

    def test_own_answers_are_not_recalled_as_evidence(self) -> None:
        agent = self.make_agent()
        agent.run("请把需求拆成三步")
        second = agent.run("请把需求拆成三步")
        self.assertEqual(second.memories_used, [])

    def test_write_memory_can_be_disabled(self) -> None:
        agent = self.make_agent()
        agent.run("请把需求拆成三步", write_memory=False)
        self.assertEqual(self.store.stats()["total"], 0)

    def test_memory_grounded_task(self) -> None:
        first = self.store.write("用户偏好中文回答", kind="semantic", source="user")
        second = self.store.write("用户偏好可复现的实验步骤", kind="semantic", source="user")

        result = self.make_agent().run("还记得我上次说的偏好吗")

        self.assertEqual(result.decision.action, "memory")
        self.assertIn(first.id, result.memories_used)
        self.assertIn(second.id, result.memories_used)

    def test_tool_failure_degrades_instead_of_crashing(self) -> None:
        result = self.make_agent().run("读取 missing.md 的内容")

        self.assertEqual(result.decision.action, "tool")
        self.assertFalse(result.tool_result.ok)
        self.assertIn("降级", result.answer)

    def test_planner_failure_is_reported(self) -> None:
        result = self.make_agent(planner=FailingPlanner()).run("请把需求拆成三步")

        self.assertTrue(result.answer.startswith("[planner error]"))
        self.assertFalse(result.steps[-2].ok)

    def test_empty_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_agent().run("   ")

    def test_result_serializes_to_dict(self) -> None:
        payload = self.make_agent().run("3 * 9").to_dict()
        self.assertEqual(payload["decision"]["action"], "tool")
        self.assertIn("steps", payload)
        self.assertIsInstance(payload["memories_used"], list)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
