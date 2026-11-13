# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the rule-based router baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(SRC))

from poge_ai.router import (  # noqa: E402
    extract_expression,
    extract_file_target,
    RuleRouter,
)


class ExpressionExtractionTests(unittest.TestCase):
    def test_bare_expression_is_detected(self) -> None:
        self.assertEqual(extract_expression("3 * (4 + 5) 等于多少"), "3 * (4 + 5)")

    def test_explicit_compute_prefix(self) -> None:
        self.assertEqual(extract_expression("计算 12/4"), "12/4")

    def test_date_is_not_an_expression(self) -> None:
        self.assertIsNone(extract_expression("请总结 2026-11-13 的会议记录"))

    def test_plain_text_has_no_expression(self) -> None:
        self.assertIsNone(extract_expression("帮我写一段项目介绍"))

    def test_overlong_expression_is_rejected(self) -> None:
        self.assertIsNone(extract_expression("1+" * 200))


class FileTargetTests(unittest.TestCase):
    def test_relative_path_is_found(self) -> None:
        self.assertEqual(extract_file_target("读取 docs/quickstart.md 的内容"), "docs/quickstart.md")

    def test_no_path_returns_none(self) -> None:
        self.assertIsNone(extract_file_target("讲讲你的记忆机制"))


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = RuleRouter()

    def test_arithmetic_routes_to_calc(self) -> None:
        decision = self.router.decide("3 * (4 + 5) 等于多少")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool, "calc")
        self.assertEqual(decision.tool_args["expression"], "3 * (4 + 5)")
        self.assertGreater(decision.confidence, 0.5)

    def test_temporal_routes_to_now(self) -> None:
        decision = self.router.decide("现在几点了")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool, "now")

    def test_file_reference_routes_to_read_text(self) -> None:
        decision = self.router.decide("读取 docs/quickstart.md 的前几行")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool, "read_text")
        self.assertEqual(decision.tool_args["path"], "docs/quickstart.md")

    def test_history_with_hits_routes_to_memory(self) -> None:
        decision = self.router.decide("还记得我上次说的偏好吗", memory_hits=2)
        self.assertEqual(decision.action, "memory")
        self.assertIn("prior context", decision.reason)

    def test_history_without_hits_falls_through(self) -> None:
        decision = self.router.decide("还记得我上次说的偏好吗", memory_hits=0)
        self.assertNotEqual(decision.action, "memory")

    def test_risk_without_grounding_escalates(self) -> None:
        decision = self.router.decide("这是重要的医疗决策，请确认这个剂量是否安全", memory_hits=0)
        self.assertEqual(decision.action, "review")
        self.assertIn("high-stakes", decision.reason)

    def test_short_task_is_low_confidence(self) -> None:
        decision = self.router.decide("你好")
        self.assertEqual(decision.action, "review")
        self.assertLess(decision.confidence, 0.5)

    def test_generic_task_is_internal(self) -> None:
        decision = self.router.decide("请把这段需求拆成三步并说明每一步的理由")
        self.assertEqual(decision.action, "internal")

    def test_decision_serializes(self) -> None:
        payload = self.router.decide("3 * 9").to_dict()
        self.assertEqual(payload["action"], "tool")
        self.assertEqual(payload["tool"], "calc")
        self.assertIn("arithmetic", payload["signals"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
