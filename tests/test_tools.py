# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for tool validation, policy and sandboxing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(SRC))

from poge_ai.tools import (  # noqa: E402
    BudgetExceeded,
    Param,
    PathEscapeError,
    ToolNotFound,
    ToolPermissionError,
    ToolPolicy,
    ToolRegistry,
    ToolSpec,
    ToolValidationError,
    safe_eval,
)


class SafeEvalTests(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        self.assertEqual(safe_eval("3 * (4 + 5)"), 27)
        self.assertEqual(safe_eval("2 ** 3"), 8)
        self.assertEqual(safe_eval("10 % 4"), 2)

    def test_unary_and_names(self) -> None:
        self.assertAlmostEqual(safe_eval("-pi"), -3.141592653589793)

    def test_whitelisted_functions(self) -> None:
        self.assertEqual(safe_eval("max(1, 5, 3)"), 5)
        self.assertEqual(safe_eval("round(2.6)"), 3)

    def test_rejects_imports(self) -> None:
        with self.assertRaises(ToolValidationError):
            safe_eval("__import__('os').system('echo hi')")

    def test_rejects_attributes(self) -> None:
        with self.assertRaises(ToolValidationError):
            safe_eval("(1).__class__")

    def test_rejects_unknown_names(self) -> None:
        with self.assertRaises(ToolValidationError):
            safe_eval("open")

    def test_rejects_huge_exponent(self) -> None:
        with self.assertRaises(ToolValidationError):
            safe_eval("2 ** 100000")

    def test_rejects_division_by_zero(self) -> None:
        with self.assertRaises(ToolValidationError):
            safe_eval("1 / 0")

    def test_rejects_empty_and_garbage(self) -> None:
        for text in ("", "   ", "4 +", "hello"):
            with self.assertRaises(ToolValidationError):
                safe_eval(text)


class RegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "notes.txt").write_text("hello sandbox", encoding="utf-8")
        (self.root / "sub").mkdir()
        self.registry = ToolRegistry(self.root)

    def test_builtin_tools_are_registered(self) -> None:
        self.assertEqual(self.registry.names, ["calc", "echo", "list_dir", "now", "read_text"])

    def test_calc_runs(self) -> None:
        result = self.registry.call("calc", {"expression": "2 * 21"})
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 42)
        self.assertTrue(result.args_digest)

    def test_unknown_tool_raises(self) -> None:
        with self.assertRaises(ToolNotFound):
            self.registry.call("rm_rf", {})

    def test_missing_required_argument(self) -> None:
        with self.assertRaises(ToolValidationError):
            self.registry.call("calc", {})

    def test_unknown_argument_is_rejected(self) -> None:
        with self.assertRaises(ToolValidationError):
            self.registry.call("calc", {"expression": "1+1", "sneaky": True})

    def test_type_coercion(self) -> None:
        result = self.registry.call("read_text", {"path": "notes.txt", "max_chars": "5"})
        self.assertTrue(result.ok)
        self.assertEqual(result.value, "hello")

    def test_read_text_within_root(self) -> None:
        result = self.registry.call("read_text", {"path": "notes.txt"})
        self.assertEqual(result.value, "hello sandbox")

    def test_path_escape_is_blocked(self) -> None:
        with self.assertRaises(PathEscapeError):
            self.registry.call("read_text", {"path": "../secret.txt"})

    def test_absolute_path_escape_is_blocked(self) -> None:
        with self.assertRaises(PathEscapeError):
            self.registry.call("read_text", {"path": "/etc/passwd"})

    def test_list_dir(self) -> None:
        result = self.registry.call("list_dir", {"path": "."})
        self.assertTrue(result.ok)
        self.assertIn("notes.txt", result.value)
        self.assertIn("sub/", result.value)

    def test_handler_error_is_returned_not_raised(self) -> None:
        result = self.registry.call("read_text", {"path": "missing.txt"})
        self.assertFalse(result.ok)
        self.assertIn("not a readable file", result.error or "")

    def test_policy_denies_write_tools(self) -> None:
        self.registry.register(
            ToolSpec(
                name="write_file",
                description="pretend to write",
                params=(Param("path", "str", True),),
                handler=lambda path: path,
                side_effect="write",
            )
        )
        with self.assertRaises(ToolPermissionError):
            self.registry.call("write_file", {"path": "x"})

    def test_policy_can_allow_write_tools(self) -> None:
        registry = ToolRegistry(self.root, policy=ToolPolicy(allow_write=True))
        registry.register(
            ToolSpec(
                name="write_file",
                description="pretend to write",
                params=(Param("path", "str", True),),
                handler=lambda path: f"wrote {path}",
                side_effect="write",
            )
        )
        result = registry.call("write_file", {"path": "x"})
        self.assertTrue(result.ok)

    def test_call_budget(self) -> None:
        registry = ToolRegistry(self.root, policy=ToolPolicy(max_calls=1))
        registry.call("now", {})
        with self.assertRaises(BudgetExceeded):
            registry.call("now", {})

    def test_long_output_is_truncated(self) -> None:
        registry = ToolRegistry(self.root, policy=ToolPolicy(max_output_chars=3))
        result = registry.call("echo", {"text": "abcdef"})
        self.assertTrue(result.ok)
        self.assertTrue(str(result.value).endswith("[truncated]"))

    def test_describe_is_serializable(self) -> None:
        described = self.registry.describe()
        self.assertEqual([item["name"] for item in described], self.registry.names)
        self.assertIn("params", described[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
