# SPDX-License-Identifier: AGPL-3.0-or-later
"""Permission-scoped tool registry (MVP-1 baseline).

Design points that matter for the research questions:

* **Tools are actions, not text.** Each tool declares typed parameters, and every
  call is validated before it runs.
* **Least privilege by default.** Side effects are classified as
  ``read`` / ``write`` / ``external``; write and external tools are refused
  unless the registry was constructed to allow them.
* **Path confinement.** File tools cannot escape the configured root, including
  through symlinks or ``..`` segments (the target is resolved first).
* **Auditability.** Every call yields a :class:`ToolResult` carrying a digest of
  its arguments, its duration and any error, so the trace can be replayed.

What this is *not*: a process-level sandbox. There is no seccomp, no container and
no network isolation. A tool's handler runs in-process. Treat write/external
tools as trusted code, not as untrusted input handling.
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SIDE_EFFECTS: tuple[str, ...] = ("read", "write", "external")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base class for tool problems raised to the caller."""


class ToolNotFound(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolPermissionError(ToolError):
    pass


class PathEscapeError(ToolPermissionError):
    pass


class BudgetExceeded(ToolError):
    pass


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------

_PY_TYPES: dict[str, Callable[[Any], Any]] = {
    "str": lambda value: value if isinstance(value, str) else str(value),
    "int": lambda value: int(value),
    "float": lambda value: float(value),
    "bool": lambda value: bool(value),
}


@dataclass(frozen=True)
class Param:
    """One declared tool parameter."""

    name: str
    type: str = "str"
    required: bool = True
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.type not in _PY_TYPES:
            raise ValueError(f"unsupported param type {self.type!r}")


@dataclass(frozen=True)
class ToolSpec:
    """A tool's public contract plus its handler."""

    name: str
    description: str
    params: tuple[Param, ...]
    handler: Callable[..., Any]
    side_effect: str = "read"
    cost_units: float = 1.0

    def __post_init__(self) -> None:
        if self.side_effect not in SIDE_EFFECTS:
            raise ValueError(f"side_effect must be one of {SIDE_EFFECTS}")
        if not self.name.isidentifier():
            raise ValueError(f"tool name must be a python identifier, got {self.name!r}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "side_effect": self.side_effect,
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "default": p.default,
                    "description": p.description,
                }
                for p in self.params
            ],
        }


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool call."""

    tool: str
    ok: bool
    value: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    args_digest: str = ""

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "value": self.value,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
            "args_digest": self.args_digest,
        }


def digest(value: Any) -> str:
    """Stable short digest for audit records (not a security primitive)."""

    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# path confinement
# ---------------------------------------------------------------------------


class PathGuard:
    """Confines every path a tool touches to one root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"tool root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"tool root is not a directory: {self.root}")

    def resolve(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PathEscapeError(f"path escapes the tool root {self.root}: {relative!r}")
        return candidate


# ---------------------------------------------------------------------------
# built-in tools
# ---------------------------------------------------------------------------


def _tool_echo(text: str) -> str:
    return text


def _tool_now(fmt: str = "%Y-%m-%dT%H:%M:%S%z") -> str:
    return datetime.now().astimezone().strftime(fmt)


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
}
_ALLOWED_NAMES: dict[str, float] = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
    "tau": 6.283185307179586,
}

_MAX_NODES = 200
_MAX_POW_EXP = 64
_MAX_MAGNITUDE = 1e100


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolValidationError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POW_EXP:
                raise ToolValidationError(f"exponent too large: {right}")
            result = left**right
        elif isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolValidationError("division by zero")
        else:
            result = _BIN_OPS[type(node.op)](left, right)
        if isinstance(result, complex) or abs(result) > _MAX_MAGNITUDE:
            raise ToolValidationError("result out of allowed range")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ToolValidationError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ToolValidationError("only whitelisted functions may be called")
        if node.keywords:
            raise ToolValidationError("keyword arguments are not supported")
        return _ALLOWED_FUNCS[node.func.id](*[_eval_node(arg) for arg in node.args])
    raise ToolValidationError(f"disallowed expression node: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression without ``eval``.

    Only numbers, ``+ - * / // % **``, parentheses, unary signs and six
    whitelisted functions are permitted. Nesting and magnitude are bounded.
    """

    text = (expression or "").strip()
    if not text:
        raise ToolValidationError("empty expression")
    if len(text) > 200:
        raise ToolValidationError("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolValidationError(f"could not parse expression: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolValidationError("expression too complex")
    return _eval_node(tree)


def _read_text(guard: PathGuard, path: str, max_chars: int = 8000) -> str:
    target = guard.resolve(path)
    if not target.is_file():
        raise ToolValidationError(f"not a readable file: {path}")
    limit = max(1, min(int(max_chars), 200_000))
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(limit)


def _list_dir(guard: PathGuard, path: str = ".") -> list[str]:
    target = guard.resolve(path)
    if not target.is_dir():
        raise ToolValidationError(f"not a directory: {path}")
    return sorted(entry.name + ("/" if entry.is_dir() else "") for entry in target.iterdir())


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolPolicy:
    """What a registry is allowed to do. Defaults are intentionally narrow."""

    allow_write: bool = False
    allow_external: bool = False
    max_calls: int | None = 32
    max_output_chars: int = 20000

    def allows(self, side_effect: str) -> bool:
        if side_effect == "read":
            return True
        if side_effect == "write":
            return self.allow_write
        return self.allow_external


class ToolRegistry:
    """Registers tools, validates arguments, enforces policy, records results."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: ToolPolicy | None = None,
        register_builtins: bool = True,
    ) -> None:
        self.guard = PathGuard(root)
        self.policy = policy or ToolPolicy()
        self._tools: dict[str, ToolSpec] = {}
        self.calls = 0
        if register_builtins:
            self._register_builtin_tools()

    # -- registration -----------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ToolError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(f"unknown tool {name!r}; available: {sorted(self._tools)}") from None

    def describe(self) -> list[dict]:
        return [spec.to_dict() for spec in sorted(self._tools.values(), key=lambda s: s.name)]

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def _register_builtin_tools(self) -> None:
        self.register(
            ToolSpec(
                name="calc",
                description="Evaluate a pure arithmetic expression.",
                params=(Param("expression", "str", True, None, "e.g. '3 * (4 + 5)'"),),
                handler=safe_eval,
                side_effect="read",
            )
        )
        self.register(
            ToolSpec(
                name="now",
                description="Return the current local date and time.",
                params=(Param("fmt", "str", False, "%Y-%m-%dT%H:%M:%S%z", "strftime format"),),
                handler=_tool_now,
                side_effect="read",
            )
        )
        self.register(
            ToolSpec(
                name="echo",
                description="Return the input unchanged (diagnostics).",
                params=(Param("text", "str", True, None, "text to echo"),),
                handler=_tool_echo,
                side_effect="read",
            )
        )
        self.register(
            ToolSpec(
                name="read_text",
                description="Read a UTF-8 text file inside the tool root.",
                params=(
                    Param("path", "str", True, None, "path relative to the tool root"),
                    Param("max_chars", "int", False, 8000, "maximum characters to read"),
                ),
                handler=lambda path, max_chars=8000: _read_text(self.guard, path, max_chars),
                side_effect="read",
            )
        )
        self.register(
            ToolSpec(
                name="list_dir",
                description="List a directory inside the tool root.",
                params=(Param("path", "str", False, ".", "path relative to the tool root"),),
                handler=lambda path=".": _list_dir(self.guard, path),
                side_effect="read",
            )
        )

    # -- validation -------------------------------------------------------

    def _validate(self, spec: ToolSpec, args: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(args, Mapping):
            raise ToolValidationError("arguments must be a mapping")
        declared = {param.name for param in spec.params}
        unknown = sorted(set(args) - declared)
        if unknown:
            raise ToolValidationError(
                f"unknown argument(s) {unknown} for tool {spec.name!r}; declared: {sorted(declared)}"
            )

        cleaned: dict[str, Any] = {}
        for param in spec.params:
            if param.name in args:
                raw = args[param.name]
                try:
                    cleaned[param.name] = _PY_TYPES[param.type](raw)
                except (TypeError, ValueError) as exc:
                    raise ToolValidationError(
                        f"argument {param.name!r} expects {param.type}, got {raw!r}"
                    ) from exc
            elif param.required:
                raise ToolValidationError(f"missing required argument {param.name!r}")
            else:
                cleaned[param.name] = param.default
        return cleaned

    # -- execution --------------------------------------------------------

    def call(self, name: str, args: Mapping[str, Any] | None = None) -> ToolResult:
        """Validate and run a tool.

        Raises for unknown tools, invalid arguments, policy violations and budget
        exhaustion. Runtime failures of the handler are returned as
        ``ToolResult(ok=False, error=...)`` so callers can plan a fallback.
        """

        spec = self.get(name)
        if not self.policy.allows(spec.side_effect):
            raise ToolPermissionError(
                f"tool {name!r} has side_effect={spec.side_effect!r} which this registry disallows"
            )
        if self.policy.max_calls is not None and self.calls >= self.policy.max_calls:
            raise BudgetExceeded(f"tool budget of {self.policy.max_calls} calls exhausted")

        cleaned = self._validate(spec, args or {})
        args_digest = digest(cleaned)
        self.calls += 1

        started = time.perf_counter()
        try:
            value = spec.handler(**cleaned)
            ok = True
            error = None
        except (PathEscapeError, ToolPermissionError):
            # security-relevant refusals must never be silently downgraded
            raise
        except Exception as exc:  # noqa: BLE001 - handler failures are data, not crashes
            value = None
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        duration_ms = (time.perf_counter() - started) * 1000.0

        if isinstance(value, str) and len(value) > self.policy.max_output_chars:
            value = value[: self.policy.max_output_chars] + "…[truncated]"

        return ToolResult(
            tool=name,
            ok=ok,
            value=value,
            error=error,
            duration_ms=duration_ms,
            args_digest=args_digest,
        )
