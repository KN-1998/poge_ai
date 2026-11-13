# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit rule-based action router (MVP-1 baseline).

The router decides, for one task, whether to answer internally, consult memory,
call a tool, or escalate to review. It is intentionally a *rule* baseline: it is
inspectable, brittle, and easy to beat. ``docs/research_agenda.md`` (MVP-3)
requires that any learned router be compared against it.

Signals are derived from the task text only. Memory hit counts come from the
caller, so routing and retrieval stay separable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ACTIONS: tuple[str, ...] = ("internal", "memory", "tool", "review")

#: action -> default tool when the router picks ``tool``
_TEMPORAL_RE = re.compile(
    r"(现在|当前|今天|此刻|几点|日期|时间|星期|now|today|current|clock|date)", re.IGNORECASE
)
_HISTORY_RE = re.compile(
    r"(上次|之前|以前|曾经|记得|记忆|偏好|习惯|我们说过|上次说|remember|previously|"
    r"last time|earlier|as before|my preference)",
    re.IGNORECASE,
)
_RISK_RE = re.compile(
    r"(重要|严肃|决策|审批|医疗|法律|财务|安全|不可逆|确保|核实|验证|"
    r"important|critical|verify|validate|high.stakes)",
    re.IGNORECASE,
)
_FILE_RE = re.compile(
    r"([\w./\\-]+\.(?:md|txt|py|json|toml|ya?ml|csv|log|ini|cfg))\b", re.IGNORECASE
)
_ARITH_RE = re.compile(r"(?:计算|算一下|calc(?:ulate)?|compute)[:：]?\s*(.+)$", re.IGNORECASE)
_MATH_CHARS_RE = re.compile(r"^[\s0-9+\-*/%().]+$")
_OPERATOR_RE = re.compile(r"[+\-*/%]")
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")

MAX_EXPRESSION_CHARS = 200


@dataclass(frozen=True)
class Signals:
    """Observable features extracted from a task string."""

    arithmetic: bool
    temporal: bool
    historical: bool
    risky: bool
    file_target: str | None
    memory_hits: int
    length: int

    def labels(self) -> list[str]:
        names: list[str] = []
        if self.arithmetic:
            names.append("arithmetic")
        if self.temporal:
            names.append("temporal")
        if self.historical:
            names.append("historical")
        if self.risky:
            names.append("risky")
        if self.file_target:
            names.append("file")
        if self.memory_hits:
            names.append(f"memory_hits={self.memory_hits}")
        return names


@dataclass(frozen=True)
class Decision:
    """One routing decision with its justification."""

    action: str
    confidence: float
    reason: str
    signals: Signals
    tool: str | None = None
    tool_args: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "tool": self.tool,
            "tool_args": dict(self.tool_args),
            "signals": self.signals.labels(),
        }


def extract_expression(task: str) -> str | None:
    """Pull a bare arithmetic expression out of a natural-language task."""

    candidate: str | None = None
    match = _ARITH_RE.search(task)
    if match:
        candidate = match.group(1).strip()

    if candidate is None:
        # look for any run of characters that is pure arithmetic and has an operator
        for chunk in re.findall(r"[\d\s+\-*/%().]+", task):
            text = chunk.strip()
            if len(text) >= 3 and _OPERATOR_RE.search(text) and _MATH_CHARS_RE.match(text):
                candidate = text
                break

    if not candidate or not _OPERATOR_RE.search(candidate):
        return None
    if len(candidate) > MAX_EXPRESSION_CHARS:
        return None
    # "2026-11-13" is a date, not a subtraction; do not let it reach the calculator
    if _DATE_RE.match(candidate.strip()):
        return None
    return candidate


def extract_file_target(task: str) -> str | None:
    match = _FILE_RE.search(task)
    return match.group(1) if match else None


class RuleRouter:
    """Deterministic router used as the comparison baseline."""

    def __init__(self, *, memory_hit_threshold: int = 2) -> None:
        self.memory_hit_threshold = memory_hit_threshold

    def signals(self, task: str, *, memory_hits: int = 0) -> Signals:
        text = task or ""
        return Signals(
            arithmetic=extract_expression(text) is not None,
            temporal=bool(_TEMPORAL_RE.search(text)),
            historical=bool(_HISTORY_RE.search(text)),
            risky=bool(_RISK_RE.search(text)),
            file_target=extract_file_target(text),
            memory_hits=int(memory_hits),
            length=len(text.strip()),
        )

    def decide(self, task: str, *, memory_hits: int = 0) -> Decision:
        signals = self.signals(task, memory_hits=memory_hits)

        # 1. deterministic computation beats generation
        expression = extract_expression(task)
        if expression is not None:
            return Decision(
                action="tool",
                confidence=0.85,
                reason=f"task contains a computable expression: {expression!r}",
                signals=signals,
                tool="calc",
                tool_args={"expression": expression},
            )

        # 2. anything asking for "now" cannot be answered from parameters
        if signals.temporal:
            return Decision(
                action="tool",
                confidence=0.8,
                reason="task asks for time/date information",
                signals=signals,
                tool="now",
                tool_args={},
            )

        # 3. explicit local file reference
        if signals.file_target:
            return Decision(
                action="tool",
                confidence=0.7,
                reason=f"task references a local file: {signals.file_target!r}",
                signals=signals,
                tool="read_text",
                tool_args={"path": signals.file_target},
            )

        # 4. recalled history
        if signals.historical and memory_hits > 0:
            return Decision(
                action="memory",
                confidence=0.75,
                reason="task refers to prior context and memory has hits",
                signals=signals,
            )

        # 5. enough memory hits to be worth grounding the answer
        if signals.memory_hits >= self.memory_hit_threshold:
            return Decision(
                action="memory",
                confidence=0.6,
                reason=f"{signals.memory_hits} memory hits meet the threshold",
                signals=signals,
            )

        # 6. high-stakes and ungrounded -> escalate before answering
        if signals.risky and signals.memory_hits == 0:
            return Decision(
                action="review",
                confidence=0.4,
                reason="high-stakes task without grounding; escalate for review",
                signals=signals,
            )

        # 7. very short or empty tasks carry little signal
        if signals.length < 6:
            return Decision(
                action="review",
                confidence=0.3,
                reason="task too short to route confidently",
                signals=signals,
            )

        return Decision(
            action="internal",
            confidence=0.55,
            reason="no tool, memory or risk signal matched",
            signals=signals,
        )
