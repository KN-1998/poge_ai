# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MVP-1 agent loop: recall -> route -> act -> answer -> remember.

This module is deliberately boring. It wires four inspectable components together
so that each decision can be logged, replayed and argued with:

    MemoryStore  -- what the system knows from before
    RuleRouter   -- why it chose this action
    ToolRegistry -- what it was allowed to execute
    Planner      -- what generated the final text

The loop does not claim to be smart. It exists so that later replacements
(learned router, neural memory) have something honest to be compared against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from .memory import Memory, MemoryStore, SearchHit, tokenize
from .planner import Planner, PlannerError
from .router import Decision, RuleRouter
from .tools import ToolError, ToolRegistry, ToolResult
from .trace import TraceRecorder, redact


@dataclass(frozen=True)
class Step:
    """One observable step of a run."""

    index: int
    action: str
    detail: str
    ok: bool
    duration_ms: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "action": self.action,
            "detail": self.detail,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class RunResult:
    """Everything a caller needs to audit one task."""

    task: str
    answer: str
    decision: Decision
    steps: list[Step] = field(default_factory=list)
    memories_used: list[str] = field(default_factory=list)
    tool_result: ToolResult | None = None
    run_id: str = ""
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "answer": self.answer,
            "decision": self.decision.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "memories_used": list(self.memories_used),
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "run_id": self.run_id,
            "trace_path": self.trace_path,
        }


class Agent:
    """Baseline agent. See ``docs/decisions/0003-baseline-prototype.md``."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        tools: ToolRegistry,
        planner: Planner,
        router: RuleRouter | None = None,
        tracer: TraceRecorder | None = None,
        memory_limit: int = 5,
        write_memory: bool = True,
        exclude_sources: Sequence[str] = ("agent",),
    ) -> None:
        self.store = store
        self.tools = tools
        self.planner = planner
        self.router = router or RuleRouter()
        self.tracer = tracer
        self.memory_limit = max(1, int(memory_limit))
        self.write_memory = write_memory
        # do not feed the agent's own past answers back in as evidence
        self.exclude_sources = tuple(exclude_sources)

    # -- public API -------------------------------------------------------

    def run(self, task: str, *, write_memory: bool | None = None) -> RunResult:
        text = (task or "").strip()
        if not text:
            raise ValueError("task must not be empty")

        remember = self.write_memory if write_memory is None else write_memory
        steps: list[Step] = []
        memories_used: list[str] = []

        def mark(action: str, detail: str, ok: bool, started: float) -> None:
            steps.append(
                Step(
                    index=len(steps),
                    action=action,
                    detail=detail,
                    ok=ok,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )
            )

        # 1. recall -------------------------------------------------------
        started = time.perf_counter()
        hits = self.store.search(text, limit=self.memory_limit, exclude_sources=self.exclude_sources)
        self._record(0, "recall", query=text, hits=len(hits), memories=[h.memory.id for h in hits])
        mark("recall", f"{len(hits)} memory hit(s)", True, started)

        # 2. route --------------------------------------------------------
        started = time.perf_counter()
        decision = self.router.decide(text, memory_hits=len(hits))
        self._record(
            1,
            "route",
            decision=decision.action,
            confidence=round(decision.confidence, 3),
            reason=decision.reason,
            tool=decision.tool,
        )
        mark("route", f"{decision.action} ({decision.confidence:.2f})", True, started)

        context: list[str] = []
        tool_result: ToolResult | None = None
        note: str | None = None

        # 3. act ----------------------------------------------------------
        started = time.perf_counter()
        if decision.action == "tool" and decision.tool:
            try:
                tool_result = self.tools.call(decision.tool, decision.tool_args)
                ok = tool_result.ok
                detail = (
                    f"{decision.tool} -> {redact(tool_result.value)}"
                    if ok
                    else f"{decision.tool} failed: {tool_result.error}"
                )
            except ToolError as exc:
                tool_result = None
                ok = False
                detail = f"{decision.tool} rejected: {exc}"
            self._record(2, "tool", tool=decision.tool, ok=ok, detail=detail)
            mark("tool", detail, ok, started)

            if tool_result is not None and tool_result.ok:
                context.append(f"工具 {decision.tool} 的结果：{tool_result.value}")
                if remember:
                    self._remember_tool_outcome(decision.tool, tool_result)
            else:
                note = "工具不可用，回答降级为内部生成。"
                context.append(f"（工具 {decision.tool} 调用失败：{detail}）")

        elif decision.action == "memory":
            context, memories_used = self._memory_context(hits)
            self._record(2, "memory", used=memories_used, hits=len(hits))
            mark("memory", f"using {len(memories_used)} memory record(s)", True, started)

        elif decision.action == "review":
            reviewed = self._review(text, hits)
            if reviewed:
                context, memories_used = self._memory_context(reviewed)
                detail = f"review found {len(memories_used)} grounded record(s)"
            else:
                detail = "review found no extra grounding"
                note = "该任务被判定为高风险或低置信，但没有可用证据，回答仅供参考。"
            self._record(2, "review", detail=detail, hits=len(reviewed))
            mark("review", detail, True, started)

        else:
            self._record(2, "internal", prompt_chars=len(text))
            mark("internal", "answered without external support", True, started)

        # 4. generate -----------------------------------------------------
        started = time.perf_counter()
        try:
            answer = self.planner.complete(text, context=context)
            ok = True
            detail = f"{self.planner.name} produced {len(answer)} chars"
        except PlannerError as exc:
            answer = f"[planner error] {exc}"
            ok = False
            detail = f"planner failed: {exc}"
        self._record(3, "generate", planner=self.planner.name, ok=ok, detail=detail)
        mark("generate", detail, ok, started)

        if note:
            answer = f"{answer}\n\n（{note}）"

        # 5. remember -----------------------------------------------------
        if remember:
            started = time.perf_counter()
            memory = self._remember_interaction(text, answer, decision)
            self._record(4, "remember", memory_id=memory.id, kind=memory.kind)
            mark("remember", f"wrote {memory.id}", True, started)

        return RunResult(
            task=text,
            answer=answer,
            decision=decision,
            steps=steps,
            memories_used=memories_used,
            tool_result=tool_result,
            run_id=self.tracer.run_id if self.tracer else "",
            trace_path=str(self.tracer.path) if self.tracer and self.tracer.path else None,
        )

    # -- helpers ----------------------------------------------------------

    def _record(self, step: int, event_action: str, **fields: object) -> None:
        if self.tracer is None:
            return
        self.tracer.record(step=step, action=event_action, **fields)

    @staticmethod
    def _format_hit(hit: SearchHit) -> str:
        memory = hit.memory
        provenance = memory.source + (f" ({memory.source_ref})" if memory.source_ref else "")
        return f"[{memory.kind}] {memory.content}（来源：{provenance}，置信度 {memory.confidence:.2f}）"

    def _memory_context(self, hits: Sequence[SearchHit]) -> tuple[list[str], list[str]]:
        return [self._format_hit(hit) for hit in hits], [hit.memory.id for hit in hits]

    def _review(self, task: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Widen the recall query and merge the results with the original hits."""

        tokens = tokenize(task)
        expanded = " ".join(tokens[:20]) or task
        merged: dict[str, SearchHit] = {hit.memory.id: hit for hit in hits}
        for hit in self.store.search(
            expanded,
            limit=self.memory_limit * 2,
            include_archived=False,
            exclude_sources=self.exclude_sources,
        ):
            merged.setdefault(hit.memory.id, hit)
        return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[: self.memory_limit]

    def _remember_tool_outcome(self, tool: str, tool_result: ToolResult) -> None:
        self.store.write(
            f"工具 {tool} 返回：{tool_result.value}",
            kind="episodic",
            title=f"tool:{tool}",
            source="tool",
            source_ref=f"{tool}#{tool_result.args_digest}",
            confidence=0.9,
            importance=0.3,
            tags=(tool, "tool-output"),
        )

    def _remember_interaction(self, task: str, answer: str, decision: Decision) -> Memory:
        summary = f"任务：{task}\n选用动作：{decision.action}\n回答：{answer[:400]}"
        return self.store.write(
            summary,
            kind="episodic",
            title=task[:60],
            source="agent",
            source_ref=f"run:{self.tracer.run_id}" if self.tracer else "",
            confidence=0.4,
            importance=0.2,
            tags=(decision.action, "interaction"),
        )
