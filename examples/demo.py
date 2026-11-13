# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scripted demo of one full baseline run.

Run it from a checkout:

    python examples/demo.py

It prints, for three tasks, the routing decision, whether a tool was used, and the
trace events that were written. The answer text comes from ``OfflinePlanner`` and
is a placeholder -- the demo shows *control flow*, not intelligence.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from poge_ai.agent import Agent  # noqa: E402
from poge_ai.memory import MemoryStore  # noqa: E402
from poge_ai.planner import OfflinePlanner  # noqa: E402
from poge_ai.tools import ToolRegistry  # noqa: E402
from poge_ai.trace import TraceRecorder  # noqa: E402

TASKS = [
    "3 * (4 + 5) 等于多少",
    "还记得我上次说的偏好吗",
    "请把这段需求拆成三步并说明理由",
]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = MemoryStore(root / "demo.db")
        tracer = TraceRecorder(root / "trace.jsonl")
        registry = ToolRegistry(root)

        store.write(
            "用户偏好中文回答，并要求给出可复现的实验步骤",
            kind="semantic",
            source="user",
            confidence=0.9,
            importance=0.8,
        )

        try:
            for index, task in enumerate(TASKS, start=1):
                # a fresh recorder per task keeps one run_id per task
                tracer = TraceRecorder(root / "trace.jsonl")
                agent = Agent(
                    store=store,
                    tools=registry,
                    planner=OfflinePlanner(),
                    tracer=tracer,
                )
                result = agent.run(task)
                print(f"--- 任务 {index}: {task}")
                print(f"    动作: {result.decision.action} ({result.decision.confidence:.2f})")
                print(f"    理由: {result.decision.reason}")
                print(f"    用到的记忆: {result.memories_used or '无'}")
                print(f"    run_id: {result.run_id}")
                if result.tool_result is not None:
                    status = "ok" if result.tool_result.ok else f"failed: {result.tool_result.error}"
                    print(f"    工具: {result.tool_result.tool} -> {status}")
                print()

                print("  轨迹：")
                for event in tracer.events():
                    print(f"    step={event['step']:<2} {event['action']}")
                print()
        finally:
            store.close()

    print("\n完成。默认 Planner 不产生真实答案，只用于观察控制流。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
