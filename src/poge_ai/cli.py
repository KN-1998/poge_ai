# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line interface for the poge_ai baseline.

Examples
--------
    poge-ai ask "3 * (4 + 5) 等于多少"
    poge-ai remember "用户偏好可复现的实验步骤" --kind semantic --source user
    poge-ai recall "实验偏好"
    poge-ai stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .agent import Agent
from .memory import KINDS, MemoryStore
from .planner import PlannerError, build_planner
from .router import RuleRouter
from .tools import ToolPolicy, ToolRegistry
from .trace import TraceRecorder

DEFAULT_DB = "poge_ai.db"
DEFAULT_TRACE = "runs/trace.jsonl"


def _build_store(args: argparse.Namespace) -> MemoryStore:
    return MemoryStore(args.db)


def _build_registry(args: argparse.Namespace) -> ToolRegistry:
    policy = ToolPolicy(allow_write=getattr(args, "allow_write_tools", False))
    return ToolRegistry(args.root or ".", policy=policy)


def _emit(args: argparse.Namespace, payload: dict, human: str) -> int:
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(human)
    return 0


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        planner = build_planner(args.planner)
    except PlannerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    store = _build_store(args)
    try:
        tracer = TraceRecorder(args.trace)
        agent = Agent(
            store=store,
            tools=_build_registry(args),
            planner=planner,
            router=RuleRouter(),
            tracer=tracer,
            memory_limit=args.limit,
            write_memory=not args.no_memory,
        )
        result = agent.run(args.task)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()

    decision = result.decision
    human_lines = [
        f"动作：{decision.action}（置信度 {decision.confidence:.2f}）",
        f"理由：{decision.reason}",
    ]
    if result.tool_result is not None:
        human_lines.append(
            f"工具：{result.tool_result.tool} -> {result.tool_result.value}"
            if result.tool_result.ok
            else f"工具：{result.tool_result.tool} 失败：{result.tool_result.error}"
        )
    human_lines.extend(["", "回答：", result.answer, "", f"轨迹：{result.trace_path} (run_id={result.run_id})"])
    return _emit(args, result.to_dict(), "\n".join(human_lines))


def cmd_remember(args: argparse.Namespace) -> int:
    store = _build_store(args)
    try:
        memory = store.write(
            args.content,
            kind=args.kind,
            title=args.title,
            source=args.source,
            source_ref=args.source_ref,
            confidence=args.confidence,
            importance=args.importance,
            tags=args.tag,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    return _emit(args, memory.to_dict(), f"已写入 {memory.id}（{memory.kind}）")


def cmd_recall(args: argparse.Namespace) -> int:
    store = _build_store(args)
    try:
        hits = store.search(args.query, kind=args.kind, limit=args.limit)
    finally:
        store.close()

    if not hits:
        return _emit(args, {"query": args.query, "hits": []}, "没有命中任何记忆。")

    human = [f"命中 {len(hits)} 条（查询：{args.query}）："]
    for index, hit in enumerate(hits, start=1):
        human.append(
            f"{index}. [{hit.memory.kind}] score={hit.score:.3f} id={hit.memory.id}\n"
            f"   {hit.memory.content}"
        )
    return _emit(
        args,
        {"query": args.query, "hits": [hit.to_dict() for hit in hits]},
        "\n".join(human),
    )


def cmd_forget(args: argparse.Namespace) -> int:
    store = _build_store(args)
    try:
        memory = store.forget(args.memory_id, mode=args.mode, reason=args.reason)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    return _emit(
        args,
        memory.to_dict(),
        f"已将 {memory.id} 标记为 {memory.status}（理由：{memory.note}）",
    )


def cmd_stats(args: argparse.Namespace) -> int:
    store = _build_store(args)
    try:
        stats = store.stats()
    finally:
        store.close()
    human = "\n".join(
        [
            f"数据库：{stats['path']}",
            f"总数：{stats['total']}（active={stats['active']}, "
            f"archived={stats['archived']}, revoked={stats['revoked']}）",
            "分类：" + ", ".join(f"{kind}={count}" for kind, count in stats["by_kind"].items()),
        ]
    )
    return _emit(args, stats, human)


def cmd_tools(args: argparse.Namespace) -> int:
    registry = _build_registry(args)
    specs = registry.describe()
    human_lines = [f"已注册 {len(specs)} 个工具："]
    for spec in specs:
        params = ", ".join(
            f"{p['name']}:{p['type']}" + ("" if p["required"] else "?") for p in spec["params"]
        )
        human_lines.append(f"- {spec['name']}({params}) [{spec['side_effect']}] {spec['description']}")
    return _emit(args, {"tools": specs, "policy": vars(registry.policy)}, "\n".join(human_lines))


def cmd_trace(args: argparse.Namespace) -> int:
    events = TraceRecorder.read(args.path or args.trace, limit=args.limit)
    if not events:
        return _emit(args, {"path": args.path or args.trace, "events": []}, "轨迹为空。")
    human = [json.dumps(event, ensure_ascii=False) for event in events]
    return _emit(args, {"path": args.path or args.trace, "events": events}, "\n".join(human))


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poge-ai",
        description="poge_ai research prototype: inspectable memory / routing / tool baseline.",
    )
    parser.add_argument("--version", action="version", version=f"poge_ai {__version__}")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path (default: {DEFAULT_DB})")
    parser.add_argument(
        "--trace", default=DEFAULT_TRACE, help=f"JSONL trace path (default: {DEFAULT_TRACE})"
    )
    parser.add_argument("--root", default=".", help="tool sandbox root (default: .)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="run one task through the agent loop")
    ask.add_argument("task")
    ask.add_argument("--planner", default="offline", choices=["offline", "openai"])
    ask.add_argument("--limit", type=int, default=5, help="max memories to recall")
    ask.add_argument("--no-memory", action="store_true", help="do not write the interaction to memory")
    ask.add_argument("--allow-write-tools", action="store_true", help="allow side-effecting tools")
    ask.set_defaults(func=cmd_ask)

    remember = sub.add_parser("remember", help="write one memory")
    remember.add_argument("content")
    remember.add_argument("--kind", default="episodic", choices=list(KINDS))
    remember.add_argument("--title", default="")
    remember.add_argument("--source", default="user", help="where this memory came from")
    remember.add_argument("--source-ref", default="", help="source-specific reference")
    remember.add_argument("--confidence", type=float, default=0.7)
    remember.add_argument("--importance", type=float, default=0.5)
    remember.add_argument("--tag", action="append", default=[])
    remember.set_defaults(func=cmd_remember)

    recall = sub.add_parser("recall", help="search memory")
    recall.add_argument("query")
    recall.add_argument("--kind", default=None, choices=list(KINDS))
    recall.add_argument("--limit", type=int, default=5)
    recall.set_defaults(func=cmd_recall)

    forget = sub.add_parser("forget", help="archive or revoke a memory")
    forget.add_argument("memory_id")
    forget.add_argument("--mode", default="archive", choices=["archive", "revoke"])
    forget.add_argument("--reason", required=True, help="why this memory should no longer be recalled")
    forget.set_defaults(func=cmd_forget)

    stats = sub.add_parser("stats", help="show memory statistics")
    stats.set_defaults(func=cmd_stats)

    tools = sub.add_parser("tools", help="list registered tools")
    tools.add_argument("--allow-write-tools", action="store_true")
    tools.set_defaults(func=cmd_tools)

    trace = sub.add_parser("trace", help="print a trace file")
    trace.add_argument("--path", default=None, help="trace file (defaults to --trace)")
    trace.add_argument("--limit", type=int, default=20)
    trace.set_defaults(func=cmd_trace)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
