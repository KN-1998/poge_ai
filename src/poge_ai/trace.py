# SPDX-License-Identifier: AGPL-3.0-or-later
"""Append-only JSONL trace of what an agent run actually did.

The point of the trace is to make claims checkable: which action was chosen, with
which arguments, how long it took, whether it failed, and which memories were
used. Traces are also the raw material for the process-supervision experiments
proposed in ``docs/architecture.md`` §4.

Anything whose key looks like a credential is replaced before writing, and long
strings are truncated. This is a defence against accidental leakage, not a
guarantee: callers must still avoid putting secrets in tool arguments.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|apikey|access[_-]?token|token|secret|password|passwd|"
    r"authorization|auth[_-]?header|cookie|credential|private[_-]?key)",
    re.IGNORECASE,
)
MAX_STRING = 240
MAX_DEPTH = 8


def redact(value: Any, *, depth: int = 0) -> Any:
    """Return a trace-safe copy of ``value``."""

    if depth > MAX_DEPTH:
        return "<max-depth>"
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            cleaned[name] = "***" if SENSITIVE_KEY_RE.search(name) else redact(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [redact(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        text = value.replace("\r", " ")
        return text if len(text) <= MAX_STRING else text[: MAX_STRING - 1] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(repr(value), depth=depth + 1)


class TraceRecorder:
    """Collects run events and optionally appends them to a JSONL file."""

    def __init__(self, path: str | Path | None = None, *, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []

    def record(self, *, step: int, action: str, **fields: Any) -> dict:
        event: dict[str, Any] = {
            "run_id": self.run_id,
            "step": int(step),
            "action": str(action),
        }
        # redact the whole mapping so that sensitive *keys* are masked too
        for key, value in redact(dict(fields)).items():
            event[str(key)] = value

        self._events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def events(self) -> list[dict]:
        """Events recorded by this recorder instance, in order."""

        return list(self._events)

    @staticmethod
    def read(path: str | Path, *, limit: int | None = None) -> list[dict]:
        """Read a trace file back. Malformed lines are skipped, not fatal."""

        events: list[dict] = []
        file_path = Path(path)
        if not file_path.exists():
            return events
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if limit is not None:
            events = events[-max(1, int(limit)) :]
        return events

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None
