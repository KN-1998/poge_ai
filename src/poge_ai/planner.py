# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pluggable generation backends.

The baseline must be runnable on a machine with no GPU and no network, so the
default backend is a deterministic offline stub. **Its output is not an answer**;
it exists to make the control flow observable.

Wiring a real model is a one-method change: implement :class:`Planner` and pass it
to :class:`poge_ai.agent.Agent`. ``OpenAICompatPlanner`` is a minimal HTTP client
for Chat-Completions-compatible servers, kept dependency-free on purpose.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol, Sequence, runtime_checkable


class PlannerError(Exception):
    """Raised when a backend cannot produce a completion."""


@runtime_checkable
class Planner(Protocol):
    """Minimal contract for a text-generation backend."""

    name: str

    def complete(self, prompt: str, *, context: Sequence[str] = ()) -> str:  # pragma: no cover
        ...


class OfflinePlanner:
    """Deterministic stub: echoes the task and the context it was given.

    Useful for unit tests, demos of the trace format, and for showing exactly
    which context reached the model. It performs no reasoning and must not be
    used to evaluate answer quality.
    """

    name = "offline"

    def complete(self, prompt: str, *, context: Sequence[str] = ()) -> str:
        lines = [
            "【离线基线占位输出】",
            f"任务：{prompt}",
        ]
        if context:
            lines.append("收到的上下文：")
            lines.extend(f"  - {item}" for item in context)
        else:
            lines.append("收到的上下文：无")
        lines.append("说明：OfflinePlanner 不做推理，接入真实模型后需重新评测。")
        return "\n".join(lines)


class OpenAICompatPlanner:
    """Chat-Completions-compatible HTTP backend using only the stdlib.

    Configuration comes from arguments or the environment:

    * ``POGE_AI_BASE_URL`` – e.g. ``https://api.example.com/v1``
    * ``POGE_AI_API_KEY``  – bearer token, never logged
    * ``POGE_AI_MODEL``    – model name
    * ``POGE_AI_TIMEOUT``  – seconds, optional
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout: float = 60.0,
        system_prompt: str | None = None,
    ) -> None:
        if not model:
            raise PlannerError("model must not be empty")
        if not base_url.startswith(("http://", "https://")):
            raise PlannerError(f"base_url must be an http(s) URL, got {base_url!r}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.system_prompt = system_prompt or (
            "You are the decision core of a memory-managed agent. Answer from the "
            "provided context. If the context is insufficient, say so instead of "
            "inventing facts."
        )

    @classmethod
    def from_env(cls) -> "OpenAICompatPlanner":
        base_url = os.environ.get("POGE_AI_BASE_URL", "").strip()
        model = os.environ.get("POGE_AI_MODEL", "").strip()
        if not base_url or not model:
            raise PlannerError(
                "set POGE_AI_BASE_URL and POGE_AI_MODEL (and POGE_AI_API_KEY if required)"
            )
        timeout_text = os.environ.get("POGE_AI_TIMEOUT", "").strip()
        try:
            timeout = float(timeout_text) if timeout_text else 60.0
        except ValueError:
            timeout = 60.0
        return cls(
            model=model,
            base_url=base_url,
            api_key=os.environ.get("POGE_AI_API_KEY") or None,
            timeout=timeout,
        )

    def _payload(self, prompt: str, context: Sequence[str]) -> dict[str, Any]:
        user_content = prompt
        if context:
            user_content = "上下文：\n" + "\n".join(f"- {item}" for item in context) + f"\n\n任务：{prompt}"
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
        }

    def complete(self, prompt: str, *, context: Sequence[str] = ()) -> str:
        body = json.dumps(self._payload(prompt, context)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._api_key:
            request.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise PlannerError(f"backend returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PlannerError(f"backend call failed: {exc}") from exc

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerError("backend response did not contain choices[0].message.content") from exc


def build_planner(kind: str = "offline") -> Planner:
    """Factory used by the CLI."""

    normalized = (kind or "offline").strip().lower()
    if normalized in {"offline", "stub"}:
        return OfflinePlanner()
    if normalized in {"openai", "http"}:
        return OpenAICompatPlanner.from_env()
    raise PlannerError(f"unknown planner {kind!r}; expected 'offline' or 'openai'")
