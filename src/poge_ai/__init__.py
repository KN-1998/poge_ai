# SPDX-License-Identifier: AGPL-3.0-or-later
"""poge_ai — research prototype for an environment-native, memory-managed agent.

This package implements *only* the MVP-1 systems baseline described in
``docs/research_agenda.md``: an inspectable memory store, a rule-based action
router, a permission-scoped tool registry, a JSONL trace, and an agent loop that
ties them together.

It does **not** implement the neural memory slots, learned gating, or joint
training proposed in ``docs/architecture.md``. Default answer quality is
intentionally meaningless; see ``docs/decisions/0003-baseline-prototype.md``.
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
