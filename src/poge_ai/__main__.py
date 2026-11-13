# SPDX-License-Identifier: AGPL-3.0-or-later
"""``python -m poge_ai`` entry point."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
