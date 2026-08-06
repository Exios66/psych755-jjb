#!/usr/bin/env python3
"""Quarto ``project.pre-render`` hook: refresh the PR/issue progress graph.

Runs ``scripts/sync_pr_graph_data.py`` so the Progress page always reflects the
latest GitHub PRs and issues on every ``quarto render`` / ``quarto preview``.
Degrades gracefully: if the ``gh`` CLI is missing or not authenticated (offline
sandboxes, CI without tokens), it warns and exits 0 so the render never fails on
stale graph data.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts" / "sync_pr_graph_data.py"


def main() -> int:
    if shutil.which("gh") is None:
        print(
            "pre-render: gh CLI not found; skipping PR graph sync (Progress page keeps last baked data)",
            file=sys.stderr,
        )
        return 0

    try:
        auth = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        print(f"pre-render: could not probe gh auth: {exc}; skipping PR graph sync", file=sys.stderr)
        return 0
    if auth.returncode != 0:
        print(
            "pre-render: gh not authenticated; skipping PR graph sync (Progress page keeps last baked data)",
            file=sys.stderr,
        )
        return 0

    try:
        subprocess.run([sys.executable, str(SYNC)], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"pre-render: PR graph sync failed ({exc}); continuing render with existing data",
            file=sys.stderr,
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
