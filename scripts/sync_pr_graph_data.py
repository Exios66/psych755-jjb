#!/usr/bin/env python3
"""Bake GitHub PR graph data for the Quarto Progress page.

Fetches all pull requests for Exios66/psych755-jjb via ``gh``, classifies each
into a theme group, emits timeline + relates-to edges, and writes offline JSON
under ``artifacts/pr_graph/`` so Posit Connect renders without live GitHub auth.

Usage:

    python scripts/sync_pr_graph_data.py
    python scripts/sync_pr_graph_data.py --repo Exios66/psych755-jjb
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "pr_graph"
DEFAULT_REPO = "Exios66/psych755-jjb"

# Deterministic theme rules: first match wins (order matters).
# Prefer specific domains (vllm, secondary-rq, ml) before broad "persona/prompt".
THEME_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "vllm",
        re.compile(r"vllm|llama|deepseek|inference.?launcher|cross.?model", re.I),
    ),
    (
        "secondary-rq",
        re.compile(
            r"transit|geo(?:location)?|secondary|q27|q28|rideshare|car.?access|"
            r"employment.?predict|demographics.?predict|country.?predict|"
            r"mobility|follow.?up.?experiment|wave.?2|comprehensive.?feature|"
            r"comprehensive.?predictor|kitchen.?sink|research.?memo",
            re.I,
        ),
    ),
    (
        "ml",
        re.compile(
            r"\bml\b|baseline|random.?forest|\brf\b|knn|ridge|xgboost|\bmlp\b|"
            r"shap|factor.?analys|feature.?import|feature.?predict",
            re.I,
        ),
    ),
    (
        "personas",
        re.compile(
            r"persona|prompt|terrarium|digital.?twin|prca|stereotyp|"
            r"system_prompt|tier|band.?metric|distance.?from.?correct|"
            r"ground.?truth|gt.?scor",
            re.I,
        ),
    ),
    (
        "site",
        re.compile(
            r"quarto|posit|manuscript|landing|site.?nav|site.?audit|"
            r"apa.?site|abstract|github.?access|nav(?:igation)?",
            re.I,
        ),
    ),
    (
        "data",
        re.compile(
            r"file.?[abc]|sibling|cohort|merge.?coverage|cleaning|excerpt|"
            r"pipeline|load(?:ing)?|artifact",
            re.I,
        ),
    ),
    (
        "infra",
        re.compile(
            r"agents\.md|pr.?template|issue.?template|dev.?environment|"
            r"setup|gitignore|ci|workflow|changelog|harden",
            re.I,
        ),
    ),
    (
        "docs",
        re.compile(r"readme|getstarted|docs?|memo|documentation|contributions", re.I),
    ),
]

THEME_LABELS = {
    "personas": "Personas & prompts",
    "secondary-rq": "Secondary RQs",
    "ml": "ML baselines",
    "vllm": "vLLM / models",
    "site": "Quarto / Posit site",
    "data": "Data & pipeline",
    "infra": "Infra & tooling",
    "docs": "Docs",
}

# Cap relates-to degree so ~40–50 nodes stay readable.
MAX_RELATES_DEGREE = 4


def _run_gh_json(args: list[str]) -> Any:
    cmd = ["gh", *args]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("gh CLI not found; install GitHub CLI to sync PR data.") from exc
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        raise SystemExit(f"gh failed ({exc.returncode}): {err}") from exc
    return json.loads(proc.stdout)


def fetch_prs(repo: str) -> list[dict[str, Any]]:
    """Fetch all PRs (open + closed) with fields needed for the graph."""
    raw = _run_gh_json(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            "200",
            "--json",
            "number,title,state,mergedAt,createdAt,closedAt,author,url,labels,isDraft,body",
        ]
    )
    if not isinstance(raw, list):
        raise SystemExit(f"Unexpected gh pr list payload: {type(raw)}")
    return raw


def classify_theme(title: str, body: str | None = None) -> str:
    """Classify from title first; fall back to body only if title is uninformative."""
    for theme, pattern in THEME_RULES:
        if pattern.search(title or ""):
            return theme
    if body:
        for theme, pattern in THEME_RULES:
            if pattern.search(body):
                return theme
    return "docs"


def normalize_status(pr: dict[str, Any]) -> str:
    """Map GitHub PR fields onto graph legend statuses."""
    if pr.get("mergedAt"):
        return "merged"
    state = str(pr.get("state") or "").upper()
    if state == "OPEN":
        return "draft" if pr.get("isDraft") else "open"
    if state == "CLOSED":
        return "closed"
    # MERGED sometimes appears as state from gh
    if state == "MERGED":
        return "merged"
    return "closed"


def author_login(pr: dict[str, Any]) -> str:
    author = pr.get("author") or {}
    if isinstance(author, dict):
        return str(author.get("login") or "unknown")
    return str(author)


def body_excerpt(body: str | None, limit: int = 280) -> str:
    if not body:
        return ""
    cleaned = re.sub(r"\s+", " ", body).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def sort_key_iso(pr_node: dict[str, Any]) -> str:
    return (
        pr_node.get("mergedAt")
        or pr_node.get("closedAt")
        or pr_node.get("createdAt")
        or ""
    )


def build_nodes(raw_prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for pr in raw_prs:
        title = str(pr.get("title") or "")
        body = pr.get("body")
        theme = classify_theme(title, body if isinstance(body, str) else None)
        labels = []
        for lab in pr.get("labels") or []:
            if isinstance(lab, dict):
                name = lab.get("name")
                if name:
                    labels.append(str(name))
            else:
                labels.append(str(lab))
        number = int(pr["number"])
        nodes.append(
            {
                "id": f"PR-{number}",
                "number": number,
                "title": title,
                "status": normalize_status(pr),
                "theme": theme,
                "themeLabel": THEME_LABELS.get(theme, theme),
                "author": author_login(pr),
                "url": str(pr.get("url") or f"https://github.com/{DEFAULT_REPO}/pull/{number}"),
                "createdAt": pr.get("createdAt"),
                "mergedAt": pr.get("mergedAt"),
                "closedAt": pr.get("closedAt"),
                "isDraft": bool(pr.get("isDraft")),
                "labels": labels,
                "excerpt": body_excerpt(body if isinstance(body, str) else None),
            }
        )
    nodes.sort(key=lambda n: n["number"])
    return nodes


def build_timeline_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chain consecutive nodes by merge/close/create time (progress spine)."""
    ordered = sorted(nodes, key=lambda n: (sort_key_iso(n), n["number"]))
    edges: list[dict[str, Any]] = []
    for a, b in zip(ordered, ordered[1:]):
        edges.append(
            {
                "id": f"timeline-{a['number']}-{b['number']}",
                "source": a["id"],
                "target": b["id"],
                "type": "timeline",
            }
        )
    return edges


def build_relates_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Connect same-theme PRs with a capped degree for readability."""
    by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_theme[node["theme"]].append(node)

    edges: list[dict[str, Any]] = []
    degree: dict[str, int] = defaultdict(int)

    for theme, group in by_theme.items():
        group_sorted = sorted(group, key=lambda n: n["number"])
        # Path through theme in number order, then limited skip links.
        for a, b in zip(group_sorted, group_sorted[1:]):
            if degree[a["id"]] >= MAX_RELATES_DEGREE or degree[b["id"]] >= MAX_RELATES_DEGREE:
                continue
            edges.append(
                {
                    "id": f"relates-{a['number']}-{b['number']}",
                    "source": a["id"],
                    "target": b["id"],
                    "type": "relates",
                    "theme": theme,
                }
            )
            degree[a["id"]] += 1
            degree[b["id"]] += 1

        # Extra relates for every other neighbor when degree allows (clusters).
        for i, a in enumerate(group_sorted):
            if degree[a["id"]] >= MAX_RELATES_DEGREE:
                continue
            for b in group_sorted[i + 2 : i + 4]:
                if degree[a["id"]] >= MAX_RELATES_DEGREE or degree[b["id"]] >= MAX_RELATES_DEGREE:
                    break
                edge_id = f"relates-{a['number']}-{b['number']}"
                if any(e["id"] == edge_id for e in edges):
                    continue
                edges.append(
                    {
                        "id": edge_id,
                        "source": a["id"],
                        "target": b["id"],
                        "type": "relates",
                        "theme": theme,
                    }
                )
                degree[a["id"]] += 1
                degree[b["id"]] += 1

    return edges


def build_payload(repo: str, raw_prs: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = build_nodes(raw_prs)
    timeline = build_timeline_edges(nodes)
    relates = build_relates_edges(nodes)
    status_counts: dict[str, int] = defaultdict(int)
    theme_counts: dict[str, int] = defaultdict(int)
    for n in nodes:
        status_counts[n["status"]] += 1
        theme_counts[n["theme"]] += 1

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "meta": {
            "repo": repo,
            "generatedAt": now,
            "nodeCount": len(nodes),
            "edgeCount": len(timeline) + len(relates),
            "statusCounts": dict(status_counts),
            "themeCounts": dict(theme_counts),
            "themes": [
                {"id": tid, "label": THEME_LABELS[tid]}
                for tid in THEME_LABELS
                if tid in theme_counts
            ],
        },
        "nodes": nodes,
        "edges": timeline + relates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/name repository")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="output directory for prs.json / meta.json",
    )
    args = parser.parse_args(argv)

    raw = fetch_prs(args.repo)
    if not raw:
        raise SystemExit("No pull requests returned from GitHub.")

    payload = build_payload(args.repo, raw)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prs_path = out_dir / "prs.json"
    meta_path = out_dir / "meta.json"
    payload_text = json.dumps(payload, indent=2) + "\n"
    prs_path.write_text(payload_text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(payload["meta"], indent=2) + "\n", encoding="utf-8"
    )

    # Co-locate a copy next to the Quarto page assets for reliable relative fetch.
    page_data = ROOT / "docs" / "pr_graph" / "prs.json"
    page_data.parent.mkdir(parents=True, exist_ok=True)
    page_data.write_text(payload_text, encoding="utf-8")

    print(
        f"Wrote {prs_path.relative_to(ROOT)} and {page_data.relative_to(ROOT)} "
        f"({payload['meta']['nodeCount']} nodes, "
        f"{payload['meta']['edgeCount']} edges)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
