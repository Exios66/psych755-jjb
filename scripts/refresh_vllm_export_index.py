#!/usr/bin/env python3
"""Rebuild exports/INDEX.md from on-disk packages + the live v2/v3 queue plan."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
QUEUE_STATUS = ROOT / "logging" / "queue_llama_deepseek_v2_v3.status"
QUEUE_STATUS_TF = ROOT / "logging" / "queue_transit_focus.status"

# Planned enhanced CA runs (tag → display label).
PLANNED: dict[str, tuple[str, str]] = {
    "llama31_8b_instruct_v2": ("v2", "Llama-3.1-8B-Instruct"),
    "llama32_3b_v2": ("v2", "Llama-3.2-3B (base)"),
    "llama32_3b_instruct_v2": ("v2", "Llama-3.2-3B-Instruct"),
    "deepseek_r1_distill_llama8b_v2": ("v2", "DeepSeek-R1-Distill-Llama-8B"),
    "llama33_70b_instruct_awq_v2": ("v2", "Llama-3.3-70B-Instruct-AWQ"),
    "llama31_8b_instruct_v3": ("v3", "Llama-3.1-8B-Instruct"),
    "llama32_3b_v3": ("v3", "Llama-3.2-3B (base)"),
    "llama32_3b_instruct_v3": ("v3", "Llama-3.2-3B-Instruct"),
    "deepseek_r1_distill_llama8b_v3": ("v3", "DeepSeek-R1-Distill-Llama-8B"),
    "llama33_70b_instruct_awq_v3": ("v3", "Llama-3.3-70B-Instruct-AWQ"),
}

# Transit-focus TF1/TF2 LLM twin queue (separate bucket).
PLANNED_TRANSIT_FOCUS: dict[str, tuple[str, str]] = {
    "llama31_8b_instruct_transit_focus": ("transit_focus", "Llama-3.1-8B-Instruct"),
    "llama32_3b_instruct_transit_focus": ("transit_focus", "Llama-3.2-3B-Instruct"),
    "deepseek_r1_distill_llama8b_transit_focus": ("transit_focus", "DeepSeek-R1-Distill-Llama-8B"),
    "llama33_70b_instruct_awq_transit_focus": ("transit_focus", "Llama-3.3-70B-Instruct-AWQ"),
}

BUCKET_ORDER = ("v1", "v2", "v3", "transit_focus", "prior_v3_greedy")
BUCKET_BLURB = {
    "v1": "Published prompt-v1 baselines (greedy / Jul-26).",
    "v2": "Signal-first 5-tier packaging + `v2_enhanced` / `large_model` presets.",
    "v3": "8-tier ablations + `v3_enhanced` / `large_model` presets (refreshed anti-bleed).",
    "transit_focus": "TF1/TF2: predict regular transit + Q26 with mobility held out.",
    "prior_v3_greedy": "Pre-refresh greedy v3 archives (comparison only; not the new target).",
}


def _tag_from_dirname(name: str) -> str | None:
    # psych755_vllm_<tag>_full_cohort_<stamp>
    m = re.match(r"psych755_vllm_(.+)_full_cohort_\d{8}_\d{4}$", name)
    return m.group(1) if m else None


def _scan_bucket(bucket: str) -> list[dict]:
    root = EXPORTS / bucket
    if not root.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith("psych755_vllm_"):
            continue
        tag = _tag_from_dirname(path.name) or path.name
        meta_path = path / "00_run_metadata.json"
        meta: dict = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
        report = path / "REPORT_results_interpretation_and_model_success.md"
        rows.append(
            {
                "bucket": bucket,
                "tag": tag,
                "dir": str(path.relative_to(ROOT)),
                "model": meta.get("model") or meta.get("model_tag") or "?",
                "parse_rate": meta.get("parse_rate"),
                "n_tiers": len(meta.get("tiers") or []),
                "n_rows": meta.get("n_prompt_rows") or meta.get("n_result_rows"),
                "has_report": report.is_file(),
                "stamp": meta.get("export_stamp") or "",
            }
        )
    return rows


def _queue_progress() -> tuple[set[str], str | None]:
    """Return tags marked PACKAGE done across queues, plus current START tag."""
    done: set[str] = set()
    current: str | None = None
    for path in (QUEUE_STATUS, QUEUE_STATUS_TF):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if "PACKAGE done tag=" in line:
                done.add(line.rsplit("tag=", 1)[-1].strip())
            if "START job=" in line:
                frag = line.split("START job=", 1)[-1]
                current = frag.split()[0].strip()
    return done, current


def _status_row(tag: str, bucket: str, label: str, by_bucket: dict, queue_done: set[str], current: str | None) -> str:
    on_disk = any(r["tag"] == tag for r in by_bucket.get(bucket, []))
    packaged = tag in queue_done or on_disk
    if packaged:
        status = "✅ done"
    elif current == tag:
        status = "🔄 running"
    else:
        status = "⏳ pending"
    return f"| {status} | `{bucket}` | `{tag}` | {label} |"


def main() -> int:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    by_bucket = {b: _scan_bucket(b) for b in BUCKET_ORDER}
    queue_done, current = _queue_progress()

    lines: list[str] = []
    lines.append("# vLLM export index")
    lines.append("")
    lines.append(f"_Regenerated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("Layout:")
    lines.append("")
    lines.append("```")
    lines.append("exports/")
    lines.append("  INDEX.md              ← this file")
    lines.append("  v1/                   ← prompt-v1 CA baselines (done)")
    lines.append("  v2/                   ← enhanced 5-tier CA (filling)")
    lines.append("  v3/                   ← enhanced 8-tier CA (queued)")
    lines.append("  transit_focus/        ← TF1/TF2 transit-prediction LLM twins")
    lines.append("  prior_v3_greedy/      ← archived pre-refresh v3")
    lines.append("  zips/")
    lines.append("```")
    lines.append("")
    lines.append("CA packages: `scripts/package_vllm_export.py` → `exports/<v1|v2|v3>/`.")
    lines.append("Transit-focus packages: `scripts/package_transit_focus_export.py` → `exports/transit_focus/`.")
    lines.append("")

    lines.append("## Enhanced CA queue matrix")
    lines.append("")
    lines.append("| Status | Bucket | Tag | Model |")
    lines.append("|---|---|---|---|")
    for tag, (bucket, label) in PLANNED.items():
        lines.append(_status_row(tag, bucket, label, by_bucket, queue_done, current))
    lines.append("")

    lines.append("## Transit-focus queue matrix (TF1/TF2)")
    lines.append("")
    lines.append("Predict `regular_transit` + Q26 from demos/employment/geo/(+CA) with mobility held out.")
    lines.append("")
    lines.append("| Status | Bucket | Tag | Model |")
    lines.append("|---|---|---|---|")
    for tag, (bucket, label) in PLANNED_TRANSIT_FOCUS.items():
        lines.append(_status_row(tag, bucket, label, by_bucket, queue_done, current))
    lines.append("")
    if current:
        lines.append(f"**Queue currently on:** `{current}`")
        lines.append("")

    for bucket in BUCKET_ORDER:
        rows = by_bucket[bucket]
        lines.append(f"## `{bucket}/` — {BUCKET_BLURB[bucket]}")
        lines.append("")
        if not rows:
            lines.append("_Empty — new packages will appear here._")
            lines.append("")
            continue
        lines.append("| Package dir | Model | Tiers | Parse | Report |")
        lines.append("|---|---|---:|---:|---|")
        for r in rows:
            parse = "—" if r["parse_rate"] is None else f"{100 * float(r['parse_rate']):.1f}%"
            tiers = r["n_tiers"] or "—"
            report = "yes" if r["has_report"] else "no"
            lines.append(
                f"| [`{Path(r['dir']).name}`]({Path(r['dir']).relative_to('exports')}/) "
                f"| `{r['model']}` | {tiers} | {parse} | {report} |"
            )
        lines.append("")

    planned_all = {**PLANNED, **PLANNED_TRANSIT_FOCUS}
    missing = [
        tag
        for tag, (bucket, _) in planned_all.items()
        if not any(r["tag"] == tag for r in by_bucket.get(bucket, []))
    ]
    lines.append("## Not on disk yet (will appear as queues finish)")
    lines.append("")
    if not missing:
        lines.append("_All planned packages are present._")
    else:
        for tag in missing:
            bucket, label = planned_all[tag]
            mark = "← running" if tag == current else ""
            lines.append(f"- `{bucket}/psych755_vllm_{tag}_full_cohort_<stamp>/` — {label} {mark}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Refresh this file anytime:")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/refresh_vllm_export_index.py")
    lines.append("```")
    lines.append("")

    out = EXPORTS / "INDEX.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out.relative_to(ROOT)} ({len(missing)} pending packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
