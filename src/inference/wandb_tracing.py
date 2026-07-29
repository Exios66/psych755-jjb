"""Weights & Biases logging for vLLM digital-twin runs.

Mirrors the Braintrust chunk summaries: opt-in when ``WANDB_API_KEY`` (or
``WANDB_ENABLED=true``) is set, never blocks GPU inference on W&B failures.

Env vars
--------
WANDB_API_KEY       required to upload (else no-op)
WANDB_PROJECT       default ``psych755-ca-personas``
WANDB_ENTITY        optional team/user
WANDB_RUN_NAME      optional run name (else model_preset_timestamp)
WANDB_ENABLED       force on/off (true|false)
WANDB_MODE          online|offline|disabled (default online when keyed)
WANDB_TAGS          comma-separated extra tags
WANDB_DIR           local wandb directory (default ``./wandb`` under cwd)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from inference.braintrust_tracing import score_ca_generation
from inference.ca_prompts import parse_caseid
from inference.utils import normalize_caseid

DEFAULT_PROJECT = "psych755-ca-personas"
ENV_API_KEY = "WANDB_API_KEY"
ENV_PROJECT = "WANDB_PROJECT"
ENV_ENTITY = "WANDB_ENTITY"
ENV_RUN_NAME = "WANDB_RUN_NAME"
ENV_ENABLED = "WANDB_ENABLED"
ENV_MODE = "WANDB_MODE"
ENV_TAGS = "WANDB_TAGS"


def wandb_configured(*, enabled: bool | None = None) -> bool:
    """Return True when W&B logging should run."""
    if enabled is False:
        return False
    if enabled is True:
        return True
    flag = os.getenv(ENV_ENABLED, "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    mode = os.getenv(ENV_MODE, "").strip().lower()
    if mode == "disabled":
        return False
    return bool(os.getenv(ENV_API_KEY, "").strip()) or mode == "offline"


def project_name() -> str:
    return (os.getenv(ENV_PROJECT) or DEFAULT_PROJECT).strip() or DEFAULT_PROJECT


@dataclass
class WandbRun:
    """Handle for an open W&B run (or a disabled no-op)."""

    enabled: bool
    project: str
    run_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _run: Any = field(default=None, repr=False)
    n_logged: int = 0
    n_chunks: int = 0

    def log_chunk(
        self,
        *,
        chunk_idx: int,
        n_chunks: int,
        rows: list[Mapping[str, Any]],
        model: str | None = None,
        preset: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a chunk and ``wandb.log`` aggregate metrics (+ optional table)."""
        n = 0
        parse_ok = 0
        mae_vals: list[float] = []
        exact_vals: list[float] = []
        band_vals: list[float] = []
        by_tier: dict[str, list[float]] = {}

        for row in rows:
            scored = score_ca_generation(
                str(row.get("generated_text", "")),
                row.get("answer"),
            )
            n += 1
            scores = scored.get("scores") or {}
            metrics = scored.get("metrics") or {}
            if scores.get("parse_ok", 0.0) >= 1.0:
                parse_ok += 1
            mae = metrics.get("mae_mean")
            if mae is not None:
                mae_vals.append(float(mae))
            if "exact_match_mean" in scores:
                exact_vals.append(float(scores["exact_match_mean"]))
            if "band_match_mean" in scores:
                band_vals.append(float(scores["band_match_mean"]))
            cid = normalize_caseid(str(row.get("caseid", "")))
            try:
                _, tier = parse_caseid(cid)
            except ValueError:
                tier = None
            if tier and mae is not None:
                by_tier.setdefault(str(tier), []).append(float(mae))

        summary = {
            "n": n,
            "parse_rate": (parse_ok / n) if n else None,
            "mae_mean": (sum(mae_vals) / len(mae_vals)) if mae_vals else None,
            "exact_match_mean": (sum(exact_vals) / len(exact_vals)) if exact_vals else None,
            "band_match_mean": (sum(band_vals) / len(band_vals)) if band_vals else None,
        }
        if not self.enabled or self._run is None:
            return summary

        payload: dict[str, Any] = {
            "chunk": float(chunk_idx + 1),
            "chunk_frac": float((chunk_idx + 1) / max(n_chunks, 1)),
            "chunk/n": float(n),
            "chunk/parse_rate": float(summary["parse_rate"] or 0.0),
        }
        if summary["mae_mean"] is not None:
            payload["chunk/mae_mean"] = float(summary["mae_mean"])
        if summary["exact_match_mean"] is not None:
            payload["chunk/exact_match_mean"] = float(summary["exact_match_mean"])
        if summary["band_match_mean"] is not None:
            payload["chunk/band_match_mean"] = float(summary["band_match_mean"])
        for tier, vals in by_tier.items():
            if vals:
                payload[f"chunk/mae_{tier}"] = float(sum(vals) / len(vals))
        if model:
            payload["meta/model"] = model
        if preset:
            payload["meta/preset"] = preset
        if extra:
            for k, v in extra.items():
                if isinstance(v, (int, float)) and v is not None:
                    payload[f"extra/{k}"] = float(v)

        try:
            import wandb

            wandb.log(payload, step=chunk_idx + 1)
            self.n_chunks += 1
            self.n_logged += n
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] log_chunk failed: {exc}")
            self.enabled = False
        return summary

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            import wandb

            flat = {
                f"run/{k}": float(v)
                for k, v in summary.items()
                if isinstance(v, (int, float)) and v is not None
            }
            if flat:
                wandb.log(flat)
            wandb.summary.update(
                {k: v for k, v in summary.items() if v is not None}
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] log_summary failed: {exc}")

    def close(self) -> str | None:
        if not self.enabled or self._run is None:
            return None
        url = None
        try:
            import wandb

            url = getattr(self._run, "url", None) or wandb.run.url if wandb.run else None
            wandb.finish()
            if url:
                print(f"[wandb] run: {url}")
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] finish failed: {exc}")
        return url


def start_wandb_run(
    *,
    model: str,
    preset: str,
    enabled: bool | None = None,
    project: str | None = None,
    run_name: str | None = None,
    system_prompt_meta: Mapping[str, Any] | None = None,
    extra_config: Mapping[str, Any] | None = None,
) -> WandbRun:
    """Open a W&B run for one vLLM batch, or a disabled stub."""
    proj = (project or project_name()).strip()
    if not wandb_configured(enabled=enabled):
        return WandbRun(enabled=False, project=proj)

    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed; logging disabled "
              "(pip install -e '.[wandb]' or '.[vllm]')")
        return WandbRun(enabled=False, project=proj)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_short = model.rsplit("/", 1)[-1]
    name = (
        run_name
        or os.getenv(ENV_RUN_NAME)
        or f"vllm_{model_short}_{preset}_{ts}"
    )
    config: dict[str, Any] = {
        "provider": "vllm",
        "model": model,
        "preset": preset,
        "repo": "psych755-ca-personas",
        "task": "ca-digital-twin",
    }
    if system_prompt_meta:
        config["system_prompt"] = dict(system_prompt_meta)
    if extra_config:
        config.update(dict(extra_config))

    tags = ["vllm", "ca-digital-twin", preset]
    extra_tags = os.getenv(ENV_TAGS, "").strip()
    if extra_tags:
        tags.extend(t.strip() for t in extra_tags.split(",") if t.strip())

    init_kwargs: dict[str, Any] = {
        "project": proj,
        "name": name,
        "config": config,
        "tags": tags,
        "job_type": "vllm-infer",
        "reinit": True,
    }
    entity = (os.getenv(ENV_ENTITY) or "").strip()
    if entity:
        init_kwargs["entity"] = entity
    mode = (os.getenv(ENV_MODE) or "").strip()
    if mode:
        init_kwargs["mode"] = mode

    try:
        run = wandb.init(**init_kwargs)
        url = getattr(run, "url", None)
        print(f"[wandb] run={name!r} project={proj!r}" + (f" url={url}" if url else ""))
        return WandbRun(
            enabled=True,
            project=proj,
            run_name=name,
            metadata=config,
            _run=run,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] init failed ({exc}); continuing without W&B logging")
        return WandbRun(enabled=False, project=proj)
