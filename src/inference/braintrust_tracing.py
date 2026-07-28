"""Braintrust tracing / scoring helpers for CA digital-twin vLLM runs.

Opt-in: when ``BRAINTRUST_API_KEY`` is unset (or ``--no-braintrust``), all
helpers are no-ops so offline CI and GPU hosts without Braintrust stay clean.

Typical flow
------------
1. Optionally load the system prompt from the Braintrust prompt registry
   (``BRAINTRUST_PROMPT_SLUG``) so playground edits can be A/B'd without a
   code deploy.
2. Open an experiment for the vLLM run (model × preset × timestamp).
3. After each generation (or via ``braintrust_log_results``), log
   input / output / expected plus PRCA scores (parse, exact, band, accuracy).

Prompt iteration: push ``prompts/braintrust_ca_system.py`` with
``bt functions push``, edit in the Braintrust playground, then re-run vLLM
with the same slug (or pin ``BRAINTRUST_PROMPT_VERSION``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ca_personas.evaluate import derive_band_from_score
from ca_personas.llm.base import extract_json_object, validate_prediction
from ca_personas.scoring import (
    PRCA_SCORE_RANGE,
    band_distance,
    normalized_band_distance,
    normalized_score_distance,
)
from inference.ca_prompts import parse_caseid
from inference.utils import normalize_caseid

DEFAULT_PROJECT = "psych755-ca-personas"
DEFAULT_PROMPT_SLUG = "ca-digital-twin-system"
ENV_API_KEY = "BRAINTRUST_API_KEY"
ENV_PROJECT = "BRAINTRUST_PROJECT"
ENV_PROMPT_SLUG = "BRAINTRUST_PROMPT_SLUG"
ENV_PROMPT_VERSION = "BRAINTRUST_PROMPT_VERSION"
ENV_PROMPT_ENVIRONMENT = "BRAINTRUST_PROMPT_ENVIRONMENT"
ENV_ENABLED = "BRAINTRUST_ENABLED"
ENV_EXPERIMENT = "BRAINTRUST_EXPERIMENT"


def braintrust_configured(*, enabled: bool | None = None) -> bool:
    """Return True when Braintrust logging should run.

    Priority: explicit *enabled* flag → ``BRAINTRUST_ENABLED`` → presence of
    ``BRAINTRUST_API_KEY``.
    """
    if enabled is False:
        return False
    if enabled is True:
        return True
    flag = os.getenv(ENV_ENABLED, "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(os.getenv(ENV_API_KEY, "").strip())


def project_name() -> str:
    return (os.getenv(ENV_PROJECT) or DEFAULT_PROJECT).strip() or DEFAULT_PROJECT


def prompt_slug() -> str:
    return (os.getenv(ENV_PROMPT_SLUG) or DEFAULT_PROMPT_SLUG).strip() or DEFAULT_PROMPT_SLUG


def _parse_answer(answer: Any) -> dict[str, Any] | None:
    if answer is None:
        return None
    if isinstance(answer, float) and str(answer) == "nan":
        return None
    if isinstance(answer, Mapping):
        return dict(answer)
    text = str(answer).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _score_side(
    *,
    pred: int | None,
    gt: int | None,
    pred_band: str | None,
    gt_band: str | None,
    side: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (scores in [0,1], raw metrics) for one PRCA subscale."""
    scores: dict[str, float] = {}
    metrics: dict[str, float] = {}
    if pred is None or gt is None:
        return scores, metrics

    abs_err = abs(int(pred) - int(gt))
    metrics[f"abs_error_{side}"] = float(abs_err)
    metrics[f"signed_error_{side}"] = float(int(pred) - int(gt))
    scores[f"exact_match_{side}"] = 1.0 if int(pred) == int(gt) else 0.0
    norm = normalized_score_distance(abs_err)
    scores[f"score_accuracy_{side}"] = float(1.0 - (norm if norm is not None else 0.0))

    resolved_band = derive_band_from_score(pred) or (
        str(pred_band).strip().lower() if pred_band else None
    )
    gt_b = str(gt_band).strip().lower() if gt_band else None
    if resolved_band and gt_b:
        dist = band_distance(resolved_band, gt_b)
        if dist is not None:
            metrics[f"band_distance_{side}"] = float(dist)
            scores[f"band_match_{side}"] = 1.0 if dist == 0 else 0.0
            scores[f"band_accuracy_{side}"] = float(
                1.0 - (normalized_band_distance(dist) or 0.0)
            )
    return scores, metrics


def score_ca_generation(
    generated_text: str | None,
    answer: Any = None,
) -> dict[str, Any]:
    """Score one vLLM generation against optional ground-truth JSON.

    Returns a dict with ``parsed``, ``error``, ``scores`` (0–1), and
    ``metrics`` (raw MAE-style quantities for Braintrust metrics panels).
    """
    raw = "" if generated_text is None else str(generated_text)
    scores: dict[str, float] = {"parse_ok": 0.0}
    metrics: dict[str, float] = {"parse_ok": 0.0}
    parsed: dict[str, Any] | None = None
    error: str | None = None

    try:
        payload = extract_json_object(raw)
        parsed = validate_prediction(payload)
        scores["parse_ok"] = 1.0
        metrics["parse_ok"] = 1.0
    except Exception as exc:  # noqa: BLE001 - capture per-row failures
        error = f"{type(exc).__name__}: {exc}"

    expected = _parse_answer(answer)
    if parsed is not None and expected is not None:
        for side, pred_key, gt_key, band_key, gt_band_key in (
            (
                "group",
                "pred_group_ca",
                "gt_group_ca",
                "pred_group_band",
                "gt_group_band",
            ),
            (
                "interpersonal",
                "pred_interpersonal_ca",
                "gt_interpersonal_ca",
                "pred_interpersonal_band",
                "gt_interpersonal_band",
            ),
        ):
            gt_val = expected.get(gt_key)
            if gt_val is None:
                continue
            try:
                gt_int = int(gt_val)
            except (TypeError, ValueError):
                continue
            side_scores, side_metrics = _score_side(
                pred=parsed.get(pred_key),
                gt=gt_int,
                pred_band=parsed.get(band_key),
                gt_band=expected.get(gt_band_key),
                side=side,
            )
            scores.update(side_scores)
            metrics.update(side_metrics)

        # Aggregate helpers for experiment dashboards / prompt comparison.
        exacts = [
            scores[k]
            for k in ("exact_match_group", "exact_match_interpersonal")
            if k in scores
        ]
        if exacts:
            scores["exact_match_mean"] = float(sum(exacts) / len(exacts))
        bands = [
            scores[k]
            for k in ("band_match_group", "band_match_interpersonal")
            if k in scores
        ]
        if bands:
            scores["band_match_mean"] = float(sum(bands) / len(bands))
        accs = [
            scores[k]
            for k in ("score_accuracy_group", "score_accuracy_interpersonal")
            if k in scores
        ]
        if accs:
            scores["score_accuracy_mean"] = float(sum(accs) / len(accs))
        abs_errs = [
            metrics[k]
            for k in ("abs_error_group", "abs_error_interpersonal")
            if k in metrics
        ]
        if abs_errs:
            metrics["mae_mean"] = float(sum(abs_errs) / len(abs_errs))
            # Higher is better inverse-MAE score for prompt ranking.
            scores["inverse_mae_mean"] = float(
                1.0 - min(1.0, metrics["mae_mean"] / float(PRCA_SCORE_RANGE))
            )

    return {
        "parsed": parsed,
        "error": error,
        "expected": expected,
        "scores": scores,
        "metrics": metrics,
    }


def resolve_system_prompt(
    *,
    fallback: str,
    use_braintrust: bool | None = None,
    project: str | None = None,
    slug: str | None = None,
    version: str | None = None,
    environment: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (system_prompt, provenance metadata).

    When Braintrust is configured and a prompt slug is available, load the
    system message from the registry. On any failure, fall back to *fallback*
    (local ``SYSTEM_PROMPT``) so GPU runs never hard-fail on registry issues.
    """
    meta: dict[str, Any] = {
        "source": "local",
        "project": project or project_name(),
        "slug": slug or prompt_slug(),
    }
    if not braintrust_configured(enabled=use_braintrust):
        return fallback, meta

    try:
        from braintrust import load_prompt
    except ImportError:
        meta["source"] = "local_braintrust_missing"
        return fallback, meta

    chosen_slug = slug or prompt_slug()
    chosen_project = project or project_name()
    chosen_version = version if version is not None else os.getenv(ENV_PROMPT_VERSION) or None
    chosen_env = (
        environment
        if environment is not None
        else os.getenv(ENV_PROMPT_ENVIRONMENT) or None
    )
    meta.update(
        {
            "slug": chosen_slug,
            "project": chosen_project,
            "version": chosen_version,
            "environment": chosen_env,
        }
    )
    try:
        prompt = load_prompt(
            project=chosen_project,
            slug=chosen_slug,
            version=chosen_version,
            environment=chosen_env,
        )
        # build() takes mustache variables as kwargs (not a positional dict).
        built = prompt.build()
        messages = built.get("messages") if isinstance(built, dict) else None
        if not messages:
            raise ValueError("Braintrust prompt.build() returned no messages")
        system_parts: list[str] = []
        for m in messages:
            role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
            content = (
                getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
            )
            if role == "system" and content:
                system_parts.append(str(content).strip())
        if not system_parts:
            raise ValueError("Braintrust prompt has no system message")
        text = "\n\n".join(system_parts).strip()
        if not text:
            raise ValueError("Braintrust system message empty")
        meta["source"] = "braintrust"
        meta["prompt_id"] = getattr(prompt, "id", None) or getattr(prompt, "name", None)
        meta["version"] = chosen_version or getattr(prompt, "version", None)
        return text, meta
    except Exception as exc:  # noqa: BLE001 - never block inference
        meta["source"] = "local_fallback"
        meta["load_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[braintrust] prompt load failed ({exc}); using local SYSTEM_PROMPT")
        return fallback, meta


@dataclass
class BraintrustRun:
    """Handle for an open Braintrust experiment (or a disabled no-op)."""

    enabled: bool
    project: str
    experiment_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_meta: dict[str, Any] = field(default_factory=dict)
    _experiment: Any = field(default=None, repr=False)
    n_logged: int = 0

    def log_generation(
        self,
        *,
        caseid: str,
        prompt: str,
        generated_text: str,
        answer: Any = None,
        system_msg: str | None = None,
        model: str | None = None,
        preset: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score and optionally log one generation. Always returns score payload."""
        scored = score_ca_generation(generated_text, answer)
        if not self.enabled or self._experiment is None:
            return scored

        cid = normalize_caseid(caseid)
        try:
            participant_id, tier = parse_caseid(cid)
        except ValueError:
            participant_id, tier = None, None

        meta: dict[str, Any] = {
            "caseid": cid,
            "participant_id": participant_id,
            "tier": tier,
            "model": model,
            "preset": preset,
            "provider": "vllm",
            "prompt_source": self.prompt_meta.get("source"),
            "prompt_slug": self.prompt_meta.get("slug"),
            "prompt_version": self.prompt_meta.get("version"),
        }
        if scored.get("error"):
            meta["parse_error"] = scored["error"]
        if extra_metadata:
            meta.update(dict(extra_metadata))

        tags = ["vllm", "ca-digital-twin"]
        if tier:
            tags.append(str(tier))
        if preset:
            tags.append(str(preset))

        output: dict[str, Any] = {
            "generated_text": generated_text,
            "parsed": scored.get("parsed"),
        }
        try:
            self._experiment.log(
                input={
                    "caseid": cid,
                    "prompt": prompt,
                    "system_prompt": system_msg,
                },
                output=output,
                expected=scored.get("expected"),
                error=scored.get("error"),
                scores=scored.get("scores") or None,
                metrics=scored.get("metrics") or None,
                metadata={k: v for k, v in meta.items() if v is not None},
                tags=tags,
                id=cid,
                allow_concurrent_with_spans=True,
            )
            self.n_logged += 1
        except Exception as exc:  # noqa: BLE001 - do not abort GPU batch
            print(f"[braintrust] log failed for {cid}: {exc}")
        return scored

    def log_batch(
        self,
        rows: list[Mapping[str, Any]],
        *,
        system_msg: str | None = None,
        model: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        """Log many generations; returns aggregate parse / MAE summary."""
        n = 0
        parse_ok = 0
        mae_vals: list[float] = []
        for row in rows:
            scored = self.log_generation(
                caseid=str(row["caseid"]),
                prompt=str(row.get("prompt", "")),
                generated_text=str(row.get("generated_text", "")),
                answer=row.get("answer"),
                system_msg=system_msg,
                model=model,
                preset=preset,
            )
            n += 1
            if scored.get("scores", {}).get("parse_ok", 0.0) >= 1.0:
                parse_ok += 1
            mae = scored.get("metrics", {}).get("mae_mean")
            if mae is not None:
                mae_vals.append(float(mae))
        summary = {
            "n": n,
            "n_logged": self.n_logged,
            "parse_rate": (parse_ok / n) if n else None,
            "mae_mean": (sum(mae_vals) / len(mae_vals)) if mae_vals else None,
        }
        return summary

    def close(self) -> dict[str, Any] | None:
        if not self.enabled or self._experiment is None:
            return None
        summary = None
        try:
            summary = self._experiment.summarize()
        except Exception as exc:  # noqa: BLE001
            print(f"[braintrust] summarize failed: {exc}")
        try:
            self._experiment.flush()
        except Exception:
            pass
        try:
            permalink = getattr(self._experiment, "permalink", None)
            if callable(permalink):
                url = permalink()
                if url:
                    print(f"[braintrust] experiment: {url}")
        except Exception:
            pass
        return summary


def start_vllm_run(
    *,
    model: str,
    preset: str,
    enabled: bool | None = None,
    project: str | None = None,
    experiment: str | None = None,
    system_prompt_meta: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> BraintrustRun:
    """Open a Braintrust experiment for one vLLM batch, or a disabled stub."""
    proj = (project or project_name()).strip()
    if not braintrust_configured(enabled=enabled):
        return BraintrustRun(enabled=False, project=proj, prompt_meta=dict(system_prompt_meta or {}))

    try:
        from braintrust import init as init_experiment
    except ImportError:
        print("[braintrust] package not installed; logging disabled "
              "(pip install -e '.[braintrust]' or '.[vllm]')")
        return BraintrustRun(enabled=False, project=proj, prompt_meta=dict(system_prompt_meta or {}))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_short = model.rsplit("/", 1)[-1]
    exp_name = (
        experiment
        or os.getenv(ENV_EXPERIMENT)
        or f"vllm_{model_short}_{preset}_{ts}"
    )
    meta: dict[str, Any] = {
        "provider": "vllm",
        "model": model,
        "preset": preset,
        "repo": "psych755-jjb",
        "task": "ca-digital-twin",
    }
    if system_prompt_meta:
        meta["system_prompt"] = dict(system_prompt_meta)
    if extra_metadata:
        meta.update(dict(extra_metadata))

    try:
        exp = init_experiment(
            project=proj,
            experiment=exp_name,
            metadata=meta,
            tags=["vllm", "ca-digital-twin", preset],
        )
        print(f"[braintrust] experiment={exp_name!r} project={proj!r}")
        return BraintrustRun(
            enabled=True,
            project=proj,
            experiment_name=exp_name,
            metadata=meta,
            prompt_meta=dict(system_prompt_meta or {}),
            _experiment=exp,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[braintrust] init failed ({exc}); continuing without logging")
        return BraintrustRun(enabled=False, project=proj, prompt_meta=dict(system_prompt_meta or {}))


def log_results_csv(
    *,
    result_csv: str | Path,
    prompt_csv: str | Path | None = None,
    model: str = "unknown",
    preset: str = "unknown",
    system_msg: str | None = None,
    enabled: bool | None = None,
    project: str | None = None,
    experiment: str | None = None,
    system_prompt_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Post-hoc: score a results CSV into a Braintrust experiment."""
    import pandas as pd

    result_path = Path(result_csv)
    df = pd.read_csv(result_path)
    if "caseid" not in df.columns or "generated_text" not in df.columns:
        raise SystemExit("Result CSV must have columns: caseid, generated_text")

    if prompt_csv and Path(prompt_csv).is_file():
        prompts = pd.read_csv(prompt_csv)
        prompts["caseid"] = prompts["caseid"].map(normalize_caseid)
        keep = ["caseid"]
        if "prompt" in prompts.columns:
            keep.append("prompt")
        if "answer" in prompts.columns and "answer" not in df.columns:
            keep.append("answer")
        df["caseid"] = df["caseid"].map(normalize_caseid)
        df = df.merge(prompts[keep].drop_duplicates("caseid"), on="caseid", how="left")
    else:
        df["caseid"] = df["caseid"].map(normalize_caseid)
        if "prompt" not in df.columns:
            df["prompt"] = ""

    run = start_vllm_run(
        model=model,
        preset=preset,
        enabled=enabled,
        project=project,
        experiment=experiment,
        system_prompt_meta=system_prompt_meta,
        extra_metadata={"source_result_csv": str(result_path)},
    )
    rows = df.to_dict(orient="records")
    summary = run.log_batch(rows, system_msg=system_msg, model=model, preset=preset)
    run.close()
    summary["experiment"] = run.experiment_name
    summary["project"] = run.project
    summary["enabled"] = run.enabled
    return summary
