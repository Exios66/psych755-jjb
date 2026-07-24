"""
vLLM batch inference for digital-twin prompt CSVs.

Reads a CSV with ``caseid`` and ``prompt`` columns, applies a chat template via
the model's tokenizer, runs generation through vLLM's offline ``LLM`` engine,
and writes results to a CSV that supports checkpoint-resume.

CLI usage::

    python -m inference.predict_vllm \
        --prompt_csv prompts.csv --result_csv results.csv

See ``python -m inference.predict_vllm --help`` for all flags.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ca_personas.personas import SYSTEM_PROMPT
from inference.utils import (
    convert_prompt_to_messages,
    find_duplicate_caseids,
    normalize_caseid,
    read_completed_caseids,
    resolve_hf_token,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("HF_HOME", str(_REPO_ROOT / "hf_cache"))

# Launcher structure adapted from ai_terrarium_v2; system text is this repo's
# CA digital-twin instruction (kept in sync via ca_personas.personas.SYSTEM_PROMPT).
DEFAULT_SYSTEM_MSG = SYSTEM_PROMPT.strip()

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _device_to_cuda_id(device: str | int) -> int:
    if isinstance(device, int):
        return device
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return 0


def _apply_cuda_visible_devices(device: str, tensor_parallel_size: int) -> None:
    """Select physical GPUs before vLLM / CUDA initialises.

    When TP > 1 but the environment only exposes one GPU, override to the
    needed consecutive span.  If the env already lists enough GPUs, leave it
    unchanged.
    """
    start = _device_to_cuda_id(device)

    def _count_visible(cvd: str) -> int:
        cvd = (cvd or "").strip()
        if not cvd:
            return 0
        return len([p for p in cvd.split(",") if p.strip()])

    if tensor_parallel_size <= 1:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(start)
        return

    needed = ",".join(str(start + i) for i in range(tensor_parallel_size))
    cur = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if _count_visible(cur) >= tensor_parallel_size:
        return
    if cur:
        print(
            f"[predict_vllm] CUDA_VISIBLE_DEVICES was {cur!r}; need {tensor_parallel_size} GPUs for TP. "
            f"Setting to {needed!r}. If CUDA already initialized in this process, restart the kernel and run again."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = needed


# ---------------------------------------------------------------------------
# Prompt preparation
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: list[dict], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _build_prompts(df: pd.DataFrame, system_msg: str, tokenizer) -> list[str]:
    out: list[str] = []
    for _, row in df.iterrows():
        msgs = convert_prompt_to_messages(row["prompt"], system_msg=system_msg)
        out.append(_messages_to_prompt(msgs, tokenizer))
    return out


def _attach_ground_truth(df: pd.DataFrame, ground_truth_csv: str | None) -> pd.DataFrame:
    """Attach answer labels generated beside prompt batches, when provided."""
    if not ground_truth_csv:
        return df

    truth_path = Path(ground_truth_csv)
    if not truth_path.is_file():
        raise SystemExit(f"Ground-truth CSV not found: {truth_path}")

    truth_df = pd.read_csv(truth_path)
    truth_df = truth_df.copy()
    truth_df["caseid"] = truth_df["caseid"].map(normalize_caseid)
    required = {"caseid", "answer"}
    missing = required - set(truth_df.columns)
    if missing:
        raise SystemExit(f"Ground-truth CSV must have columns {sorted(required)}; missing {sorted(missing)}")

    dupes = find_duplicate_caseids(truth_df["caseid"])
    if dupes:
        sample = ", ".join(dupes[:5])
        suffix = " ..." if len(dupes) > 5 else ""
        raise SystemExit(
            "Ground-truth CSV contains duplicate caseid rows "
            f"({len(dupes)} caseids affected; examples: {sample}{suffix}). "
            "Remove or consolidate duplicates before merging."
        )

    if "answer" in df.columns:
        return df

    truth_cols = ["caseid", "answer"]
    if "raw_answer" in truth_df.columns:
        truth_cols.append("raw_answer")
    out = df.merge(truth_df[truth_cols], on="caseid", how="left")
    if out["answer"].isna().any():
        missing_count = int(out["answer"].isna().sum())
        raise SystemExit(f"Ground-truth CSV missing answers for {missing_count} prompt rows.")
    return out


# ---------------------------------------------------------------------------
# HF token resolution
# ---------------------------------------------------------------------------


def _log_model_load_path(model_name: str, token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=model_name, token=token, local_files_only=True)
        print(f"[vLLM] Loading weights from local cache: {local}")
    except Exception:
        cache_dir = (
            os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or "~/.cache/huggingface"
        )
        print(f"[vLLM] Model not in local cache or cache not set; will load from Hub / cache dir: {cache_dir}")


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def vllm_predict(
    df: pd.DataFrame,
    device: str,
    model_cfg,
    result_csv: str,
    *,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
) -> None:
    """Run chat-templated vLLM generation and append results to *result_csv*.

    Parameters
    ----------
    df : DataFrame
        Must contain ``caseid`` and ``prompt`` columns.  Optionally ``answer``
        (ground truth) which is carried through to the output.
    device : str
        E.g. ``"cuda:0"``; combined with *tensor_parallel_size* to set
        ``CUDA_VISIBLE_DEVICES``.
    model_cfg : namespace
        Attributes consumed: ``model_full_name`` or ``model_name``,
        ``system_msg``, ``max_output_tokens``, ``temperature``, ``top_p``,
        ``repetition_penalty``, ``batch_size``, ``save_freq``,
        ``hf_access_token_file``, ``load_mode``, ``quantization``,
        ``tensor_parallel_size``, ``gpu_memory_utilization``, ``max_model_len``.
    result_csv : str
        Output path.  Rows whose ``caseid`` already exists will be skipped
        (checkpoint-resume).
    """
    df = df.copy()
    df["caseid"] = df["caseid"].map(normalize_caseid)

    tp = getattr(model_cfg, "tensor_parallel_size", tensor_parallel_size)
    _apply_cuda_visible_devices(device, tp)
    print(
        f"[predict_vllm] tensor_parallel_size={tp}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )

    # Late imports so CUDA_VISIBLE_DEVICES is set before the CUDA runtime loads,
    # and so `python -m inference.predict_vllm --help` works without [vllm] deps.
    from tqdm import tqdm  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    # ---- checkpoint resume ------------------------------------------------
    completed_caseids: set[str] = set()
    if os.path.exists(result_csv):
        try:
            completed_caseids = read_completed_caseids(result_csv)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid existing result CSV: {exc}\n"
                "Delete or repair the file before resuming so new generations "
                "are not appended to a corrupt checkpoint."
            ) from exc
        save_freq = getattr(model_cfg, "save_freq", 100)
        print(f"Found existing results: {len(completed_caseids)} samples completed")
        print(f"Resuming from chunk {len(completed_caseids) // save_freq + 1}")

    if completed_caseids:
        before = len(df)
        df = df[~df["caseid"].isin(completed_caseids)].reset_index(drop=True)
        print(f"Filtered out {before - len(df)} completed samples")
        print(f"Remaining samples to process: {len(df)}")
        if len(df) == 0:
            print("All samples already completed!")
            return

    # ---- model setup ------------------------------------------------------
    model_name = (
        getattr(model_cfg, "model_full_name", None)
        or getattr(model_cfg, "model_name", None)
    )
    if not model_name:
        raise ValueError("model_cfg must set model_full_name or model_name")

    token = resolve_hf_token(getattr(model_cfg, "hf_access_token_file", None))
    if token:
        os.environ.setdefault("HF_TOKEN", token)

    _log_model_load_path(model_name, token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    quantization = getattr(model_cfg, "quantization", None)
    if getattr(model_cfg, "load_mode", None) in ("4bit", "8bit"):
        quantization = "bitsandbytes"

    llm_kwargs: dict = {
        "model": model_name,
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": getattr(
            model_cfg, "gpu_memory_utilization", gpu_memory_utilization,
        ),
    }
    if quantization:
        llm_kwargs["quantization"] = quantization
    resolved_len = (
        max_model_len if max_model_len is not None
        else getattr(model_cfg, "max_model_len", None)
    )
    if resolved_len is not None:
        llm_kwargs["max_model_len"] = resolved_len

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=model_cfg.max_output_tokens,
        temperature=getattr(model_cfg, "temperature", 0.0),
        top_p=getattr(model_cfg, "top_p", 1.0),
        repetition_penalty=getattr(model_cfg, "repetition_penalty", 1.0),
    )

    # ---- generation -------------------------------------------------------
    batch_size = getattr(model_cfg, "batch_size", 16)
    save_freq = getattr(model_cfg, "save_freq", 200)
    system_msg = getattr(model_cfg, "system_msg", DEFAULT_SYSTEM_MSG)

    total_samples = len(df)
    t0 = time.perf_counter()
    dataset = df.reset_index(drop=True)
    chunks = [dataset.iloc[i : i + save_freq] for i in range(0, len(dataset), save_freq)]

    for i, chunk_df in tqdm(enumerate(chunks), total=len(chunks), desc="vLLM chunks"):
        prompts = _build_prompts(chunk_df, system_msg, tokenizer)
        all_texts: list[str] = []
        for j in range(0, len(prompts), batch_size):
            batch_prompts = prompts[j : j + batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for out in outputs:
                all_texts.append(out.outputs[0].text if out.outputs else "")

        out_df = chunk_df[["caseid"]].copy()
        if "answer" in chunk_df.columns:
            out_df.insert(1, "answer", chunk_df["answer"].values)
        out_df["generated_text"] = all_texts
        out_df["generated_text"] = (
            out_df["generated_text"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )

        write_header = not os.path.exists(result_csv)
        out_df.to_csv(
            result_csv,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        print(f"Saved chunk {i + 1}/{len(chunks)} to {result_csv}")

    elapsed = time.perf_counter() - t0
    rate = total_samples / elapsed if elapsed > 0 else 0.0
    print(f"[vLLM] Total: {elapsed:.1f}s, {total_samples} samples, {rate:.2f} samples/s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _model_cfg_from_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_full_name=args.model,
        system_msg=DEFAULT_SYSTEM_MSG,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        load_mode=args.load_mode,
        quantization=args.quantization,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_access_token_file=args.hf_access_token_file,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="vLLM batch inference: prompt CSV (caseid, prompt) -> result CSV.",
    )
    ap.add_argument("--prompt_csv", required=True,
                    help="Input CSV with columns: caseid, prompt.")
    ap.add_argument("--result_csv", required=True,
                    help="Output CSV path (append + resume supported).")
    ap.add_argument("--ground_truth_csv", default="",
                    help="Optional CSV with caseid, answer columns to merge into results.")
    ap.add_argument("--gpu", type=int, default=0,
                    help="First physical GPU id; with --tensor_parallel_size N "
                         "uses GPUs id..id+N-1.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="HuggingFace model id or local path.")
    ap.add_argument(
        "--hf_access_token_file",
        type=str,
        default="hf_access_token.txt",
        help=(
            "HF token file for gated models (also checked relative to repo root). "
            "If missing or empty, falls back to HF_TOKEN / HUGGINGFACE_HUB_TOKEN / "
            "HUGGING_FACE_HUB_TOKEN like predict_transformers."
        ),
    )
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Sub-batch size for llm.generate calls.")
    ap.add_argument("--save_freq", type=int, default=200,
                    help="Flush results to CSV every N rows.")
    ap.add_argument("--max_output_tokens", type=int, default=256,
                    help="Maximum new tokens per sample (CA JSON needs headroom).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0 = greedy).")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="Nucleus sampling (1.0 = disabled).")
    ap.add_argument("--repetition_penalty", type=float, default=1.0,
                    help=">1 penalises repeated tokens (e.g. 1.1 with 4-bit + greedy).")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="vLLM GPU memory fraction.")
    ap.add_argument("--max_model_len", type=int, default=8192,
                    help="Maximum model context length.")
    ap.add_argument("--tensor_parallel_size", type=int, default=2,
                    help="Tensor-parallel degree (number of GPUs).")
    ap.add_argument("--load_mode", type=str, default="none",
                    choices=("4bit", "8bit", "none"),
                    help="Legacy shorthand: 4bit/8bit -> bitsandbytes quantisation.")
    ap.add_argument("--quantization", type=str, default="fp8",
                    choices=("fp8", "bitsandbytes", "awq", "gptq", "none"),
                    help="vLLM quantisation method. 'fp8' recommended for Ada/Hopper.")

    args = ap.parse_args(argv)
    if args.load_mode == "none":
        args.load_mode = ""
    if args.quantization == "none":
        args.quantization = None

    df = pd.read_csv(args.prompt_csv)
    if "caseid" not in df.columns or "prompt" not in df.columns:
        raise SystemExit("Input CSV must have columns: caseid, prompt")
    df["caseid"] = df["caseid"].map(normalize_caseid)
    df = _attach_ground_truth(df, args.ground_truth_csv)

    model_cfg = _model_cfg_from_args(args)
    vllm_predict(df, f"cuda:{args.gpu}", model_cfg, args.result_csv)


if __name__ == "__main__":
    main()
