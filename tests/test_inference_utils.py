"""Tests for inference.utils (no GPU / vLLM required)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from inference.utils import (
    convert_prompt_to_messages,
    find_duplicate_caseids,
    normalize_caseid,
    read_completed_caseids,
    resolve_hf_token,
)


def test_normalize_caseid_strips_float_suffix():
    assert normalize_caseid(12.0) == "12"
    assert normalize_caseid("12.0") == "12"
    assert normalize_caseid("p1__demos") == "p1__demos"
    assert normalize_caseid(None) == ""
    assert normalize_caseid(float("nan")) == ""


def test_find_duplicate_caseids_preserves_order():
    series = pd.Series(["a", "b", "a", "c", "b"])
    assert find_duplicate_caseids(series) == ["a", "b"]


def test_convert_prompt_to_messages_with_system():
    msgs = convert_prompt_to_messages("Hello", system_msg="System")
    assert msgs == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
    ]


def test_convert_prompt_to_messages_without_system():
    msgs = convert_prompt_to_messages("Hello")
    assert msgs == [{"role": "user", "content": "Hello"}]


def test_read_completed_caseids_ok(tmp_path: Path):
    path = tmp_path / "results.csv"
    pd.DataFrame(
        {"caseid": ["p1__demos", "p2__demos"], "generated_text": ["{}", "{}"]}
    ).to_csv(path, index=False)
    assert read_completed_caseids(path) == {"p1__demos", "p2__demos"}


def test_read_completed_caseids_rejects_duplicates(tmp_path: Path):
    path = tmp_path / "results.csv"
    pd.DataFrame(
        {"caseid": ["p1__demos", "p1__demos"], "generated_text": ["a", "b"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate caseid"):
        read_completed_caseids(path)


def test_read_completed_caseids_rejects_missing_column(tmp_path: Path):
    path = tmp_path / "results.csv"
    pd.DataFrame({"id": ["1"], "generated_text": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="caseid"):
        read_completed_caseids(path)


def test_resolve_hf_token_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_test_token\n", encoding="utf-8")
    assert resolve_hf_token(str(token_path)) == "hf_test_token"


def test_resolve_hf_token_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    missing = tmp_path / "missing.txt"
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert resolve_hf_token(str(missing)) == "from_env"
