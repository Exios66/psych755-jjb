"""Tests for Posit secondary_results.json builder and path hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ca_personas.pipeline import _resolve_prolific_paths
from ca_personas.posit_secondary import build_secondary_results, write_secondary_results


def _synthetic_secondary_cohort(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    q26_reg = ["4-8 days a month", "8 or more days a month"]
    q26_other = ["Never", "0-1 days a month", "2-4 days a month"]
    q27 = [
        "1-2 rides in a typical day",
        "3-4 rides in a typical day",
        "5-6 rides in a typical day",
    ]
    q28 = [
        "Never",
        "0-1 days a month",
        "2-4 days a month",
        "4-8 days a month",
        "8 or more days a month",
    ]
    for i in range(n):
        regular = i % 5 != 0
        rows.append(
            {
                "participant_id": f"p{i}",
                "Q26": q26_reg[i % 2] if regular else q26_other[i % 3],
                "Q27": q27[i % 3],
                "Q28": q28[i % 5],
                "Q29": q27[i % 3],
                "Q20": "Yes" if i % 3 else "No",
                "Q21": "Yes" if i % 4 else "No",
                "Employment status": ["Full-Time", "Part-Time", "Other"][i % 3],
                "Country of residence": ["United States", "United Kingdom"][i % 2],
                "LocationLatitude": float(30 + rng.normal(0, 5)),
                "LocationLongitude": float(-90 + rng.normal(0, 10)),
                "gt_group_ca": float(rng.normal(12 if regular else 16, 3)),
                "gt_interpersonal_ca": float(rng.normal(12 if regular else 15, 3)),
                "gt_group_band": "low",
                "gt_interpersonal_band": "moderate",
            }
        )
    return pd.DataFrame(rows)


def test_build_secondary_results_shape():
    df = _synthetic_secondary_cohort()
    payload = build_secondary_results(
        df, random_state=0, n_boot=40, n_perm_repeats=2
    )
    assert payload["n_analytic"] == len(df)
    assert payload["n_regular"] + payload["n_not_regular"] == payload["n_analytic"]
    assert "geo_rf" in payload and "roc_auc" in payload["geo_rf"]
    assert "ca_rf" in payload and "confusion" in payload["ca_rf"]
    assert {"tn", "fp", "fn", "tp"} <= set(payload["ca_rf"]["confusion"])
    assert "group" in payload["transit_ca"] and "interpersonal" in payload["transit_ca"]
    keys = {r["spec_key"] for r in payload["covariate_comparison"]}
    assert {"q28_days", "geo_benchmark", "ca_benchmark", "chance"} <= keys
    # Live geo/CA benchmarks should be wired into comparison rows (rounded).
    geo_row = next(r for r in payload["covariate_comparison"] if r["spec_key"] == "geo_benchmark")
    ca_row = next(r for r in payload["covariate_comparison"] if r["spec_key"] == "ca_benchmark")
    assert geo_row["roc_auc"] == pytest.approx(round(payload["geo_rf"]["roc_auc"], 3))
    assert ca_row["roc_auc"] == pytest.approx(round(payload["ca_rf"]["roc_auc"], 3))
    assert payload["q27_prevalence"]
    assert payload["q28_prevalence"]


def test_write_secondary_results(tmp_path: Path):
    path = write_secondary_results(
        _synthetic_secondary_cohort(n=60),
        tmp_path / "secondary_results.json",
        random_state=1,
        n_boot=30,
        n_perm_repeats=2,
    )
    assert path.is_file()
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_analytic"] == 60


def test_partial_config_prolific_files_raises(tmp_path: Path):
    """Wave A alone must not silently yield a truncated cohort."""
    a = tmp_path / "PRCAProlificExport_FileA.csv"
    a.write_text("Participant id\np1\n", encoding="utf-8")
    b = tmp_path / "PRCAProlificExport_FileB.csv"  # missing on purpose
    config = {
        "paths": {
            "prolific_files": [str(a), str(b)],
            "qualtrics": str(tmp_path / "missing_c.csv"),
        }
    }
    with pytest.raises(FileNotFoundError, match="Incomplete config prolific_files"):
        _resolve_prolific_paths(None, config)
