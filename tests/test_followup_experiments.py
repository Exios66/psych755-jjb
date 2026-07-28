"""Tests for extended secondary follow-up experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.followup_experiments import (
    EXPERIMENT_RUNNERS,
    run_all_followup_experiments,
    run_demographics_experiment,
    run_nested_q28_car_experiment,
    save_followup_experiment_bundle,
)


def _synthetic(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, size=n).astype(float)
    sex = rng.choice(["Male", "Female"], size=n)
    student = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    country = rng.choice(["United States", "United Kingdom", "Canada"], size=n, p=[0.7, 0.2, 0.1])
    emp = rng.choice(["Full-Time", "Part-Time", "Other"], size=n, p=[0.55, 0.2, 0.25])
    q20 = rng.choice(["Yes", "No"], size=n, p=[0.8, 0.2])
    q21 = np.where(
        q20 == "Yes",
        rng.choice(["Yes", "No", "Not Sure"], size=n, p=[0.75, 0.2, 0.05]),
        rng.choice(["Yes", "No"], size=n, p=[0.2, 0.8]),
    )
    q28 = rng.choice(
        [
            "Never",
            "0-1 days a month",
            "2-4 days a month",
            "4-8 days a month",
            "8 or more days a month",
        ],
        size=n,
        p=[0.28, 0.2, 0.25, 0.17, 0.1],
    )
    q27 = rng.choice(
        [
            "1-2 rides in a typical day",
            "3-4 rides in a typical day",
            "5-6 rides in a typical day",
        ],
        size=n,
        p=[0.7, 0.25, 0.05],
    )
    logit = (
        -0.5
        + 1.2 * np.isin(q28, ["4-8 days a month", "8 or more days a month"])
        + 0.9 * (q21 == "No")
        + 0.25 * (student == "Yes")
        - 0.015 * (age - 35)
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.random(n) < p
    q26 = np.where(y, "8 or more days a month", "Never")
    group_ca = np.clip(18 - 2.5 * y + rng.normal(0, 4, n), 6, 30)
    ip_ca = np.clip(17 - 1.5 * y + rng.normal(0, 4, n), 6, 30)
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(n)],
            "Age": age,
            "Sex": sex,
            "Student status": student,
            "Country of residence": country,
            "Employment status": emp,
            "Q20": q20,
            "Q21": q21,
            "Q26": q26,
            "Q27": q27,
            "Q28": q28,
            "Q29": rng.choice(
                ["1-2 rides in a typical day", "3-4 rides in a typical day"],
                size=n,
                p=[0.85, 0.15],
            ),
            "gt_group_ca": group_ca,
            "gt_interpersonal_ca": ip_ca,
            "LocationLatitude": rng.normal(40, 5, n),
            "LocationLongitude": rng.normal(-90, 10, n),
        }
    )


def test_experiment_runners_registered():
    assert set(EXPERIMENT_RUNNERS) >= {
        "demographics",
        "country",
        "nested_q28_car",
        "ca_q28_car",
        "country_car",
        "q27_among_regular",
        "common_n",
        "residual_ca_q28",
        "mi_head_to_head",
    }


def test_demographics_recovers_signal():
    analysis = run_demographics_experiment(
        _synthetic(n=260, seed=2),
        n_splits=4,
        n_perm_repeats=4,
        random_state=2,
    )
    assert analysis["metrics"]["roc_auc"] > 0.52
    assert len(analysis["frame"]) >= 100


def test_nested_q28_car_same_n():
    analysis = run_nested_q28_car_experiment(
        _synthetic(n=220, seed=5),
        n_splits=4,
        n_perm_repeats=3,
        random_state=5,
    )
    assert analysis["summary_delta"]["n_common"] == len(analysis["frame"])
    assert analysis["summary_delta"]["q28_only_auc"] > 0.55
    assert set(analysis["analyses"]) >= {"q28_only", "q28_q21", "q21_only"}


def test_mi_head_to_head_preserves_q28_signal():
    from ca_personas.followup_experiments import run_mi_head_to_head_experiment

    df = _synthetic(n=260, seed=11)
    # Induce car / student missingness like the real cohort (~38% / ~7%).
    rng = np.random.default_rng(11)
    miss_car = rng.random(len(df)) < 0.35
    df.loc[miss_car, ["Q20", "Q21"]] = pd.NA
    miss_stu = rng.random(len(df)) < 0.08
    df.loc[miss_stu, "Student status"] = pd.NA
    analysis = run_mi_head_to_head_experiment(
        df,
        n_splits=3,
        n_imputations=3,
        random_state=11,
    )
    assert analysis["spec_key"] == "mi_head_to_head"
    assert analysis["summary_delta"]["n_full"] == len(df)
    assert "q28_days" in analysis["analyses"]["multiple_imputation"]
    assert analysis["summary_delta"]["q28_mi_auc"] > 0.55
    assert analysis["summary_delta"]["q28_leads_singletons_under_mi"] is True
    assert analysis["summary_delta"]["verdict"] in {"q28_lead_preserved", "mixed"}
    assert not analysis["comparison"].empty
    assert {"cc_roc_auc", "mi_roc_auc"} <= set(analysis["comparison"].columns)
    # Observed-only families should not jitter across imputations when CV seed is fixed.
    q28_sd = float(analysis["analyses"]["multiple_imputation"]["q28_days"]["roc_auc_sd"])
    assert q28_sd < 1e-9


def test_all_experiments_bundle(tmp_path: Path):
    bundle = run_all_followup_experiments(
        _synthetic(n=280, seed=7),
        experiment_keys=[
            "demographics",
            "country",
            "nested_q28_car",
            "common_n",
            "mi_head_to_head",
        ],
        n_splits=3,
        n_perm_repeats=2,
        n_boot=200,
        n_imputations=2,
        random_state=7,
    )
    assert set(bundle["analyses"]) >= {"demographics", "mi_head_to_head"}
    paths = save_followup_experiment_bundle(bundle, tmp_path)
    assert paths["overview"].is_file()
    assert paths["results_card"].is_file()
    assert (tmp_path / "demographics" / "demographics_results_card.json").is_file()
    assert (tmp_path / "mi_head_to_head" / "mi_head_to_head_results_card.json").is_file()
