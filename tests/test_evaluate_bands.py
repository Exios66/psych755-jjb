import pandas as pd

from ca_personas.evaluate import evaluate_predictions, summarize_errors


def test_exact_and_band_accuracy_metrics():
    participants = pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "Age": 30,
                "Sex": "Female",
                "gt_group_ca": 12,
                "gt_interpersonal_ca": 18,
                "gt_group_band": "low",
                "gt_interpersonal_band": "moderate",
            },
            {
                "participant_id": "p2",
                "Age": 40,
                "Sex": "Male",
                "gt_group_ca": 22,
                "gt_interpersonal_ca": 10,
                "gt_group_band": "high",
                "gt_interpersonal_band": "low",
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "tier": "demos",
                "pred_group_ca": 12,
                "pred_interpersonal_ca": 20,
                "pred_group_band": "low",
                "pred_interpersonal_band": "high",
            },
            {
                "participant_id": "p2",
                "tier": "demos",
                "pred_group_ca": 21,
                "pred_interpersonal_ca": 10,
                "pred_group_band": "high",
                "pred_interpersonal_band": "low",
            },
        ]
    )

    evaluated = evaluate_predictions(participants, predictions)
    assert bool(evaluated.loc[0, "exact_match_group"]) is True
    assert bool(evaluated.loc[0, "exact_match_interpersonal"]) is False
    assert bool(evaluated.loc[0, "band_match_group"]) is True
    assert bool(evaluated.loc[0, "band_match_interpersonal"]) is False
    assert bool(evaluated.loc[1, "band_match_group"]) is True

    summary = summarize_errors(evaluated)
    all_row = summary[summary["tier"] == "all"].iloc[0]
    assert all_row["exact_acc_group"] == 0.5
    assert all_row["band_acc_group"] == 1.0
    assert all_row["band_acc_interpersonal"] == 0.5
