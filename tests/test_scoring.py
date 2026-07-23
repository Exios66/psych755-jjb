import pandas as pd

from ca_personas.scoring import add_ground_truth_scores, likert_to_int, subscale_score


def test_likert_mapping():
    assert likert_to_int("Strongly disagree") == 1
    assert likert_to_int("Somewhat agree") == 4
    assert likert_to_int("") is None


def test_subscale_all_neutral_is_eighteen():
    # All 3s: reverse items also become 3 → 6*3 = 18
    row = pd.Series(
        {
            "Q1": "Neither agree nor disagree",
            "Q2": "Neither agree nor disagree",
            "Q3": "Neither agree nor disagree",
            "Q4": "Neither agree nor disagree",
            "Q5": "Neither agree nor disagree",
            "Q6": "Neither agree nor disagree",
        }
    )
    assert subscale_score(row, ("Q1", "Q3", "Q5"), ("Q2", "Q4", "Q6")) == 18


def test_low_apprehension_group_score():
    # Comfort high / anxiety low → low CA
    row = pd.Series(
        {
            "Q1": "Strongly disagree",  # 1
            "Q2": "Strongly agree",  # reverse 1
            "Q3": "Strongly disagree",  # 1
            "Q4": "Strongly agree",  # reverse 1
            "Q5": "Strongly disagree",  # 1
            "Q6": "Strongly agree",  # reverse 1
            "Q13": "Strongly disagree",
            "Q14": "Strongly agree",
            "Q15": "Strongly disagree",
            "Q16": "Strongly agree",
            "Q17": "Strongly agree",
            "Q18": "Strongly disagree",
        }
    )
    scored = add_ground_truth_scores(pd.DataFrame([row]))
    assert int(scored.loc[0, "gt_group_ca"]) == 6
    assert int(scored.loc[0, "gt_interpersonal_ca"]) == 6
    assert scored.loc[0, "gt_group_band"] == "low"
