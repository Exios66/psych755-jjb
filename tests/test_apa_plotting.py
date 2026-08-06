from ca_personas import apa_plotting
from ca_personas.apa_plotting import (
    apply_apa_style,
    grouped_bars,
    horizontal_bars,
    prevalence_bars,
    roc_curve_apa,
    scatter_by_group,
)
import matplotlib.pyplot as plt


def test_apply_apa_style_uses_seaborn_theme(monkeypatch):
    calls = []
    monkeypatch.setattr(apa_plotting.sns, "set_theme", lambda **kw: calls.append(kw))
    apply_apa_style()
    assert calls and calls[-1].get("style") == "white"


def test_grouped_bars_smoke():
    apply_apa_style()
    fig, ax = plt.subplots()
    grouped_bars(ax, ["a", "b"], {"x": [1.0, 2.0], "y": [1.5, 2.5]}, ylabel="y")
    assert ax.get_ylabel() == "y"
    assert len(ax.patches) == 4
    assert ax.patches[0].get_hatch() == ""
    assert ax.patches[1].get_hatch() == "///"
    plt.close(fig)


def test_horizontal_bars_smoke():
    fig, ax = plt.subplots()
    horizontal_bars(ax, ["a", "b"], [0.5, 0.7], xlabel="auc", highlight_index=1)
    assert ax.get_xlabel() == "auc"
    plt.close(fig)


def test_roc_and_prevalence_helpers():
    fig, axes = plt.subplots(1, 2)
    roc_curve_apa(axes[0], [0, 0.2, 1], [0, 0.5, 1], auc=0.75, label="RF")
    prevalence_bars(axes[1], ["Never", "Often"], [0.2, 0.8], [10, 12], sample_prevalence=0.4)
    assert axes[0].get_xlabel() == "False positive rate"
    assert axes[1].get_xlabel().startswith("Proportion")
    plt.close(fig)


def test_scatter_by_group():
    fig, ax = plt.subplots()
    scatter_by_group(ax, [1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 1, 1], xlabel="x", ylabel="y")
    assert ax.get_xlabel() == "x"
    assert len(ax.get_legend().get_texts()) == 2
    plt.close(fig)
