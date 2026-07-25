from ca_personas.apa_plotting import apply_apa_style, grouped_bars, horizontal_bars
import matplotlib.pyplot as plt


def test_grouped_bars_smoke():
    apply_apa_style()
    fig, ax = plt.subplots()
    grouped_bars(ax, ["a", "b"], {"x": [1.0, 2.0], "y": [1.5, 2.5]}, ylabel="y")
    assert ax.get_ylabel() == "y"
    plt.close(fig)


def test_horizontal_bars_smoke():
    fig, ax = plt.subplots()
    horizontal_bars(ax, ["a", "b"], [0.5, 0.7], xlabel="auc", highlight_index=1)
    assert ax.get_xlabel() == "auc"
    plt.close(fig)
