from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def draw_graph(model, feature_names, save_path):
    if model.means is None:
        raise ValueError("Model belum dilatih, panggil fit terlebih dahulu.")
    if len(model.classes) != 2:
        raise ValueError("Graf hanya bisa dibuat untuk model dengan dua kelas.")
    if len(feature_names) != model.means.shape[1]:
        raise ValueError("Jumlah nama fitur harus sama dengan jumlah kolom yang dilatih.")

    mean_gap = np.abs(model.means[1] - model.means[0])
    average_variance = (model.variances[0] + model.variances[1]) / 2
    strengths = mean_gap / np.sqrt(average_variance)

    largest_strength = strengths.max()
    if largest_strength == 0:
        largest_strength = 1

    feature_count = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, feature_count, endpoint=False)
    used_cols = list(model.used_cols)

    figure = Figure(figsize=(16, 12))
    ax = figure.subplots()

    for i in range(feature_count):
        if i % 2 == 0:
            distance = 2
        else:
            distance = 1.5
        x = 2 * distance * np.cos(angles[i])
        y = distance * np.sin(angles[i])
        if i in used_cols:
            color = "tab:blue"
        else:
            color = "lightgray"

        line_width = 1 + 5 * strengths[i] / largest_strength
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=line_width, shrinkB=15),
        )
        ax.text(
            x, y, feature_names[i],
            ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=color),
        )

    ax.text(
        0, 0, "is_canceled",
        ha="center", va="center", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="tab:orange"),
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    ax.set_title("Naive Bayes: biru = fitur terpilih, tebal panah = beda rata-rata antar kelas dibanding sebarannya")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=120, bbox_inches="tight")
    return save_path
