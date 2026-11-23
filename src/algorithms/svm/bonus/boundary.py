from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ....processing.dimensionality_reduction.pca import fitPCA, transformPCA
from ..model import SVM

KERNELS = (("linear", "Linear"), ("poly", "Polynomial"), ("rbf", "Gaussian (RBF)"))


def plot_boundaries(X, y, save_path, sample_size=300, random_state=0, **svm_params):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if len(np.unique(y)) != 2:
        raise ValueError("Visualisasi SVM membutuhkan tepat dua kelas.")
    rng = np.random.default_rng(random_state)
    per_class = max(1, sample_size // 2)
    idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per_class] for c in np.unique(y)])
    X, y = X[idx], y[idx]

    if X.shape[1] > 2:
        pca = fitPCA(X, 2)
        X, axis_labels = transformPCA(X, pca), [f"PC{i + 1} ({r:.1%})" for i, r in enumerate(pca["explainedRatio"])]
    else:
        axis_labels = ["x1", "x2"]
    std = X.std(0)
    X = (X - X.mean(0)) / np.where(std > 0, std, 1)

    pad = 0.5
    (x0, y0), (x1, y1) = X.min(0) - pad, X.max(0) + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    grid = np.c_[gx.ravel(), gy.ravel()]

    svm_params = {k: v for k, v in svm_params.items() if k != "kernel"}
    figure = Figure(figsize=(18, 6))
    for ax, (kernel, title) in zip(figure.subplots(1, 3), KERNELS):
        model = SVM(kernel=kernel, **svm_params).fit(X, y)
        z = model.decision_function(grid).reshape(gx.shape)
        limit = max(np.abs(z).max(), 1e-9)
        ax.contourf(gx, gy, z, levels=20, cmap="RdBu", vmin=-limit, vmax=limit, alpha=0.35)
        ax.contour(gx, gy, z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
        for label, color in zip(model.classes_, ("tab:red", "tab:blue")):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], s=14, c=color, label=f"kelas {label}")
        ax.scatter(*model.support_vectors_.T, s=70, facecolors="none", edgecolors="k", linewidths=1,
                   label=f"support vector ({len(model.support_vectors_)})")
        ax.set(title=f"SVM {title}", xlabel=axis_labels[0], ylabel=axis_labels[1], xlim=(x0, x1), ylim=(y0, y1))
        ax.legend(loc="upper right", fontsize=8)
    figure.suptitle("Bidang pemisah (garis tegas), margin (putus-putus), dan support vector (lingkaran)")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=120, bbox_inches="tight")
    return save_path
