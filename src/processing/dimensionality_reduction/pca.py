from numbers import Integral, Real
import numpy as np

def fitPCA(X, nComponents):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("PCA butuh matriks dua dimensi dengan minimal dua baris.")
    if not np.isfinite(X).all():
        raise ValueError("PCA butuh nilai numerik terhingga.")
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    variance = S**2
    total = variance.sum()
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    if isinstance(nComponents, bool):
        raise TypeError("nComponents harus int atau float.")
    if isinstance(nComponents, Integral):
        if not 1 <= nComponents <= len(S):
            raise ValueError(f"nComponents harus di antara 1 dan {len(S)}.")
        k = int(nComponents)
    elif isinstance(nComponents, Real):
        if not 0 < nComponents < 1:
            raise ValueError("nComponents berupa float harus di antara 0 dan 1.")
        k = min(int(np.searchsorted(np.cumsum(ratio), nComponents) + 1), len(S))
    else:
        raise TypeError("nComponents harus int atau float.")
    return {"mean": mean, "components": Vt[:k], "explainedRatio": ratio[:k]}


def transformPCA(X, pca):
    return (np.asarray(X, dtype=float) - pca["mean"]) @ pca["components"].T
