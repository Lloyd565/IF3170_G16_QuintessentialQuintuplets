import pandas as pd
from .pca import fitPCA, transformPCA

def dimensionality_reduction(X, reducer=None, fit=False, nComponents=None):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X harus berupa DataFrame.")
    if fit:
        if nComponents is None:
            raise ValueError("Butuh nComponents untuk fit PCA.")
        reducer = fitPCA(X, nComponents)
        reducer["inputColumns"] = X.columns.tolist()
    elif reducer is None:
        raise ValueError("Butuh reducer yang sudah di-fit.")
    if X.columns.tolist() != reducer["inputColumns"]:
        raise ValueError("Kolom data harus sama dengan kolom saat PCA dilatih.")
    values = transformPCA(X, reducer)
    columns = [f"PC{i + 1}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=X.index, columns=columns), reducer
