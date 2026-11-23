from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from .selection import mrmr, topK


def featureSelection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    k: int,
    *,
    continuousColumns: Sequence[Hashable] | None = None,
    binCount: int = 10,
) -> tuple[pd.DataFrame | np.ndarray, list[Hashable]]:
    columns = topK(mrmr(X, y, continuousColumns=continuousColumns, binCount=binCount), k)
    if isinstance(X, pd.DataFrame):
        return X.loc[:, columns].copy(), columns
    return np.asarray(X)[:, columns].copy(), columns
