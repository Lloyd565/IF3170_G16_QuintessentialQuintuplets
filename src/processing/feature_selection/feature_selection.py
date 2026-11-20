from collections.abc import Hashable

import numpy as np
import pandas as pd

from .selection import mrmr, topK


def featureSelection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    k: int,
) -> tuple[pd.DataFrame | np.ndarray, list[Hashable]]:
    columns = topK(mrmr(X, y), k)
    if isinstance(X, pd.DataFrame):
        return X.loc[:, columns].copy(), columns
    return np.asarray(X)[:, columns].copy(), columns
