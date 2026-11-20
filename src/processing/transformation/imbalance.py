import numpy as np
import pandas as pd


def handle_imbalance(X_train: pd.DataFrame, y_train: pd.Series, *, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    if not X_train.index.equals(y_train.index):
        raise ValueError("Indeks fitur dan label train harus sama.")

    if not y_train.isin([0, 1]).all():
        raise ValueError("Label train harus berisi 0 atau 1.")

    counts = y_train.value_counts()
    if len(counts) != 2:
        raise ValueError("Oversampling membutuhkan kedua kelas di data train.")

    additional_rows = counts.max() - counts.min()
    if additional_rows == 0:
        return X_train.copy(), y_train.copy()

    rng = np.random.default_rng(random_state)
    minority_positions = np.flatnonzero(y_train.to_numpy() == counts.idxmin())
    sampled_positions = rng.choice(minority_positions, size=additional_rows, replace=True)
    positions = rng.permutation(np.concatenate((np.arange(len(y_train)), sampled_positions)))

    return X_train.iloc[positions].reset_index(drop=True), y_train.iloc[positions].reset_index(drop=True)
