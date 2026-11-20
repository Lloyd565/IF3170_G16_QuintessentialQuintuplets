import pickle
from pathlib import Path

import pandas as pd

from .data_split import ID_COL, TARGET_COL
from .missing_values import MissingValueImputer
from .validation import CAT_COLS, FEATURE_COLS, INT_COLS, learn_cats, recheck_dates, validate_data

OUTPUT_COLS = FEATURE_COLS
REPORT_COLS = ["data", "step", "rule", "col", "n_rows"]


def fix_types(data):
    data = data[OUTPUT_COLS].copy()

    for col in INT_COLS:
        data[col] = data[col].astype("int64")
    for col in CAT_COLS:
        data[col] = data[col].astype(str)
    data["nightly_rate_idr"] = data["nightly_rate_idr"].astype("float64")
    return data


def final_check(data):
    n_nan = int(data.isna().sum().sum())
    if n_nan > 0:
        raise ValueError(f"Masih ada {n_nan} NaN setelah cleaning.")
    if list(data.columns) != OUTPUT_COLS:
        raise ValueError("Urutan kolom output gak sesuai kontrak.")


class DataCleaner:
    def __init__(self):
        self.valid_cats_ = None
        self.imputer_ = None
        self.logs_ = []

    def fit(self, data):
        clean = validate_data(data)
        valid_cats = learn_cats(clean)
        imputer = MissingValueImputer().fit(clean)

        self.valid_cats_ = valid_cats
        self.imputer_ = imputer
        self.logs_ = []
        return self

    def transform(self, data, name="data"):
        if self.valid_cats_ is None or self.imputer_ is None:
            raise RuntimeError("DataCleaner belum di-fit.")

        new_logs = []
        data = validate_data(data, self.valid_cats_, new_logs)
        data = self.imputer_.transform(data, new_logs)
        
        recheck_dates(data, new_logs)
        data = fix_types(data)
        final_check(data)

        for row in new_logs:
            row["data"] = name
            self.logs_.append(row)
        return data

    def fit_transform(self, data, name="data"):
        return self.fit(data).transform(data, name=name)

    def report(self):
        return pd.DataFrame(self.logs_, columns=REPORT_COLS)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)


def load_cleaner(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def cleaning(data, cleaner=None, fit=False, name="data"):
    if ID_COL in data.columns or TARGET_COL in data.columns:
        raise ValueError(f"data gak boleh punya kolom '{ID_COL}' atau '{TARGET_COL}'; pakai split_data() dulu.")

    if fit:
        if cleaner is None:
            cleaner = DataCleaner()
        data = cleaner.fit_transform(data, name=name)
    else:
        if cleaner is None:
            raise ValueError("Butuh `cleaner` yang udah di-fit kalau fit=False.")
        data = cleaner.transform(data, name=name)
    return data, cleaner
