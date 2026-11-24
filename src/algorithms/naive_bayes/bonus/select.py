import numpy as np

from ..model import NaiveBayes
from ....evaluation import evaluateMetrics

# Sumber  : https://arxiv.org/pdf/1302.6828v1
def split_data(X, y, seed):
    rng = np.random.default_rng(seed)
    train_rows = []
    check_rows = []

    for label in np.unique(y):
        rows = np.arange(len(y))[y == label]
        rng.shuffle(rows)
        check_count = round(len(rows) * 0.25)
        if check_count == 0:
            check_count = 1
        check_rows.extend(rows[:check_count])
        train_rows.extend(rows[check_count:])

    train_rows = np.array(train_rows)
    check_rows = np.array(check_rows)
    return X[train_rows], y[train_rows], X[check_rows], y[check_rows]


def macro_f1(y_true, y_pred):
    cancel_f1 = evaluateMetrics(y_pred, y_true)["f1Score"]
    stay_f1 = evaluateMetrics(1 - y_pred, 1 - y_true)["f1Score"]
    return (cancel_f1 + stay_f1) / 2


def pick_features(X, y, smoothing=1e-9, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X harus berupa matriks dua dimensi dengan minimal satu kolom.")
    if len(X) != len(y):
        raise ValueError("Jumlah baris X dan y harus sama.")

    train_X, train_y, check_X, check_y = split_data(X, y, seed)

    model = NaiveBayes(smoothing=smoothing)
    model.fit(train_X, train_y)

    picked = []
    round_scores = []
    previous_score = -1.0

    while len(picked) < X.shape[1]:
        best_col = None
        best_score = -1.0

        for col in range(X.shape[1]):
            if col in picked:
                continue
            model.used_cols = picked + [col]
            score = macro_f1(check_y, model.predict(check_X))
            if score > best_score:
                best_score = score
                best_col = col

        if best_score <= previous_score:
            break

        picked.append(best_col)
        round_scores.append(best_score)
        previous_score = best_score

    return picked, round_scores
