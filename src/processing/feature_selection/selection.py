#referensi yang digunakan : https://ieeexplore.ieee.org/document/1453511

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral

import numpy as np
import pandas as pd


def mrmr(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> dict[Hashable, float]:
    if np.ndim(X) != 2:
        raise ValueError("X harus berupa matriks fitur dua dimensi.")
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if frame.empty:
        raise ValueError("X harus memiliki setidaknya satu baris dan satu fitur.")
    if not frame.columns.is_unique:
        raise ValueError("Nama kolom pada X harus unik.")

    target = np.asarray(y)
    if target.ndim != 1 or len(target) != len(frame):
        raise ValueError("y harus satu dimensi dengan satu label untuk setiap baris X.")
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        if not X.index.equals(y.index):
            raise ValueError("Indeks baris X dan y harus sama dan memiliki urutan yang sama.")

    columns = [
        validateColumn(frame.iloc[:, index])
        for index in range(frame.shape[1])
    ]
    target = validateColumn(pd.Series(target, name="target"))
    relevance = np.array([
        mutualInformation(column, target) for column in columns
    ])

    redundancy = np.zeros((len(columns), len(columns)), dtype=float)
    for left in range(len(columns)):
        for right in range(left):
            value = mutualInformation(columns[left], columns[right])
            redundancy[left, right] = redundancy[right, left] = value

    ranking = iterateMID(relevance, redundancy)
    return {frame.columns[index]: score for index, score in ranking.items()}


def topK(scores: Mapping[Hashable, float], k: int) -> list[Hashable]:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k harus berupa bilangan bulat.")
    if not 1 <= k <= len(scores):
        raise ValueError("k harus berada di antara 1 dan jumlah fitur.")
    return list(scores)[:k]


def mid(relevance: float, redundancies: Sequence[float] = ()) -> float:
    penalty = float(np.mean(redundancies)) if len(redundancies) else 0.0
    return float(relevance) - penalty


def iterateMID(
    relevance: np.ndarray, redundancy: np.ndarray
) -> dict[int, float]:
    relevance = np.asarray(relevance, dtype=float)
    redundancy = np.asarray(redundancy, dtype=float)
    if relevance.ndim != 1 or relevance.size == 0:
        raise ValueError("relevance harus berupa array satu dimensi yang tidak kosong.")
    if redundancy.shape != (relevance.size, relevance.size):
        raise ValueError("redundancy harus berupa matriks persegi sesuai ukuran relevance.")
    if not np.isfinite(relevance).all() or not np.isfinite(redundancy).all():
        raise ValueError("Nilai informasi mutual harus terhingga.")
    if (relevance < 0).any() or (redundancy < 0).any():
        raise ValueError("Nilai informasi mutual tidak boleh negatif.")

    remaining = list(range(relevance.size))
    selected = []
    ranking = {}
    while remaining:
        scores = [
            mid(relevance[index], redundancy[index, selected])
            for index in remaining
        ]
        best = int(np.argmax(scores))
        index = remaining.pop(best)
        ranking[index] = scores[best]
        selected.append(index)
    return ranking


def validateColumn(column: pd.Series) -> np.ndarray:
    if column.isna().any():
        raise ValueError(
            f"Kolom {column.name!r} memiliki nilai yang hilang; bersihkan data terlebih dahulu."
        )
    if (not pd.api.types.is_numeric_dtype(column.dtype)
            or pd.api.types.is_complex_dtype(column.dtype)):
        raise ValueError(
            f"Kolom {column.name!r} harus berisi nilai numerik riil; "
            "siapkan data pada tahap transformasi terlebih dahulu."
        )
    if not np.isfinite(column.to_numpy(dtype=float)).all():
        raise ValueError(
            f"Kolom {column.name!r} memiliki nilai tidak terhingga; "
            "bersihkan data terlebih dahulu."
        )
    return column.to_numpy()


def mutualInformation(x: np.ndarray, y: np.ndarray) -> float:
    countX = Counter(x)
    countY = Counter(y)
    jointCounts = Counter(zip(x, y))
    sampleCount = len(x)
    score = sum(
        (count / sampleCount)
        * np.log((count * sampleCount) / (countX[left] * countY[right]))
        for (left, right), count in jointCounts.items()
    )
    return max(0.0, float(score))
