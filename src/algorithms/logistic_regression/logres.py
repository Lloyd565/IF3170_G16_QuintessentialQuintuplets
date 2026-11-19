#referensi yang digunakan : https://edunexcontentprodhot.blob.core.windows.net/edunex/2026/97453-Artificial-Intelligence-Parent-Class/486049-Logistic-Regression/file/1793356531850_2026_IF3170_Materi11-LinearLogisticRegressionpptx?sv=2024-11-04&spr=https&st=2026-10-30T10%3A35%3A30Z&se=2028-10-29T10%3A35%3A30Z&sr=b&sp=r&sig=oDS70ldLLX2EQjUFYyZ5Q1%2Fuw4PLXW6%2FJs%2BIfQp6EIw%3D&rsct=application%2Fpdf

from dataclasses import dataclass
from functools import partial
from numbers import Integral, Real

import numpy as np


@dataclass
class LogisticRegression:
    weights: np.ndarray
    bias: float
    lossHistory: list[float]

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        values = validateFeatures(X)
        if values.shape[1] != len(self.weights):
            raise ValueError("Jumlah fitur prediksi harus sama dengan jumlah fitur pelatihan.")
        try:
            with np.errstate(over="raise", invalid="raise"):
                scores = values @ self.weights + self.bias
                if not np.isfinite(scores).all():
                    raise FloatingPointError
                positive = sigmoid(scores)
        except FloatingPointError as error:
            raise ValueError("Perhitungan prediksi menghasilkan nilai tidak terhingga.") from error
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        threshold = validateScalar(threshold, "threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold harus berada di antara 0 dan 1.")
        return (self.predictProba(X)[:, 1] >= threshold).astype(int)


def sigmoid(scores: np.ndarray | float) -> np.ndarray:
    with np.errstate(under="ignore"):
        return np.exp(-np.logaddexp(0.0, -np.asarray(scores, dtype=float)))


def sga(
    weights: np.ndarray,
    bias: float,
    row: np.ndarray,
    target: float,
    learningRate: float = 0.01,
) -> tuple[np.ndarray, float]:
    error = target - float(sigmoid(weights @ row + bias))
    return weights + learningRate * error * row, float(bias + learningRate * error)


def sgaL1(
    weights: np.ndarray,
    bias: float,
    row: np.ndarray,
    target: float,
    learningRate: float = 0.01,
    regularizationStrength: float = 0.01,
) -> tuple[np.ndarray, float]:
    updated, updatedBias = sga(weights, bias, row, target, learningRate)
    updated -= learningRate * regularizationStrength * np.sign(weights)
    return updated, updatedBias


def sgaL2(
    weights: np.ndarray,
    bias: float,
    row: np.ndarray,
    target: float,
    learningRate: float = 0.01,
    regularizationStrength: float = 0.01,
) -> tuple[np.ndarray, float]:
    updated, updatedBias = sga(weights, bias, row, target, learningRate)
    updated -= learningRate * regularizationStrength * weights
    return updated, updatedBias


def sgaElasticNet(
    weights: np.ndarray,
    bias: float,
    row: np.ndarray,
    target: float,
    learningRate: float = 0.01,
    regularizationStrength: float = 0.01,
    l1Ratio: float = 0.5,
) -> tuple[np.ndarray, float]:
    updated, updatedBias = sga(weights, bias, row, target, learningRate)
    penalty = l1Ratio * np.sign(weights) + (1.0 - l1Ratio) * weights
    updated -= learningRate * regularizationStrength * penalty
    return updated, updatedBias


def iterateSGA(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learningRate: float = 0.01,
    epochs: int = 100,
    regularization: str | None = "none",
    regularizationStrength: float = 0.01,
    l1Ratio: float = 0.5,
    shuffle: bool = True,
    randomState: int | None = 42,
) -> LogisticRegression:
    values = validateFeatures(X)
    target = validateTarget(y, len(values))
    learningRate = validateScalar(learningRate, "learningRate")
    regularizationStrength = validateScalar(regularizationStrength, "regularizationStrength")
    l1Ratio = validateScalar(l1Ratio, "l1Ratio")
    if learningRate <= 0:
        raise ValueError("learningRate harus lebih besar dari 0.")
    if regularizationStrength < 0:
        raise ValueError("regularizationStrength tidak boleh negatif.")
    if not 0.0 <= l1Ratio <= 1.0:
        raise ValueError("l1Ratio harus berada di antara 0 dan 1.")
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 1:
        raise ValueError("epochs harus berupa bilangan bulat positif.")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("shuffle harus berupa nilai boolean.")
    if randomState is not None:
        if isinstance(randomState, bool) or not isinstance(randomState, Integral) or randomState < 0:
            raise ValueError("randomState harus berupa bilangan bulat nonnegatif atau None.")
        randomState = int(randomState)
    if regularization is None:
        regularization = "none"
    if not isinstance(regularization, str):
        raise TypeError("regularization harus berupa teks atau None.")
    regularization = regularization.lower()
    if regularization not in ("none", "l1", "l2", "elasticnet"):
        raise ValueError("regularization harus berupa 'none', 'l1', 'l2', atau 'elasticnet'.")

    update = sga
    l1Strength = 0.0
    l2Strength = 0.0
    if regularization == "l1":
        update = partial(sgaL1, regularizationStrength=regularizationStrength)
        l1Strength = regularizationStrength
    elif regularization == "l2":
        update = partial(sgaL2, regularizationStrength=regularizationStrength)
        l2Strength = regularizationStrength
    elif regularization == "elasticnet":
        update = partial(
            sgaElasticNet,
            regularizationStrength=regularizationStrength,
            l1Ratio=l1Ratio,
        )
        l1Strength = regularizationStrength * l1Ratio
        l2Strength = regularizationStrength * (1.0 - l1Ratio)

    weights = np.zeros(values.shape[1], dtype=float)
    bias = 0.0
    lossHistory = [logLoss(values @ weights + bias, target, weights, l1Strength, l2Strength)]
    generator = np.random.default_rng(randomState)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for epoch in range(epochs):
                order = generator.permutation(len(values)) if shuffle else range(len(values))
                for index in order:
                    weights, bias = update(
                        weights, bias, values[index], target[index], learningRate
                    )
                loss = logLoss(values @ weights + bias, target, weights, l1Strength, l2Strength)
                if not np.isfinite(loss) or not np.isfinite(weights).all() or not np.isfinite(bias):
                    raise FloatingPointError
                lossHistory.append(loss)
    except FloatingPointError as error:
        raise ValueError(
            "Pelatihan menghasilkan nilai tidak terhingga; "
            "gunakan data yang sudah diskalakan atau learningRate yang lebih kecil."
        ) from error
    return LogisticRegression(weights=weights, bias=bias, lossHistory=lossHistory)


def logLoss(
    scores: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    l1Strength: float = 0.0,
    l2Strength: float = 0.0,
) -> float:
    likelihoodLoss = np.mean(np.logaddexp(0.0, (1.0 - 2.0 * target) * scores))
    penalty = l1Strength * np.sum(np.abs(weights)) + 0.5 * l2Strength * (weights @ weights)
    return float(likelihoodLoss + penalty)


def validateFeatures(X: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(X)
        if np.iscomplexobj(values) or values.dtype.kind in "USV":
            raise ValueError
        values = values.astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("X harus berisi nilai numerik riil yang sudah ditransformasi.") from error
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X harus berupa matriks dua dimensi dengan baris dan fitur yang tidak kosong.")
    if not np.isfinite(values).all():
        raise ValueError("X tidak boleh memiliki nilai yang hilang atau tidak terhingga.")
    return values


def validateTarget(y: np.ndarray, sampleCount: int) -> np.ndarray:
    try:
        target = np.asarray(y)
        if np.iscomplexobj(target) or target.dtype.kind in "USV":
            raise ValueError
        target = target.astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("y harus berisi label numerik 0 dan 1.") from error
    if target.ndim != 1 or len(target) != sampleCount:
        raise ValueError("y harus satu dimensi dengan satu label untuk setiap baris X.")
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("y hanya boleh berisi label 0 dan 1 tanpa nilai yang hilang.")
    if np.unique(target).size != 2:
        raise ValueError("Data pelatihan harus memiliki kedua kelas, yaitu 0 dan 1.")
    return target


def validateScalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} harus berupa bilangan riil terhingga.")
    return float(value)
