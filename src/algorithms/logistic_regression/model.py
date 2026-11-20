import numpy as np

from .logres import LogisticRegression, iterateSGA


def logisticRegression(
    X: np.ndarray,
    y: np.ndarray,
    regularization: str | None = "none",
    *,
    learningRate: float = 0.01,
    epochs: int = 100,
    regularizationStrength: float = 0.01,
    l1Ratio: float = 0.5,
    shuffle: bool = True,
    randomState: int | None = 42,
) -> LogisticRegression:
    return iterateSGA(
        X,
        y,
        learningRate=learningRate,
        epochs=epochs,
        regularization=regularization,
        regularizationStrength=regularizationStrength,
        l1Ratio=l1Ratio,
        shuffle=shuffle,
        randomState=randomState,
    )
