import numpy as np

from ..algorithms.dtl import dtl
from ..algorithms.knn import KNN
from ..algorithms.logistic_regression import logisticRegression
from ..algorithms.naive_bayes import naive_bayes as naiveBayes
from ..algorithms.svm import svm


def trainKNN(X, y, **parameters):
    return KNN(**parameters).fit(np.asarray(X), np.asarray(y))


TRAINERS = {
    "knn": trainKNN,
    "dtl": dtl,
    "logistic_regression": logisticRegression,
    "svm": svm,
    "naive_bayes": naiveBayes,
}


def getModelNames() -> tuple[str, ...]:
    return tuple(TRAINERS)


def validateModelNames(modelNames) -> list[str]:
    available = getModelNames()
    if modelNames is None:
        names = list(available)
    elif isinstance(modelNames, str):
        names = [modelNames]
    else:
        try:
            names = list(modelNames)
        except TypeError as error:
            raise TypeError("modelNames harus berupa nama model atau kumpulan nama model.") from error
    if not names or any(not isinstance(name, str) or name not in available for name in names):
        raise ValueError(f"Nama model harus salah satu dari: {', '.join(available)}.")
    if len(set(names)) != len(names):
        raise ValueError("Nama model tidak boleh berulang.")
    return names


def getTrainer(modelName):
    name = validateModelNames([modelName])[0]
    return TRAINERS[name]
