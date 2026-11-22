from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ..algorithms.dtl import dtl
from ..algorithms.knn import KNN
from ..algorithms.logistic_regression import logisticRegression
from ..algorithms.naive_bayes import naive_bayes as naiveBayes
from ..algorithms.svm import svm
from ..core.config import MODELS_DIR
from ..core.serialization import saveModel


def train(X, y, *, modelNames=None, modelParameters=None, modelDir=MODELS_DIR) -> dict:
    trainers = {
        "knn": (trainKNN, saveKNNModel),
        "dtl": (trainDTL, saveDTLModel),
        "logistic_regression": (trainLogRes, saveLogResModel),
        "svm": (trainSVM, saveSVMModel),
        "naive_bayes": (trainBayes, saveBayesModel),
    }
    names = validateModelNames(modelNames, trainers)
    parameters = validateModelParameters(modelParameters, names)
    values = validateFeatures(X)
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != len(values):
        raise ValueError("y harus satu dimensi dengan satu label untuk setiap baris X.")
    if pd.isna(target).any():
        raise ValueError("y tidak boleh memiliki nilai yang hilang.")
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        if not X.index.equals(y.index):
            raise ValueError("Indeks baris X dan y harus sama dan memiliki urutan yang sama.")

    models = {}
    for name in names:
        try:
            model = trainers[name][0](values.copy(), target.copy(), **parameters.get(name, {}))
        except NotImplementedError:
            continue
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"Hasil pelatihan {name!r} harus berupa model yang dapat memprediksi.")
        model.featureCount = values.shape[1]
        model.featureNames = tuple(X.columns) if isinstance(X, pd.DataFrame) else None
        models[name] = model
    if not models:
        raise NotImplementedError("Belum ada algoritma yang tersedia untuk dilatih.")
    for name, model in models.items():
        trainers[name][1](model, modelDir)
    return models


def trainDTL(X, y, **parameters):
    return dtl(X, y, **parameters)


def trainKNN(X, y, **parameters):
    return KNN(**parameters).fit(np.asarray(X), np.asarray(y))


def trainLogRes(X, y, **parameters):
    return logisticRegression(X, y, **parameters)


def trainBayes(X, y, **parameters):
    return naiveBayes(X, y, **parameters)


def trainSVM(X, y, **parameters):
    return svm(X, y, **parameters)


def saveDTLModel(model, modelDir=MODELS_DIR) -> Path:
    return saveModel(model, Path(modelDir) / "dtl.pkl")


def saveKNNModel(model, modelDir=MODELS_DIR) -> Path:
    return saveModel(model, Path(modelDir) / "knn.pkl")


def saveLogResModel(model, modelDir=MODELS_DIR) -> Path:
    return saveModel(model, Path(modelDir) / "logistic_regression.pkl")


def saveBayesModel(model, modelDir=MODELS_DIR) -> Path:
    return saveModel(model, Path(modelDir) / "naive_bayes.pkl")


def saveSVMModel(model, modelDir=MODELS_DIR) -> Path:
    return saveModel(model, Path(modelDir) / "svm.pkl")


def validateModelNames(modelNames, available) -> list[str]:
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
        raise ValueError("Nama model harus berupa knn, dtl, logistic_regression, svm, atau naive_bayes.")
    if len(set(names)) != len(names):
        raise ValueError("Nama model tidak boleh berulang.")
    return names


def validateModelParameters(modelParameters, names) -> dict:
    if modelParameters is None:
        return {}
    if not isinstance(modelParameters, Mapping):
        raise TypeError("modelParameters harus berupa pemetaan nama model ke parameter.")
    if any(name not in names for name in modelParameters):
        raise ValueError("Parameter hanya boleh diberikan untuk model yang dipilih.")
    if any(not isinstance(parameters, Mapping) for parameters in modelParameters.values()):
        raise TypeError("Parameter setiap model harus berupa pemetaan nama parameter ke nilai.")
    if any(not isinstance(key, str) for parameters in modelParameters.values() for key in parameters):
        raise TypeError("Nama parameter harus berupa teks.")
    return {name: dict(parameters) for name, parameters in modelParameters.items()}


def validateFeatures(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame) and not X.columns.is_unique:
        raise ValueError("Nama kolom pada X harus unik.")
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
