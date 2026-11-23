from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ..algorithms.knn.bonus import predict_cost_sensitive as predictCostSensitive
from ..core.config import MODELS_DIR, SUBMISSIONS_DIR
from ..core.registry import validateModelNames
from ..core.serialization import loadModel
from .train import validateFeatures, validateModelParameters


def predict(
    X,
    models=None,
    *,
    modelNames=None,
    modelParameters=None,
    costSensitive=False,
    modelDir=MODELS_DIR,
    outputDir=SUBMISSIONS_DIR,
    ids=None,
    template=None,
    targetColumn="is_canceled",
    idColumn="id",
) -> dict[str, np.ndarray]:
    if models is None:
        names = validateModelNames(modelNames)
        models = {}
        for name in names:
            path = Path(modelDir) / f"{name}.pkl"
            if modelNames is None and not path.exists():
                continue
            models[name] = loadModel(path)
        names = list(models)
    else:
        if not isinstance(models, Mapping):
            raise TypeError("models harus berupa pemetaan nama model ke model terlatih.")
        names = validateModelNames(list(models) if modelNames is None else modelNames)
        if any(name not in models for name in names):
            raise ValueError("Semua model yang dipilih harus tersedia pada models.")
    if not names:
        raise FileNotFoundError("Tidak ada model tersimpan yang tersedia untuk prediksi.")
    parameters = validateModelParameters(modelParameters, names)

    predictions = {}
    for name in names:
        model = models[name]
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"Model {name!r} tidak memiliki fungsi prediksi.")
        values = prepareFeatures(model, X)
        if costSensitive and name != "svm":
            probabilities = model.predictProba(values) if name == "logistic_regression" else model.predict_proba(values)
            result = predictCostSensitive(np.asarray(probabilities)[:, 1])
        else:
            result = model.predict(values, **parameters.get(name, {}))
        predictions[name] = validatePrediction(result, len(values))
    for name, result in predictions.items():
        savePrediction(
            result,
            Path(outputDir) / f"{name}.csv",
            ids=ids,
            template=template,
            targetColumn=targetColumn,
            idColumn=idColumn,
        )
    return predictions


def prepareFeatures(model, X) -> np.ndarray:
    featureNames = getattr(model, "featureNames", None)
    if isinstance(X, pd.DataFrame) and featureNames is not None:
        if not X.columns.is_unique or set(X.columns) != set(featureNames):
            raise ValueError("Kolom prediksi harus sama dengan kolom pelatihan.")
        X = X.loc[:, list(featureNames)]
    values = validateFeatures(X)
    featureCount = getattr(model, "featureCount", None)
    if featureCount is not None and values.shape[1] != featureCount:
        raise ValueError("Jumlah fitur prediksi harus sama dengan jumlah fitur pelatihan.")
    return values


def validatePrediction(predictions, sampleCount=None) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Prediksi harus berupa array satu dimensi yang tidak kosong.")
    if sampleCount is not None and len(values) != sampleCount:
        raise ValueError("Jumlah prediksi harus sama dengan jumlah baris data.")
    if pd.isna(values).any() or np.iscomplexobj(values):
        raise ValueError("Prediksi tidak boleh memiliki nilai yang hilang atau bilangan kompleks.")
    if values.dtype.kind in "biuf" and not np.isfinite(values).all():
        raise ValueError("Prediksi tidak boleh memiliki nilai tidak terhingga.")
    return values


def savePrediction(
    predictions,
    path,
    *,
    ids=None,
    template=None,
    targetColumn="is_canceled",
    idColumn="id",
) -> Path:
    values = validatePrediction(predictions)
    if not isinstance(targetColumn, str) or not targetColumn:
        raise ValueError("Nama kolom target harus berupa teks yang tidak kosong.")
    if not isinstance(idColumn, str) or not idColumn or idColumn == targetColumn:
        raise ValueError("Nama kolom identitas harus berupa teks dan berbeda dari kolom target.")
    identifiers = None
    if ids is not None:
        identifiers = np.asarray(ids)
        if identifiers.ndim != 1 or len(identifiers) != len(values):
            raise ValueError("Jumlah identitas harus sama dengan jumlah prediksi.")
        if pd.isna(identifiers).any() or pd.Series(identifiers).duplicated().any():
            raise ValueError("Identitas tidak boleh kosong atau berulang.")

    if template is None:
        frame = pd.DataFrame({targetColumn: values})
        if identifiers is not None:
            frame.insert(0, idColumn, identifiers)
    else:
        if isinstance(template, pd.DataFrame):
            frame = template.copy(deep=True)
        else:
            try:
                frame = pd.read_csv(template)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                raise ValueError("Templat prediksi tidak dapat dibaca.") from error
        if not frame.columns.is_unique or targetColumn not in frame.columns:
            raise ValueError("Templat harus memiliki kolom unik dan memuat kolom target.")
        if len(frame) != len(values):
            raise ValueError("Jumlah baris templat harus sama dengan jumlah prediksi.")
        if idColumn in frame.columns:
            if frame[idColumn].isna().any() or frame[idColumn].duplicated().any():
                raise ValueError("Identitas pada templat tidak boleh kosong atau berulang.")
        if identifiers is not None:
            if idColumn not in frame.columns or not np.array_equal(frame[idColumn].to_numpy(), identifiers):
                raise ValueError("Identitas dan urutannya harus sama dengan templat prediksi.")
        frame[targetColumn] = values

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    except OSError as error:
        raise ValueError(f"Prediksi tidak dapat disimpan ke {path}.") from error
    return path
