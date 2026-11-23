#referensi yang digunakan : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from ..algorithms.sklearn_baselines import getBaseline
from ..core.config import METRICS_DIR
from ..core.registry import validateModelNames
from ..evaluation import evaluateMetrics
from .train import validateFeatures, validateModelParameters


def baselineOptions(modelName, parameters):
    options = dict(parameters)
    if modelName == "knn":
        return {"n_neighbors": options.get("k", 3)}
    if modelName == "dtl":
        return options
    if modelName == "logistic_regression":
        regularization = options.get("regularization") or "none"
        strength = options.get("regularizationStrength", 0.01)
        return {
            "penalty": None if regularization.lower() == "none" or strength == 0 else regularization.lower(),
            "alpha": strength,
            "eta0": options.get("learningRate", 0.01),
            "max_iter": options.get("epochs", 100),
            "l1_ratio": options.get("l1Ratio", 0.5),
            "shuffle": options.get("shuffle", True),
            "random_state": options.get("randomState", 42),
        }
    if modelName == "svm":
        options.pop("max_passes", None)
        return options
    return {"var_smoothing": options.get("smoothing", 1e-9)}


def prepareComparisonData(trainingData, trainingTarget, testData, testTarget):
    for features, target in ((trainingData, trainingTarget), (testData, testTarget)):
        if isinstance(features, pd.DataFrame) and isinstance(target, pd.Series):
            if not features.index.equals(target.index):
                raise ValueError("Indeks dan urutan fitur harus sama dengan label pada data perbandingan.")
    if isinstance(trainingData, pd.DataFrame) and isinstance(testData, pd.DataFrame):
        if set(trainingData.columns) != set(testData.columns):
            raise ValueError("Kolom data uji harus sama dengan kolom data latih.")
        testData = testData.loc[:, trainingData.columns]
    trainingValues = validateFeatures(trainingData)
    testValues = validateFeatures(testData)
    if trainingValues.shape[1] != testValues.shape[1]:
        raise ValueError("Jumlah fitur data latih dan data uji harus sama.")
    evaluateMetrics(trainingTarget, trainingTarget)
    evaluateMetrics(testTarget, testTarget)
    trainingLabels = np.asarray(trainingTarget)
    testLabels = np.asarray(testTarget)
    if len(trainingLabels) != len(trainingValues) or len(testLabels) != len(testValues):
        raise ValueError("Jumlah label harus sama dengan jumlah baris data perbandingan.")
    if np.unique(trainingLabels).size != 2:
        raise ValueError("Data latih perbandingan harus memiliki kedua kelas 0 dan 1.")
    return trainingValues, trainingLabels, testValues, testLabels


def compare(
    trainingData,
    trainingTarget,
    testData,
    testTarget,
    results,
    *,
    modelParameters=None,
    baselineParameters=None,
    threshold=0.5,
    outputDir=METRICS_DIR,
) -> pd.DataFrame:
    if not isinstance(results, Mapping) or not results:
        raise TypeError("results harus memetakan nama model ke prediksi from scratch dan tidak boleh kosong.")
    names = validateModelNames(list(results))
    parameters = validateModelParameters(modelParameters, names)
    overrides = validateModelParameters(baselineParameters, names)
    if (
        isinstance(threshold, bool) or not isinstance(threshold, Real)
        or not np.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold harus berupa bilangan terhingga di antara 0 dan 1.")
    trainingValues, trainingLabels, testValues, testLabels = prepareComparisonData(
        trainingData, trainingTarget, testData, testTarget,
    )
    scratchScores = {}
    for name, result in results.items():
        if isinstance(result, pd.Series) and isinstance(testTarget, pd.Series):
            if not result.index.equals(testTarget.index):
                raise ValueError("Indeks dan urutan prediksi harus sama dengan label data uji.")
        scratchScores[name] = evaluateMetrics(result, testLabels)
    rows = []
    for name in names:
        options = baselineOptions(name, parameters.get(name, {}))
        options.update(overrides.get(name, {}))
        model = getBaseline(name, **options)
        try:
            model.fit(trainingValues.copy(), trainingLabels.copy())
            if name == "logistic_regression":
                positive = int(np.flatnonzero(model.classes_ == 1)[0])
                prediction = (model.predict_proba(testValues)[:, positive] >= threshold).astype(int)
            else:
                prediction = model.predict(testValues)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Model acuan {name!r} gagal dilatih atau diprediksi; periksa parameter dan data.") from error
        baselineScores = evaluateMetrics(prediction, testLabels)
        for metric, scratchScore in scratchScores[name].items():
            baselineScore = baselineScores[metric]
            difference = scratchScore - baselineScore
            relative = 100 * difference / baselineScore if baselineScore else (0.0 if scratchScore == 0 else None)
            rows.append({
                "model": name,
                "baseline": type(model).__name__,
                "metric": metric,
                "fromScratch": scratchScore,
                "scikitLearn": baselineScore,
                "difference": difference,
                "differencePercentagePoints": 100 * difference,
                "relativeImprovementPercent": relative,
                "winner": "fromScratch" if difference > 0 else "scikitLearn" if difference < 0 else "tie",
            })
    report = pd.DataFrame(rows)
    if outputDir is not None:
        path = Path(outputDir) / "comparison.csv"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(path, index=False)
        except OSError as error:
            raise ValueError(f"Hasil perbandingan tidak dapat disimpan ke {path}.") from error
    return report
