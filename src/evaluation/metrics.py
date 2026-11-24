import numpy as np


def confusionCounts(result, testData) -> tuple[int, int, int, int]:
    try:
        predicted = np.asarray(result)
        actual = np.asarray(testData)
    except (TypeError, ValueError) as error:
        raise ValueError("Prediksi dan label asli harus berupa array satu dimensi.") from error
    if predicted.ndim != 1 or actual.ndim != 1 or not len(actual):
        raise ValueError("Prediksi dan label asli harus berupa array satu dimensi yang tidak kosong.")
    if len(predicted) != len(actual):
        raise ValueError("Jumlah prediksi harus sama dengan jumlah label asli.")
    for values in (predicted, actual):
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError("Prediksi dan label asli harus berupa label numerik 0 dan 1 yang terhingga.")
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Prediksi dan label asli hanya boleh berisi label 0 dan 1.")
    trueNegative = int(np.sum((actual == 0) & (predicted == 0)))
    falsePositive = int(np.sum((actual == 0) & (predicted == 1)))
    falseNegative = int(np.sum((actual == 1) & (predicted == 0)))
    truePositive = int(np.sum((actual == 1) & (predicted == 1)))
    return trueNegative, falsePositive, falseNegative, truePositive


def evaluateMetrics(result, testData) -> dict[str, float]:
    trueNegative, falsePositive, falseNegative, truePositive = confusionCounts(result, testData)
    total = trueNegative + falsePositive + falseNegative + truePositive
    predictedPositive = truePositive + falsePositive
    actualPositive = truePositive + falseNegative
    actualNegative = trueNegative + falsePositive
    f1Denominator = 2 * truePositive + falsePositive + falseNegative
    negativeF1Denominator = 2 * trueNegative + falsePositive + falseNegative
    positiveF1 = 2 * truePositive / f1Denominator if f1Denominator else 0.0
    negativeF1 = 2 * trueNegative / negativeF1Denominator if negativeF1Denominator else 0.0
    positiveRecall = truePositive / actualPositive if actualPositive else 0.0
    classRecalls = []
    if actualPositive:
        classRecalls.append(positiveRecall)
    if actualNegative:
        classRecalls.append(trueNegative / actualNegative)
    return {
        "accuracy": (truePositive + trueNegative) / total,
        "precision": truePositive / predictedPositive if predictedPositive else 0.0,
        "recall": positiveRecall,
        "f1Score": positiveF1,
        "macrof1Score": (positiveF1 + negativeF1) / 2,
        "balancedAccuracy": sum(classRecalls) / len(classRecalls),
    }


def accuracy(result, testData) -> float:
    return evaluateMetrics(result, testData)["accuracy"]


def precision(result, testData) -> float:
    return evaluateMetrics(result, testData)["precision"]


def recall(result, testData) -> float:
    return evaluateMetrics(result, testData)["recall"]


def f1Score(result, testData) -> float:
    return evaluateMetrics(result, testData)["f1Score"]


def balancedAccuracy(result, testData) -> float:
    return evaluateMetrics(result, testData)["balancedAccuracy"]


def macrof1Score(result, testData) -> float:
    return evaluateMetrics(result, testData)["macrof1Score"]
