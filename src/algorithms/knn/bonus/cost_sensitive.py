import numpy as np

from src.evaluation.metrics import confusionCounts, evaluateMetrics


def predict_cost_sensitive(p_cancel, *, fp_cost=2, fn_cost=1):
    threshold = fp_cost / (fp_cost + fn_cost)
    return (np.asarray(p_cancel, dtype=float) >= threshold).astype(int)


def evaluate_cost(y_true, y_pred, *, fp_cost=2, fn_cost=1):
    _, fp, fn, _ = confusionCounts(y_pred, y_true)
    return {
        "FP": fp,
        "FN": fn,
        "F1 binary": evaluateMetrics(y_pred, y_true)["f1Score"],
        "Total biaya": fp_cost * fp + fn_cost * fn,
    }