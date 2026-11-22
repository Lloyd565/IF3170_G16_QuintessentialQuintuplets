from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ..core.config import METRICS_DIR
from ..evaluation import evaluateMetrics


def evaluate(predictions, testData, *, outputDir=METRICS_DIR) -> dict[str, dict[str, float]]:
    if not isinstance(predictions, Mapping) or not predictions:
        raise TypeError("predictions harus berupa pemetaan nama model ke hasil prediksi yang tidak kosong.")
    if any(not isinstance(name, str) or not name for name in predictions):
        raise ValueError("Nama model harus berupa teks yang tidak kosong.")
    results = {name: evaluateMetrics(result, testData) for name, result in predictions.items()}
    if outputDir is not None:
        path = Path(outputDir) / "metrics.csv"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame.from_dict(results, orient="index").rename_axis("model").reset_index()
            frame.to_csv(path, index=False)
        except OSError as error:
            raise ValueError(f"Hasil evaluasi tidak dapat disimpan ke {path}.") from error
    return results
