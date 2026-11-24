import pandas as pd

from .cleaning import cleaning
from .cleaning.validation import CONTINUOUS_COLS
from .dimensionality_reduction import dimensionality_reduction
from .feature_selection import featureSelection
from .transformation import handle_imbalance as handleImbalance, transformation


def runPreprocessing(
    X,
    y=None,
    *,
    fit=False,
    state=None,
    clean=True,
    transform=True,
    featureCount=None,
    continuousColumns=None,
    binCount=10,
    components=None,
    oversample=False,
    randomState=42,
    name="data",
):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise ValueError("X harus berupa DataFrame yang tidak kosong dengan kolom unik.")
    if y is not None:
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise ValueError("y harus berupa Series dengan indeks dan urutan yang sama dengan X.")
    if fit:
        state = {
            "version": 1,
            "inputColumns": X.columns.tolist(),
            "clean": clean,
            "transform": transform,
            "cleaner": None,
            "transformer": None,
            "outputColumns": None,
        }
    elif not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Butuh state preprocessing yang sudah di-fit dan memiliki format yang sesuai.")
    if set(X.columns) != set(state["inputColumns"]):
        raise ValueError("Kolom data harus sama dengan kolom saat preprocessing dilatih.")
    features = X.loc[:, state["inputColumns"]].copy()
    target = None if y is None else y.copy()
    if state["clean"]:
        features, state["cleaner"] = cleaning(
            features, cleaner=state["cleaner"], fit=fit, name=name,
        )
    if state["transform"]:
        features, state["transformer"] = transformation(
            features, transformer=state["transformer"], fit=fit,
        )
    if fit:
        if featureCount is not None:
            if target is None:
                raise ValueError("Seleksi fitur membutuhkan label pelatihan.")
            selectionColumns = continuousColumns
            if selectionColumns is None:
                selectionColumns = [column for column in CONTINUOUS_COLS if column in features.columns]
            features, _ = featureSelection(
                features, target, featureCount,
                continuousColumns=selectionColumns, binCount=binCount,
            )
        state["selectedColumns"] = features.columns.tolist()
    else:
        features = features.loc[:, state.get("selectedColumns", state["outputColumns"])].copy()
    if fit:
        state["reducer"] = None
        if components is not None:
            features, state["reducer"] = dimensionality_reduction(
                features, fit=True, nComponents=components,
            )
        state["outputColumns"] = features.columns.tolist()
    elif state.get("reducer") is not None:
        features, _ = dimensionality_reduction(features, reducer=state["reducer"])
    if fit and oversample:
        if target is None:
            raise ValueError("Oversampling membutuhkan label pelatihan.")
        features, target = handleImbalance(features, target, random_state=randomState)
    return features, target, state
