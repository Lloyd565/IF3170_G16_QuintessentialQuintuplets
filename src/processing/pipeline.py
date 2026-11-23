import pandas as pd

from .cleaning import cleaning
from .dimensionality_reduction import dimensionality_reduction
from .dtl import DTLPreprocessor
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
    components=None,
    oversample=False,
    dtlPreprocessing=False,
    randomState=42,
    name="data",
):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise ValueError("X harus berupa DataFrame yang tidak kosong dengan kolom unik.")
    if y is not None:
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise ValueError("y harus berupa Series dengan indeks dan urutan yang sama dengan X.")
    if fit:
        if dtlPreprocessing and not (clean and transform):
            raise ValueError("DTL preprocessing membutuhkan cleaning dan transformation aktif.")
        state = {
            "version": 1,
            "inputColumns": X.columns.tolist(),
            "clean": clean,
            "transform": transform,
            "cleaner": None,
            "transformer": None,
            "outputColumns": None,
            "dtlPreprocessor": DTLPreprocessor() if dtlPreprocessing else None,
        }
    elif not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Butuh state preprocessing yang sudah di-fit dan memiliki format yang sesuai.")
    if set(X.columns) != set(state["inputColumns"]):
        raise ValueError("Kolom data harus sama dengan kolom saat preprocessing dilatih.")
    features = X.loc[:, state["inputColumns"]].copy()
    target = None if y is None else y.copy()
    dtlPreprocessor = state.get("dtlPreprocessor")
    if dtlPreprocessor is not None:
        features = dtlPreprocessor.fit_transform(features) if fit else dtlPreprocessor.transform(features)
    elif state["clean"]:
        features, state["cleaner"] = cleaning(
            features, cleaner=state["cleaner"], fit=fit, name=name,
        )
    if dtlPreprocessor is None and state["transform"]:
        features, state["transformer"] = transformation(
            features, transformer=state["transformer"], fit=fit,
        )
    if fit:
        if featureCount is not None:
            if target is None:
                raise ValueError("Seleksi fitur membutuhkan label pelatihan.")
            features, _ = featureSelection(features, target, featureCount)
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
