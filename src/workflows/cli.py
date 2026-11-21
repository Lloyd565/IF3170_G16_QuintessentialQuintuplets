import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split as trainTestSplit

from ..core.serialization import loadModel, saveModel
from ..processing.pipeline import runPreprocessing
from .predict import predict
from .train import train


def readData(path, idColumn):
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as source:
            columns = next(csv.reader(source), [])
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("CSV harus memiliki nama kolom yang tidak kosong dan unik.")
        frame = pd.read_csv(path, dtype={idColumn: str})
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"Berkas CSV tidak dapat dibaca: {path}.") from error
    if frame.empty:
        raise ValueError("Berkas CSV harus memiliki setidaknya satu baris data.")
    if idColumn in frame:
        if frame[idColumn].isna().any() or frame[idColumn].duplicated().any():
            raise ValueError("Identitas data tidak boleh kosong atau berulang.")
    return frame


def selectedModels(args):
    if "all" in args.model:
        if len(args.model) != 1:
            raise ValueError("Pilihan all tidak boleh digabung dengan nama model lain.")
        return None
    if len(set(args.model)) != len(args.model):
        raise ValueError("Nama model tidak boleh berulang.")
    return args.model


def runTrain(args, modelParameters):
    if args.target == args.idColumn:
        raise ValueError("Nama kolom target dan identitas harus berbeda.")
    if not 0 <= args.validationSize < 1:
        raise ValueError("validationSize harus berada di antara 0 dan 1, tanpa menyertakan 1.")
    if args.randomState < 0:
        raise ValueError("randomState tidak boleh negatif.")
    frame = readData(args.input, args.idColumn)
    if args.target not in frame:
        raise ValueError(f"Kolom target tidak ditemukan: {args.target}.")
    target = frame[args.target].copy()
    if target.isna().any() or not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("Kolom target harus memiliki kedua kelas 0 dan 1 tanpa nilai yang hilang.")
    features = frame.drop(columns=[args.target, args.idColumn], errors="ignore")
    validFeatures = None
    validTarget = None
    if args.validationSize:
        try:
            features, validFeatures, target, validTarget = trainTestSplit(
                features, target, test_size=args.validationSize,
                random_state=args.randomState, stratify=target,
            )
        except ValueError as error:
            raise ValueError("Data tidak cukup untuk pembagian train dan validasi yang diminta.") from error
    keep = ~features.duplicated()
    features, target = features.loc[keep], target.loc[keep]
    features, target, state = runPreprocessing(
        features, target, fit=True, clean=args.cleaning, transform=args.transformation,
        featureCount=args.features, components=args.components, oversample=args.oversample, randomState=args.randomState,
        name="train",
    )
    if validFeatures is not None:
        validFeatures, validTarget, state = runPreprocessing(
            validFeatures, validTarget, state=state, name="validation",
        )
    models = train(
        features, target, modelNames=selectedModels(args),
        modelParameters=modelParameters, modelDir=args.modelDir,
    )
    state["modelNames"] = list(models)
    saveModel(state, Path(args.modelDir) / "preprocessing.pkl")
    if validFeatures is not None:
        ids = frame.loc[validFeatures.index, args.idColumn] if args.idColumn in frame else None
        predict(validFeatures, models, outputDir=args.validationOutput, ids=ids,
                targetColumn=args.target, idColumn=args.idColumn)
    return models


def runPredict(args):
    if args.target == args.idColumn:
        raise ValueError("Nama kolom target dan identitas harus berbeda.")
    frame = readData(args.input, args.idColumn)
    ids = frame[args.idColumn].copy() if args.idColumn in frame else None
    features = frame.drop(columns=[args.target, args.idColumn], errors="ignore")
    names = selectedModels(args)
    if not args.prepared:
        state = loadModel(Path(args.modelDir) / "preprocessing.pkl")
        features, _, state = runPreprocessing(features, state=state, name="test")
        trainedNames = state.get("modelNames", [])
        if not trainedNames:
            raise ValueError("State preprocessing tidak memiliki daftar model terlatih.")
        if names is not None and any(name not in trainedNames for name in names):
            raise ValueError("Model yang dipilih tidak dilatih dengan state preprocessing tersimpan.")
        if names is None:
            names = trainedNames
    template = None
    if args.template is not None:
        if ids is None:
            raise ValueError("Data prediksi harus memiliki kolom identitas untuk dicocokkan dengan templat.")
        template = readData(args.template, args.idColumn)
    parameters = {"logistic_regression": {"threshold": args.threshold}} if args.threshold is not None else None
    if parameters and names is not None and "logistic_regression" not in names:
        raise ValueError("threshold hanya dapat digunakan saat logistic_regression dipilih.")
    return predict(
        features, modelNames=names, modelParameters=parameters, modelDir=args.modelDir,
        outputDir=args.outputDir, ids=ids, template=template,
        targetColumn=args.target, idColumn=args.idColumn,
    )
