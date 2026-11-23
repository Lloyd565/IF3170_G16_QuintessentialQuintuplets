import argparse
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys

from .core.config import DATA_DIR, MEDIA_DIR, METRICS_DIR, MODELS_DIR, OUTPUT_DIR, SUBMISSIONS_DIR
from .core.registry import getModelNames
from .workflows.cli import runEvaluate, runPredict, runTrain, selectedModels, showMetrics
from .workflows.interactive import interactiveArguments


class CliParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, "Argumen tidak valid. Gunakan --help untuk melihat pilihan dan format yang tersedia.\n")


def parseParameters(value):
    try:
        parameters = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("Parameter model harus berupa JSON yang valid.") from error
    if not isinstance(parameters, dict):
        raise argparse.ArgumentTypeError("Parameter model harus berupa objek JSON.")
    return parameters


def parseGamma(value):
    if value == "scale":
        return value
    try:
        gamma = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("gamma harus berupa scale atau bilangan positif.") from error
    if not math.isfinite(gamma) or gamma <= 0:
        raise argparse.ArgumentTypeError("gamma harus berupa scale atau bilangan positif.")
    return gamma


def parseComponents(value):
    try:
        components = int(value) if value.isdigit() else float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("components harus int >= 1 atau float di antara 0 dan 1.") from error
    if isinstance(components, int) and components < 1 or isinstance(components, float) and not 0 < components < 1:
        raise argparse.ArgumentTypeError("components harus int >= 1 atau float di antara 0 dan 1.")
    return components


def createParser():
    parser = CliParser(description="Preprocess data, train models, evaluate, and save predictions.")
    parser.add_argument("--interactive", action="store_true", help="Ask for options interactively.")
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "run": "Train models and predict on a separate test CSV.",
        "train": "Fit preprocessing and models on training data.",
        "predict": "Reuse saved preprocessing and models for prediction.",
        "evaluate": "Evaluate saved models on a labeled CSV using fitted preprocessing.",
        "compare": "Train models and compare with scikit-learn on the validation split.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--model", nargs="+", choices=("all",) + getModelNames(), default=["all"])
        training = name in ("train", "run", "compare")
        command.add_argument(
            "--input", type=Path, required=name == "evaluate",
            default=None if name == "evaluate" else DATA_DIR / ("train.csv" if training else "test.csv"),
        )
        command.add_argument("--target", default="is_canceled")
        command.add_argument("--id-column", dest="idColumn", default="id")
        command.add_argument("--model-dir", dest="modelDir", type=Path, default=MODELS_DIR)
        command.add_argument("--output-dir", dest="outputDir", type=Path,
                            default=OUTPUT_DIR / "validation" if name == "evaluate" else SUBMISSIONS_DIR)
        command.add_argument("--threshold", type=float, help="Logistic regression prediction threshold.")
        if name in ("train", "run", "compare", "evaluate"):
            command.add_argument("--metrics-dir", dest="metricsDir", type=Path, default=METRICS_DIR)
        if training:
            addTrainingArguments(command)
        if name == "compare":
            command.set_defaults(compareModels=True)
        if name in ("predict", "run"):
            command.add_argument("--template", type=Path, help="Authoritative sample-submission CSV.")
            if name == "run":
                command.add_argument("--test-input", dest="testInput", type=Path, default=DATA_DIR / "test.csv")
        if name in ("predict", "evaluate"):
            command.add_argument("--prepared", action="store_true", help="Input already matches model features.")
    return parser


def addTrainingArguments(command):
    preprocessing = command.add_argument_group("preprocessing")
    preprocessing.add_argument("--cleaning", action=argparse.BooleanOptionalAction, default=True)
    preprocessing.add_argument("--transformation", action=argparse.BooleanOptionalAction, default=True)
    preprocessing.add_argument("--features", type=int, help="Number of columns selected by mRMR.")
    preprocessing.add_argument("--components", type=parseComponents,
                               help="PCA: int = jumlah komponen, float 0-1 = target variansi kumulatif.")
    preprocessing.add_argument("--oversample", action="store_true")
    preprocessing.add_argument("--validation-size", dest="validationSize", type=float, default=0.2)
    preprocessing.add_argument("--validation-output", dest="validationOutput", type=Path, default=OUTPUT_DIR / "validation")
    preprocessing.add_argument("--random-state", dest="randomState", type=int, default=42)
    models = command.add_argument_group("model parameters")
    models.add_argument("--model-params", dest="modelParams", type=parseParameters, default={})
    models.add_argument("--k", type=int, default=3)
    models.add_argument("--epochs", type=int, default=100)
    models.add_argument("--learning-rate", dest="learningRate", type=float, default=0.01)
    models.add_argument("--regularization", choices=("none", "l1", "l2", "elasticnet"), default="none")
    models.add_argument("--regularization-strength", dest="regularizationStrength", type=float, default=0.01)
    models.add_argument("--l1-ratio", dest="l1Ratio", type=float, default=0.5)
    models.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    models.add_argument("--kernel", choices=("linear", "poly", "rbf"), default="rbf")
    models.add_argument("--C", type=float, default=1.0)
    models.add_argument("--degree", type=int, default=3)
    models.add_argument("--gamma", type=parseGamma, default="scale")
    models.add_argument("--coef0", type=float, default=0.0)
    models.add_argument("--tol", type=float, default=1e-3)
    models.add_argument("--max-passes", dest="maxPasses", type=int, default=5)
    models.add_argument("--max-iter", dest="maxIter", type=int, default=100000)
    models.add_argument("--class-weight", dest="classWeight", choices=("none", "balanced"), default="none",
                        help="SVM: balanced menaikkan C kelas minoritas sebesar n / (2 * n_kelas).")
    comparison = command.add_argument_group("scikit-learn comparison")
    comparison.add_argument("--compare", dest="compareModels", action="store_true",
                            help="Compare with scikit-learn using the same training and validation data.")
    comparison.add_argument("--baseline-params", dest="baselineParams", type=parseParameters, default={},
                            help="Per-model JSON overrides using scikit-learn parameter names.")
    bonus = command.add_argument_group("logistic regression bonus")
    bonus.add_argument("--logres-bonus", dest="logresBonus", action="store_true",
                       help="Record parameter history and generate an MP4 training video.")
    bonus.add_argument("--bonus-output", dest="bonusOutput", type=Path,
                       default=MEDIA_DIR / "logistic_regression_training.mp4")
    bonus.add_argument("--bonus-feature", dest="bonusFeature", default="0",
                       help="Prepared feature name or zero-based index for the parameter plot.")
    bayesBonus = command.add_argument_group("naive bayes bonus")
    bayesBonus.add_argument("--naive-bayes-bonus", dest="naiveBayesBonus", action="store_true",
                           help="Generate the Naive Bayes feature/class graph after training.")
    bayesBonus.add_argument("--naive-bayes-bonus-output", dest="naiveBayesBonusOutput", type=Path,
                           default=MEDIA_DIR / "naive_bayes_graph.png",
                           help="Naive Bayes graph output path (.png, .svg, or .pdf).")


def modelParameters(args):
    names = selectedModels(args)
    names = list(getModelNames()) if names is None else names
    parameters = {
        "knn": {"k": args.k},
        "logistic_regression": {
            "learningRate": args.learningRate,
            "epochs": args.epochs,
            "regularization": args.regularization,
            "regularizationStrength": args.regularizationStrength,
            "l1Ratio": args.l1Ratio,
            "shuffle": args.shuffle,
            "randomState": args.randomState,
        },
        "svm": {
            "C": args.C, "kernel": args.kernel, "degree": args.degree, "gamma": args.gamma,
            "coef0": args.coef0, "tol": args.tol, "max_passes": args.maxPasses,
            "max_iter": args.maxIter, "random_state": args.randomState,
            "class_weight": None if args.classWeight == "none" else args.classWeight,
        },
    }
    parameters = {name: values for name, values in parameters.items() if name in names}
    for name, overrides in args.modelParams.items():
        if name not in names or not isinstance(overrides, dict):
            raise ValueError("Parameter JSON harus memetakan model yang dipilih ke objek parameter.")
        parameters.setdefault(name, {}).update(overrides)
    if "svm" in parameters:
        validateSVMParameters(parameters["svm"])
    return parameters


def validateSVMParameters(parameters):
    for name in ("C", "tol", "coef0"):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Parameter SVM {name} harus berupa bilangan riil terhingga.")
        if name != "coef0" and value <= 0:
            raise ValueError(f"Parameter SVM {name} harus positif.")
    for name in ("degree", "max_passes", "max_iter"):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"Parameter SVM {name} harus berupa bilangan bulat positif.")
    if parameters["kernel"] not in ("linear", "poly", "rbf"):
        raise ValueError("Kernel SVM harus berupa linear, poly, atau rbf.")
    gamma = parameters["gamma"]
    if gamma != "scale":
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma SVM harus berupa scale atau bilangan positif.")
    seed = parameters["random_state"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("random_state SVM harus berupa bilangan bulat nonnegatif atau None.")


def main(argv=None):
    parser = createParser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments == ["--interactive"] or (not arguments and sys.stdin.isatty()):
            arguments = interactiveArguments()
        elif not arguments:
            parser.print_help()
            return 0
        args = parser.parse_args(arguments)
        if args.command in ("train", "run", "compare"):
            models = runTrain(args, modelParameters(args))
            print(f"Models saved to {args.modelDir}: {', '.join(models)}")
            if args.command == "run":
                args.input = args.testInput
                args.model = list(models)
                args.prepared = False
                predictions = runPredict(args)
                print(f"Predictions saved to {args.outputDir}: {', '.join(predictions)}")
        elif args.command == "evaluate":
            results = runEvaluate(args)
            showMetrics(results)
            print(f"Evaluation metrics saved to {Path(args.metricsDir) / 'metrics.csv'}")
        else:
            predictions = runPredict(args)
            print(f"Predictions saved to {args.outputDir}: {', '.join(predictions)}")
    except (EOFError, KeyboardInterrupt):
        print("Input dibatalkan sebelum proses selesai.", file=sys.stderr)
        return 1
    except TypeError:
        print("Parameter tidak sesuai dengan model atau tahap pengolahan yang dipilih.", file=sys.stderr)
        return 1
    except (ValueError, OSError, NotImplementedError, KeyError) as error:
        print(f"Kesalahan: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
