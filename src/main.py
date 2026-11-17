"""Parse CLI arguments and delegate execution to workflow entry points."""

import argparse
import sys

from .workflows import compare, evaluate, predict, train


def main():
    """Parse a planned command and report unfinished workflow execution."""
    parser = argparse.ArgumentParser(
        description="IF3170 supervised learning CLI (boilerplate; workflows pending)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    model_names = ("knn", "dtl", "logistic_regression", "svm", "naive_bayes")
    workflows = (
        ("train", "Train a from-scratch supervised learning model.", train.train),
        ("evaluate", "Evaluate a trained model on validation data.", evaluate.evaluate),
        ("compare", "Compare a model with its scikit-learn baseline.", compare.compare),
        ("predict", "Generate Kaggle predictions and a submission CSV.", predict.predict),
    )

    for name, description, workflow in workflows:
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument(
            "--model", required=True, choices=model_names, help="Algorithm to use."
        )
        if name == "train":
            command.add_argument("--k", type=int, help="Planned KNN neighbor count.")
            command.add_argument("--kernel", help="Planned SVM kernel, e.g. rbf.")
        elif name == "predict":
            command.add_argument(
                "--input", required=True, help="Path to the Kaggle test CSV."
            )
        command.set_defaults(workflow=workflow)

    args = parser.parse_args()
    try:
        args.workflow(args)
    except NotImplementedError:
        print(
            f"The '{args.command}' workflow is not yet implemented.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
