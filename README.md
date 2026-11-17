# IF3170_G16_QuintessentialQuintuplets

Tugas Besar 2 AI: Supervised Learning (IF3170). This project will implement KNN,
Decision Tree Learning (DTL), Logistic Regression, SVM, and Naive Bayes from
scratch, with shared preprocessing, evaluation, visualization, and Kaggle workflows.

This repository currently contains architecture and boilerplate only. Algorithm,
preprocessing, evaluation, visualization, and workflow functions deliberately raise
`NotImplementedError`. The CLI recognizes planned commands and reports that they
are not yet implemented, exiting with a nonzero status.

## Repository structure

```text
.
├── data/                              # Local Kaggle datasets (ignored)
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── main.py                        # Argument parsing and workflow delegation
│   ├── core/                          # Shared infrastructure only
│   │   ├── __init__.py
│   │   ├── base_model.py               # Optional interface documentation
│   │   ├── config.py                   # Project paths
│   │   ├── registry.py                 # Future algorithm selection
│   │   └── serialization.py            # Future artifact saving/loading
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── knn/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── distance.py
│   │   │   └── bonus/
│   │   │       └── __init__.py
│   │   ├── dtl/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── criteria.py
│   │   │   └── bonus/
│   │   │       └── __init__.py
│   │   ├── logistic_regression/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── bonus/
│   │   │       └── __init__.py
│   │   ├── svm/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── kernels.py
│   │   │   └── bonus/
│   │   │       └── __init__.py
│   │   ├── naive_bayes/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── bonus/
│   │   │       └── __init__.py
│   │   └── sklearn_baselines.py        # Separate scikit-learn comparisons
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # Future stage coordination
│   │   ├── cleaning/
│   │   │   ├── __init__.py
│   │   │   ├── cleaning.py
│   │   │   ├── missing_values.py
│   │   │   ├── duplicates.py
│   │   │   └── validation.py
│   │   ├── transformation/
│   │   │   ├── __init__.py
│   │   │   ├── transformation.py
│   │   │   ├── encoding.py
│   │   │   ├── scaling.py
│   │   │   └── imbalance.py
│   │   ├── feature_selection/
│   │   │   ├── __init__.py
│   │   │   ├── feature_selection.py
│   │   │   └── selection.py
│   │   ├── dimensionality_reduction/
│   │   │   ├── __init__.py
│   │   │   ├── dimensionality_reduction.py
│   │   │   └── pca.py
│   │   └── visualization/
│   │       ├── __init__.py
│   │       └── metrics.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── compare.py
│   │   └── predict.py
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
├── notebooks/
│   └── submission.ipynb               # Report and submission outline
├── output/                            # Generated artifacts (ignored)
│   ├── models/.gitkeep                # Models and fitted preprocessing artifacts
│   ├── media/.gitkeep                 # Images, plots, visualizations, videos
│   ├── metrics/.gitkeep               # Evaluation results and experiment records
│   └── submissions/.gitkeep           # Kaggle submission CSV files
├── docs/.gitkeep
├── tests/.gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment setup

Use Python 3.10 or newer. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependencies cover numerical processing (`numpy`), tabular data (`pandas`),
reference models (`scikit-learn`), plots (`matplotlib`), and notebook execution
(`notebook`, `ipykernel`). `argparse` is part of Python's standard library.

Place the competition's `train.csv`, `test.csv`, and `sample_submission.csv` in
`data/`. These files and generated outputs are ignored; `.gitkeep` files remain
trackable. Start the notebook with `python -m notebook` from the repository root.
Future notebook code should import functionality from `src/`; ensure the repository
root is on the kernel's import path instead of copying implementations into cells.

## Planned CLI usage

```bash
python -m src.main --help
python -m src.main train --help
python -m src.main train --model knn --k 5
python -m src.main train --model svm --kernel rbf
python -m src.main evaluate --model knn
python -m src.main compare --model knn
python -m src.main predict --model knn --input data/test.csv
```

Model names are `knn`, `dtl`, `logistic_regression`, `svm`, and `naive_bayes`.
`--k` and `--kernel` are optional training arguments reserved for future
implementations. Help commands work now; execution commands report an unfinished
workflow and exit with status 1. No models, scores, or predictions are produced.

## Contributor guidelines

- Keep each algorithm's implementation inside its own `src/algorithms/` package.
  Each package currently exports a function named after the algorithm. These are
  placeholders; owners may choose classes, functions, parameters, and internal
  representations without mandatory inheritance. `core/base_model.py` only
  documents a possible future interface. Keep scikit-learn reference models in
  `sklearn_baselines.py`, separate from the from-scratch implementations.
  Keep algorithm-specific bonus visualizations and extensions in that algorithm's
  `bonus/` package; each currently provides a `visualize_bonus()` placeholder.
- Keep preprocessing internals within their stage's folder. Each stage exposes
  one main function through its `__init__.py`: `cleaning()`, `transformation()`,
  `feature_selection()`, or `dimensionality_reduction()`. Callers should use those
  package entry points, for example `from src.processing.cleaning import cleaning`.
  Supporting functions and all placeholder signatures remain open to revision.
- `processing/pipeline.py` will coordinate cleaning, transformation, feature
  selection, then dimensionality reduction, allowing stages to be disabled.
  Separate training and validation before fitting learned preprocessing parameters;
  fit on training data only and reuse those parameters for validation and Kaggle
  test data. Support training and prediction modes, and keep targets aligned when
  operations change row counts.
- Workflows will coordinate preprocessing, model selection, training, evaluation,
  and artifact output. Keep algorithms independent of dataset paths, CLI arguments,
  and output directories. Keep `core/` limited to shared infrastructure and
  `main.py` limited to parsing and delegation. Use package imports consistently
  and run the CLI as `python -m src.main` from the repository root.
- Put scoring logic in `evaluation/`, shared metric plotting in
  `processing/visualization/`, documentation in `docs/`, and future tests in
  `tests/`. Replace placeholders with real behavior when implementing a module;
  do not return fabricated results.

## Kaggle submission format

Use the competition-provided `data/sample_submission.csv` as the authoritative
template. Future submissions must match its exact column names and column order,
include the expected test identifiers in the expected row order, and provide one
prediction per test example. Use the target representation required by the
competition (labels or probabilities), omit any extra DataFrame index column, and
write the CSV to `output/submissions/`. Confirm the exact schema against the
competition instructions; no dataset-specific column names are assumed yet.

## Group member responsibilities

Assign module ownership before implementation:

| Member | Algorithm | Preprocessing / shared responsibilities |
| --- | --- | --- |
| TBD | KNN | TBD |
| TBD | DTL | TBD |
| TBD | Logistic Regression | TBD |
| TBD | SVM | TBD |
| TBD | Naive Bayes | TBD |
