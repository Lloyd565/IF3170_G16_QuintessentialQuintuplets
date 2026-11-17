"""Placeholder for preprocessing coordination without a fitting framework.

The eventual stage order is cleaning, transformation, feature selection, then
dimensionality reduction. Some stages may be disabled per experiment or algorithm.

Separate training and validation data before fitting any learned preprocessing
parameters. Learn parameters from training data only; validation and Kaggle test
data must reuse the transformations fitted on training data. Support both
training-time and prediction-time processing. Operations that change row counts
must keep the corresponding target labels aligned.
"""


def run_preprocessing(*args, **kwargs):
    """Coordinate optional preprocessing stages; the interface remains undecided."""
    raise NotImplementedError
