"""Scikit-learn reference models, separate from from-scratch implementations."""


def get_baseline(model_name, **parameters):
    """Return a reference model for prepared numeric features."""
    if model_name != "dtl":
        raise NotImplementedError(f"The {model_name!r} baseline is not implemented.")
    from sklearn.tree import DecisionTreeClassifier

    options = {"criterion": "entropy", "random_state": 0}
    options.update(parameters)
    return DecisionTreeClassifier(**options)
