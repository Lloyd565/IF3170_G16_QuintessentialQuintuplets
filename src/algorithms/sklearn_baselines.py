"""Scikit-learn reference models, separate from from-scratch implementations."""


def get_baseline(model_name, **parameters):
    """Return an unfitted reference model; currently only CART DTL is available.

    Encode categorical columns and fit missing-value preprocessing on the
    training partition before using the scikit-learn counterpart.
    """
    if model_name != "dtl":
        raise NotImplementedError(f"The {model_name!r} baseline is not implemented.")
    from sklearn.tree import DecisionTreeClassifier

    options = {"criterion": "gini", "random_state": 0}
    options.update(parameters)
    return DecisionTreeClassifier(**options)
