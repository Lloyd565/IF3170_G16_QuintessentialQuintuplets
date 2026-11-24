#referensi yang digunakan : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html


def getBaseline(modelName, **parameters):
    from sklearn.linear_model import SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    baselines = {
        "knn": (KNeighborsClassifier, {"n_neighbors": 3, "metric": "euclidean"}),
        "dtl": (DecisionTreeClassifier, {"criterion": "entropy", "random_state": 0}),
        "logistic_regression": (SGDClassifier, {
            "loss": "log_loss", "penalty": None, "learning_rate": "constant",
            "eta0": 0.01, "max_iter": 100, "tol": None, "random_state": 42,
        }),
        "svm": (SVC, {"kernel": "rbf", "random_state": 42}),
        "naive_bayes": (GaussianNB, {}),
    }
    if modelName not in baselines:
        raise ValueError(f"Model acuan tidak tersedia: {modelName!r}.")
    constructor, defaults = baselines[modelName]
    options = {**defaults, **parameters}
    if modelName == "logistic_regression" and options["loss"] != "log_loss":
        raise ValueError("Acuan Logistic Regression harus menggunakan loss log_loss.")
    try:
        return constructor(**options)
    except TypeError as error:
        raise ValueError(f"Parameter model acuan {modelName!r} tidak sesuai.") from error


get_baseline = getBaseline
