import numpy as np


# Sumber: https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
#         https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

class NaiveBayes:
    def __init__(self, smoothing=1e-9):
        self.smoothing = smoothing
        self.classes = None
        self.prior_logs = None
        self.means = None
        self.variances = None
        self.used_cols = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X harus berupa matriks dua dimensi.")
        if len(X) != len(y):
            raise ValueError("Jumlah baris X dan y harus sama.")

        self.classes = np.unique(y)
        feature_count = X.shape[1] #kolom
        class_count = len(self.classes)

        self.prior_logs = np.zeros(class_count)
        self.means = np.zeros((class_count, feature_count))
        self.variances = np.zeros((class_count, feature_count))

        for i in range (class_count):
            rows = X[y == self.classes[i]]
            self.prior_logs[i] = np.log(len(rows) / len(X))
            self.means[i] = rows.mean(axis=0)
            self.variances[i] = rows.var(axis=0)

        largest_variance = X.var(axis=0).max()
        self.variances = self.variances + self.smoothing * largest_variance

        self.used_cols = np.arange(feature_count)
        return self

    def class_scores(self, X):
        if self.classes is None:
            raise ValueError("Model belum dilatih, panggil fit terlebih dahulu.")

        X = np.asarray(X, dtype=float)
        X = X[:, self.used_cols]
        means = self.means[:, self.used_cols]
        variances = self.variances[:, self.used_cols]

        # Turunan rumus log Gaussian (var = sigma^2, mu = mean kelas):
        #   p(x)     = 1 / sqrt(2*pi*var) * exp(-(x - mu)^2 / (2*var))
        #   log p(x) = log(1 / sqrt(2*pi*var)) + log(exp(-(x - mu)^2 / (2*var)))  
        #            = log((2*pi*var)^(-1/2)) - (x - mu)^2 / (2*var) 
        #            = -0.5 * log(2*pi*var) - (x - mu)^2 / (2*var)

        scores = np.zeros((len(X), len(self.classes)))
        for i in range (len(self.classes)):
            log_likelihood = -0.5 * np.log(2 * np.pi * variances[i]) - ((X - means[i]) ** 2) / (2 * variances[i])
            scores[:, i] = self.prior_logs[i] + log_likelihood.sum(axis=1)
        return scores

    def predict_proba(self, X):
        scores = self.class_scores(X)
        largest = scores.max(axis=1, keepdims=True)
        values = np.exp(scores - largest) # Mencegah largest score gabisa dihitung pake exp (pasti exp(0))
        total = values.sum(axis=1, keepdims=True)
        return values / total

    def predict(self, X):
        scores = self.class_scores(X)
        best = np.argmax(scores, axis=1)
        return self.classes[best]


def naive_bayes(X, y, smoothing=1e-9, pick_features=False):
    model = NaiveBayes(smoothing=smoothing)
    model.fit(X, y)

    if pick_features:
        from .bonus import select

        picked, round_scores = select.pick_features(X, y, smoothing=smoothing)
        model.used_cols = picked
        model.round_scores = round_scores

    return model
