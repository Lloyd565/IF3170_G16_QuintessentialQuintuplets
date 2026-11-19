import pickle
import numpy as np
from .kernels import compute_kernel

class SVM:
    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
                 tol=1e-3, max_passes=5, max_iter=100_000, random_state=0):
        self.C, self.kernel, self.degree = C, kernel, degree
        self.gamma, self.coef0, self.tol = gamma, coef0, tol
        self.max_passes, self.max_iter, self.random_state = max_passes, max_iter, random_state

    def _K(self, A, B):
        return compute_kernel(A, B, self.kernel, self.gamma_, self.coef0, self.degree)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM is binary; got %d classes" % len(self.classes_))
        t = np.where(y == self.classes_[1], 1.0, -1.0)
        self.gamma_ = (1.0 / (X.shape[1] * X.var() or 1.0)) if self.gamma == "scale" else float(self.gamma)

        n, C, tol = len(X), self.C, self.tol
        rng = np.random.default_rng(self.random_state)
        a, b = np.zeros(n), 0.0
        E = -t
        col = lambda i: self._K(X, X[i:i + 1])[:, 0]

        def step(i, j, Ki):
            nonlocal b
            if i == j:
                return False
            Kj = col(j)
            if t[i] != t[j]:
                L, H = max(0.0, a[j] - a[i]), min(C, C + a[j] - a[i])
            else:
                L, H = max(0.0, a[i] + a[j] - C), min(C, a[i] + a[j])
            eta = 2 * Ki[j] - Ki[i] - Kj[j]
            if L == H or eta >= 0:
                return False
            aj = np.clip(a[j] - t[j] * (E[i] - E[j]) / eta, L, H)
            if abs(aj - a[j]) < 1e-8:
                return False
            ai = a[i] + t[i] * t[j] * (a[j] - aj)
            di, dj = ai - a[i], aj - a[j]
            b1 = b - E[i] - t[i] * di * Ki[i] - t[j] * dj * Ki[j]
            b2 = b - E[j] - t[i] * di * Ki[j] - t[j] * dj * Kj[j]
            nb = b1 if 0 < ai < C else b2 if 0 < aj < C else (b1 + b2) / 2
            E[:] += t[i] * di * Ki + t[j] * dj * Kj + (nb - b)
            a[i], a[j], b = ai, aj, nb
            return True

        passes = it = 0
        while passes < self.max_passes and it < self.max_iter:
            changed = 0
            for i in range(n):
                if (t[i] * E[i] < -tol and a[i] < C) or (t[i] * E[i] > tol and a[i] > 0):
                    Ki = col(i)
                    js = [int(np.argmax(np.abs(E[i] - E)))]
                    js += list(np.roll(np.arange(n), -int(rng.integers(n))))
                    if any(step(i, j, Ki) for j in js):
                        changed += 1
                it += 1
                if it >= self.max_iter:
                    break
            passes = passes + 1 if changed == 0 else 0

        sv = a > 1e-8
        self.support_vectors_, self.dual_coef_ = X[sv], (a * t)[sv]
        self.support_, self.alpha_, self.b_ = np.flatnonzero(sv), a[sv], b
        return self

    def decision_function(self, X):
        return self._K(np.asarray(X, dtype=float), self.support_vectors_) @ self.dual_coef_ + self.b_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    @property
    def coef_(self):
        if self.kernel != "linear":
            raise AttributeError("coef_ only exists for the linear kernel")
        return self.dual_coef_ @ self.support_vectors_

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
