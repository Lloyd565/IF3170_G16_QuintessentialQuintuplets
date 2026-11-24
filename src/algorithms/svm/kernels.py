import numpy as np


def linear(A, B):
    return A @ B.T


def polynomial(A, B, gamma, coef0, degree):
    return (gamma * (A @ B.T) + coef0) ** degree


def gaussian(A, B, gamma):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * (A @ B.T)
    return np.exp(-gamma * np.maximum(sq, 0.0))


def compute_kernel(A, B, kernel="linear", gamma=1.0, coef0=0.0, degree=3):
    if kernel == "linear":
        return linear(A, B)
    if kernel == "poly":
        return polynomial(A, B, gamma, coef0, degree)
    if kernel == "rbf":
        return gaussian(A, B, gamma)
    raise ValueError(f"unknown kernel {kernel!r}; use 'linear', 'poly', 'rbf'")
