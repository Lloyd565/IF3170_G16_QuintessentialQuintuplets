import numpy as np

def linear(A, B):
    K = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            K[i, j] = np.dot(A[i], B[j])
    return K

def polynomial(A, B, gamma, coef0, degree):
    K = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            K[i, j] = (gamma * np.dot(A[i], B[j]) + coef0)**degree
    return K

def gaussian(A, B, gamma):
    K = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            diff = A[i] - B[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return K

def compute_kernel(A, B, kernel="linear", gamma=1.0, coef0=0.0, degree=3):
    if kernel == "linear":
        return linear(A, B)
    if kernel == "poly":
        return polynomial(A, B, gamma, coef0, degree)
    if kernel == "rbf":
        return gaussian(A, B, gamma)
    raise ValueError(f"unknown kernel {kernel!r}; use 'linear', 'poly', 'rbf'")
