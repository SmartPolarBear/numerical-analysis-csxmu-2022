import numpy as np

from typing import Final


def preprocess_args(y: np.ndarray, h: np.ndarray, N: Final):
    alpha: np.ndarray = np.zeros(N + 1)
    c: np.ndarray = np.zeros(N + 1)
    dy = np.diff(y)

    ddyh = np.diff(dy / h)

    # by definition
    for j in range(1, N):
        alpha[j] = h[j - 1] / (h[j - 1] + h[j])
        c[j] = 6 * (1 / (h[j - 1] + h[j])) * (ddyh[j - 1])

    beta = np.ones(N + 1) - alpha

    return alpha, beta, c, dy
