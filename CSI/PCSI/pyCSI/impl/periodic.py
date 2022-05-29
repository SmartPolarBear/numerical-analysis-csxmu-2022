import numpy as np
import sympy as sp

from typing import Final

from pyCSI.impl.preprocess import preprocess_args
from pyCSI.impl.utils import create_function, create_matrix_A, calculate_coefficients


def spline_impl_periodic(x: np.ndarray, y: np.ndarray, h: np.ndarray):
    N: Final = x.shape[0] - 1

    alpha, beta, c, dy = preprocess_args(y, h, N)

    # by constraint
    alpha[N] = h[N - 1] / (h[0] + h[N - 1])
    beta[0] = 1 - alpha[N]
    c[0] = (6.0 / (h[0] + h[N - 1])) * (dy[0] / h[0] - dy[N - 1] / h[N - 1])
    c[N] = (6.0 / (h[0] + h[N - 1])) * (dy[0] / h[0] - dy[N - 1] / h[N - 1])

    a = create_matrix_A(beta[0:N], 2 * np.ones(N + 1), alpha[1:N + 1])

    # Cannot use Thomas's algorithm
    a[N, N - 1] = a[0, N - 1] = alpha[N]
    a[0, 1] = a[N, 1] = beta[N]
    M: Final = np.transpose(np.matrix(a).I.dot(np.transpose(np.matrix(c))))

    return calculate_coefficients(np.asarray(M.A1), y, h, N)
