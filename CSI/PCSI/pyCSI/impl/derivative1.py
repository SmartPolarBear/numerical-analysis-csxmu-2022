import numpy as np
import sympy as sp

from typing import Final

from pyCSI.impl.preprocess import preprocess_args
from pyCSI.impl.thomas import thomas_solve
from pyCSI.impl.utils import create_function, create_matrix_A, calculate_coefficients


def spline_impl_derive1(x: np.ndarray, y: np.ndarray, h: np.ndarray, m0: np.float, mn: np.float):
    N: Final = x.shape[0] - 1

    alpha, beta, c, dy = preprocess_args(y, h, N)

    # by constraint
    alpha[N] = 1
    beta[0] = 1
    c[0] = (6.0 / h[0]) * (dy[0] / h[0] - m0)
    c[N] = (6.0 / h[N - 1]) * (dy[N - 1] / h[N - 1] + mn)

    # a = create_matrix_A(beta[0:N], 2 * np.ones(N + 1), alpha[1:N + 1])
    #
    # M: Final = np.transpose(np.matrix(a).I.dot(np.transpose(np.matrix(c))))

    M: Final = thomas_solve(alpha[1:N + 1], 2 * np.ones(N + 1), beta[0:N], c)

    return calculate_coefficients(M, y, h, N)
