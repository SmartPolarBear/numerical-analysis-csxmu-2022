import numpy as np
import sympy as sp

from typing import Final

from pyCSI.impl.preprocess import preprocess_args
from pyCSI.impl.thomas import thomas_solve
from pyCSI.impl.utils import create_function, create_matrix_A, calculate_coefficients


def spline_impl_not_a_knot(x: np.ndarray, y: np.ndarray, h: np.ndarray):
    N: Final = x.shape[0] - 1

    alpha, beta, c, dy = preprocess_args(y, h, N)

    # by constraint
    alpha[N] = -2
    beta[0] = -2
    c[0] = 0
    c[N] = 0

    M: Final = thomas_solve(alpha[1:N + 1], 2 * np.ones(N + 1), beta[0:N], c)

    return calculate_coefficients(M, y, h, N)
