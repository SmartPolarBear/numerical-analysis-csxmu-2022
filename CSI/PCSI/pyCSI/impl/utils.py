import numpy as np
import sympy as sp

from typing import Final


def calculate_coefficients(m:np.ndarray, y: np.ndarray, h: np.ndarray, N: Final):
    sixes: Final = np.full(N, 6)

    co1 = m[1:] / (sixes * h)
    co2 = m[:N] / (-sixes * h)
    co3 = y[1:] / h - (h * m[1:]) / sixes
    co4 = (h * m[:N]) / sixes - (y[:N] / h)

    return co1, co2, co3, co4


def create_function(co1: np.ndarray, co2: np.ndarray, co3: np.ndarray, co4: np.ndarray, x: np.ndarray, N: Final):
    formulas = []
    X = sp.Symbol('x')
    for j in range(0, N):
        formula = co1[j] * (X - x[j]) ** 3 \
                  + co2[j] * (X - x[j + 1]) ** 3 \
                  + co3[j] * (X - x[j]) \
                  + co4[j] * (X - x[j + 1])
        formula = sp.expand(formula)
        formula = sp.collect(formula, syms=x)
        formulas.append((formula, sp.And(X >= x[j], X < x[j + 1])))

    return X, sp.Piecewise(*formulas)


def create_matrix_A(r_top: np.ndarray, r_center: np.ndarray, r_bottom: np.ndarray):
    return np.diag(r_bottom, -1) + np.diag(r_center, 0) + np.diag(r_top, 1)
