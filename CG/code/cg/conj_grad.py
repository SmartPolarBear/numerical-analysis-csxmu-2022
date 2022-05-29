import numpy as np


class Equation:
    def __init__(self, coeff, vals):
        self.a = np.asmatrix(coeff, dtype=np.float)
        self.b = np.asarray(vals, dtype=np.float)
        self.solve = self._solve_cg
        assert self.a.shape[1] == self.b.shape[0]

    def use_naive(self):
        self.solve = self._solve_naive
        return self

    def use_conjugate_gradient(self):
        self.solve = self._solve_cg
        return self

    def _solve_naive(self, x0=None, max_iter=512):
        x = x0
        if x0 is None:
            x = np.zeros_like(self.b, dtype=np.float)

        n = self.a.shape[0]
        iter = 0
        prev_x = x.copy()
        while iter < max_iter:
            for j in range(0, n):
                d = self.b[j]

                for i in range(0, n):
                    if j != i:
                        d -= self.a[j, i] * x[i]
                x[j] = d / self.a[j, j]

            if np.allclose(x, prev_x, rtol=1e-5, atol=1e-5, equal_nan=True):
                return x
            prev_x = x.copy()
            iter += 1

        return x

    def _solve_cg(self, x0=None, max_iter=512):
        if x0 is None:
            x0 = np.zeros_like(self.b, dtype=np.float)

        a = np.asmatrix(self.a, dtype=np.float)
        b = np.asmatrix(self.b, dtype=np.float).transpose()
        x = np.asmatrix(x0, dtype=np.float).transpose()

        r = b - a * x
        p = r

        zeros = np.zeros_like(x)

        iter = 0
        while iter <= max_iter:
            if np.allclose(r, zeros):
                break

            ap = a * p

            if np.allclose(ap, zeros):
                break

            alpha = ((r.transpose() * r) / (p.transpose() * ap))[0, 0]

            x += alpha * p

            r1 = r - alpha * ap

            beta = ((r1.transpose() * r1) / (r.transpose() * r))[0, 0]

            p = r1 + beta * p

            r = r1
            iter += 1

        return x

    # def _solve_cg(self, x0=None, max_iter=512):
    #     if x0 is None:
    #         x0 = np.zeros_like(self.b, dtype=np.float)
    #
    #     b = np.asarray(self.b, dtype=np.float)
    #     a = self.a
    #
    #     x0 = np.asarray(x0, dtype=np.float)
    #     x = x0
    #
    #     assert x.shape == b.shape
    #
    #     r = np.asarray(b - a.dot(x), dtype=np.float).ravel()
    #     p = r
    #
    #     iter = 0
    #     while np.any(r) and iter <= max_iter:
    #         ap = np.asarray(a.dot(p), dtype=np.float).ravel()
    #         if not np.any(ap):
    #             break
    #
    #         alpha = r.dot(r) / p.dot(ap)
    #         x += alpha * p
    #         r1 = r - alpha * ap
    #
    #         beta = r1.dot(r1) / r.dot(r)
    #         p = r1 + beta * p
    #
    #         r = r1
    #         iter += 1
    #
    #     return x
