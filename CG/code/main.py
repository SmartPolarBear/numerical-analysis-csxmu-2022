import random

from cg.conj_grad import Equation

eq = Equation([[1, 1, 1], [1, 2, 3], [1, 3, 3]], [4, 5, 6])
print(eq.use_conjugate_gradient().solve())
