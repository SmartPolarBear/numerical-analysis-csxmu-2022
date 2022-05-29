import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import seaborn as sns

import timeit

import pyCSI.spline as sl


def test_func(x):
    return 1 / (1 + x ** 2)

counts = [i for i in range(10, 101)]
times = []

for i in counts:
    x = np.linspace(start=-5, stop=5, num=i)
    y = np.ones(x.shape[0]) / (np.ones(x.shape[0]) + x ** 2)

    start = timeit.default_timer()
    for j in range(0, 100):
        sl.spline(x, y, sl.ConstraintType.NOT_A_KNOT, False)
    end = timeit.default_timer()

    times.append((end - start) / 100.0)

average = sum(times) / len(times)
sns.lineplot(x=counts, y=times)
plt.xlabel("n")
plt.ylabel("t(s)")
plt.show()

plt.clf()

data = [average, 1.466272222222222e-04]
labels = ['pyCSI', 'Matlab']

plt.bar(range(len(data)), data, tick_label=labels, width=0.2, color=["violet", "orange"])
plt.ylabel("t(s)")
plt.show()

print(((average - 1.466272222222222e-04) / 1.466272222222222e-04)*100)

x = [i for i in range(-5, 6)]
y = [test_func(i) for i in x]

X, spfunc1 = sl.spline(x, y, sl.ConstraintType.NOT_A_KNOT)

print(spfunc1)
sp.plot(spfunc1, (X, -5, 5), backend='matplotlib')
