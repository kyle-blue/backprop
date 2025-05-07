import matplotlib.pyplot as plt
import numpy as np


def f(x: np.ndarray):
    return x**3 + 3 * x**2 + 8 * x + 1


xs = np.arange(-10, 10, 0.25)
ys = f(xs)

fig, ax = plt.subplots()
ax.plot(xs, ys)

plt.show()
