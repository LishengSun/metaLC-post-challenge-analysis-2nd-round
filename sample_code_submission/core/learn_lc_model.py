import numpy as np
from scipy.optimize import least_squares


def power_law(x, a, b, c):
    return a - b * np.power(x, c)


def residual(p, x, y):
    return y - power_law(x, *p)


def learn_lc_model(x, y):
    x0 = np.random.rand(3,)
    result = least_squares(residual, x0, args=[x, y])
    a, b, c = result.x

    return lambda t: power_law(t, a, b, c)
