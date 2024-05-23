import numpy as np


def my_func(x):
    return 4.0 / (1.0 + x**2)


def integrate(f, a, b, n):
    h = (b - a) / n

    s = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x = a + h * i
        s += f(x)

    return s * h


a = 0.0
b = 1.0
n = 64

num_integral = integrate(my_func, a, b, n)

print("Numerical Integral: ", num_integral)
print("Analytical Integral: ", np.pi)
