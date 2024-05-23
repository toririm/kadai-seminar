from matplotlib import pyplot as plt
import numpy as np


def dydx(x, y):
    return -2.0 * x * y


def euler_method(y, x, h):
    return y + dydx(x, y) * h


def pred_corr_method(y, x, h):
    y_euler = y + dydx(x, y) * h
    y_pred_corr = y + 0.5 * h * (dydx(x, y) + dydx(x + h, y_euler))
    return y_pred_corr


a = 0.0
b = 3.0
n = 10

h = (b - a) / n

x = a
y = 1.0

x_j = np.zeros(n + 1)
x_j[0] = x

y_euler = np.zeros(n + 1)
y_euler[0] = y

y_pred_corr = np.zeros(n + 1)
y_pred_corr[0] = y

for i in range(n):
    x = a + i * h
    x_j[i + 1] = x + h
    y_euler[i + 1] = euler_method(y_euler[i], x, h)
    y_pred_corr[i + 1] = pred_corr_method(y_pred_corr[i], x, h)

plt.plot(x_j, y_pred_corr, label="Predictor-Corrector Method")
plt.plot(x_j, y_euler, label="Euler Method")
plt.plot(x_j, np.exp(-(x_j**2)), label="Exact", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
