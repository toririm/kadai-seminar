import numpy as np
from matplotlib import pyplot as plt

DX = 1.0
NX = 52
X_START = 0.0
X_END = 7.0

x = np.linspace(X_START, X_END, NX)

fx = np.cos(x)
fx_pdx = np.cos(x + DX)
fx_mdx = np.cos(x - DX)

d2fdx2 = (fx_pdx - 2 * fx + fx_mdx) / DX**2

plt.plot(x, fx, label="f(x) = cos(x)")
plt.plot(x, d2fdx2, label="f''(x)")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Second Derivative of cos(x)")
plt.legend()

plt.show()
