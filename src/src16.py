from matplotlib import pyplot as plt
import numpy as np


def dxdt(v):
    return v


def dvdt(x):
    return -x


def euler_method(x, v, dt):
    x_new = x + dt * dxdt(v)
    v_new = v + dt * dvdt(x)
    return x_new, v_new


tini = 0.0
tfin = 10.0
n = 32
dt = (tfin - tini) / n

x = 0.0
v = 1.0

time = np.zeros(n + 1)
xt = np.zeros(n + 1)
vt = np.zeros(n + 1)

time[0] = tini
xt[0] = x
vt[0] = v

for i in range(n):
    x, v = euler_method(x, v, dt)
    xt[i + 1] = x
    vt[i + 1] = v
    time[i + 1] = tini + (i + 1) * dt

plt.plot(time, xt, label="x(v)")
plt.plot(time, vt, label="v(t)")
plt.plot(time, np.sin(time), label="x(t): Exact", linestyle="dashed")
plt.plot(time, np.cos(time), label="v(t): Exact", linestyle="dashed")

plt.xlabel("t")
plt.ylabel("x, v")
plt.legend()
plt.savefig("result16.png")
