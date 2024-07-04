import time
from numba import jit
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation

start_time = time.time()


# initialize the wavefunction
def initialize_wf(xj, x0, k0, sigma0):
    wf = np.exp(1j * k0 * (xj - x0)) * np.exp(-0.5 * (xj - x0) ** 2 / sigma0**2)
    return wf


# Initialize potential
def initialize_vpot(xj):
    k0 = 1.0
    return 0.5 * k0 * xj**2


# Operate the Hamiltonian on the wavefunction
@jit(nopython=True)
def ham_wf(wf, vpot, dx):
    n = wf.size
    hwf = np.zeros(n, dtype=np.complex128)

    for i in range(1, n - 1):
        hwf[i] = -0.5 * (wf[i + 1] - 2.0 * wf[i] + wf[i - 1]) / dx**2

    i = 0
    hwf[i] = -0.5 * (wf[i + 1] - 2.0 * wf[i]) / dx**2

    i = n - 1
    hwf[i] = -0.5 * (-2.0 * wf[i] + wf[i - 1]) / dx**2

    hwf = hwf + vpot * wf

    return hwf


# Time propagation from t to t + dt
def time_propagation(wf, vpot, dx, dt):
    n = wf.size
    twf = np.zeros(n, dtype=complex)
    hwf = np.zeros(n, dtype=complex)

    twf = wf
    zfact = 1.0 + 0j

    for iexp in range(1, 5):
        zfact = zfact * (-1j * dt) / iexp
        hwf = ham_wf(twf, vpot, dx)
        wf = wf + zfact * hwf
        twf = hwf

    return wf

def calc_expectation_values(wf, xj):

    dx = xj[1] - xj[0]
    norm = np.sum(np.abs(wf) ** 2) * dx
    x_exp = np.sum(xj * np.abs(wf) ** 2) * dx / norm

    n = wf.size
    pwf = np.zeros(n, dtype=complex)

    for i in range(1, n - 1):
        pwf[i] = -1j * (wf[i + 1] - wf[i - 1]) / (2.0 * dx)

    p_exp = np.real(np.sum(np.conjugate(wf) * pwf) * dx) / norm
    k_exp = np.real(np.sum(np.conjugate(pwf) * pwf / 2.0) * dx) / norm

    return x_exp, p_exp, norm, k_exp

# initial wavefunction parameters
x0 = -2.0
k0 = 0.0
sigma0 = 1.0

# time propagation parameters
Tprop = 40.0
dt = 0.005
nt = int(Tprop / dt) + 1

# set the coordinate
xmin = -10.0
xmax = 10.0
n = 250

dx = (xmax - xmin) / (n + 1)
xj = np.zeros(n)

for i in range(n):
    xj[i] = xmin + (i + 1) * dx

# initialize the wavefunction
wf = initialize_wf(xj, x0, k0, sigma0)
# initialize the potential
vpot = initialize_vpot(xj)


tt = np.zeros(nt + 1)
xt = np.zeros(nt + 1)
pt = np.zeros(nt + 1)
normt = np.zeros(nt + 1)
kt = np.zeros(nt + 1)


wavefunctions = []
for it in range(nt + 1):
    if it % (nt // 100) == 0:
        wavefunctions.append(wf.copy())

    tt[it] = dt * it
    xt[it], pt[it], normt[it], kt[it] = calc_expectation_values(wf, xj)
    wf = time_propagation(wf, vpot, dx, dt)
    print(it, nt)


plt.plot(tt, xt, label="$<x>$")
plt.plot(tt, pt, label="$<p>$")
plt.plot(tt, normt, label="Norm")
plt.plot(tt, kt, label="$<T>$")
plt.xlabel("$t$")
plt.ylabel('Quantities')
plt.legend()

plt.savefig("src/expectation_values.pdf")

# Define function to update plot for each frame of the animation
# def update_plot(frame):
#     plt.cla()
#     plt.xlim([-5, 5])
#     plt.ylim([-1.2, 5.0])
#     plt.plot(xj, np.real(wavefunctions[frame]), label="Real part of $\psi(x)$")
#     plt.plot(xj, np.imag(wavefunctions[frame]), label="Imaginary part of $\psi(x)$")
#     plt.plot(xj, np.abs(wavefunctions[frame]), label="$|\psi(x)|$")
#     plt.plot(xj, vpot, label="$V(x)$")
#     plt.xlabel("$x$")
#     plt.ylabel("$\psi(x)$")
#     plt.legend()


# # Create the animation
# fig = plt.figure()
# ani = animation.FuncAnimation(fig, update_plot, frames=len(wavefunctions), interval=150)
# ani.save("src/src24.gif", writer="pillow")
