from matplotlib import pyplot as plt
import numpy as np


# initialize the wavefunction
def initialize_wf(xj, x0, k0, sigma0):
    wf = np.exp(1j * k0 * (xj - x0)) * np.exp(-0.5 * (xj - x0) ** 2 / sigma0**2)
    return wf


# Operate the Hamiltonian on the wavefunction
def ham_wf(wf, vpot, dx):
    n = wf.size
    hwf = np.zeros(n, dtype=complex)

    for i in range(1, n - 1):
        hwf[i] = -0.5 * (wf[i + 1] - 2.0 * wf[i] + wf[i - 1]) / dx**2

    i = 0
    hwf[i] = -0.5 * (wf[i + 1] - 2.0 * wf[i]) / dx**2

    i = n - 1
    hwf[i] = -0.5 * (-2.0 * wf[i] + wf[i - 1]) / dx**2

    hwf = hwf + vpot * wf

    return hwf


# initial wavefunction parameters
x0 = -25.0
k0 = 0.85
sigma0 = 5.0

# set the coordinate
xmin = -100.0
xmax = 100.0
n = 2500

dx = (xmax - xmin) / (n + 1)
xj = np.zeros(n)

for i in range(n):
    xj[i] = xmin + (i + 1) * dx

# initialize the wavefunction
wf = initialize_wf(xj, x0, k0, sigma0)
vpot = np.zeros(n)

hwf = ham_wf(wf, vpot, dx)

# plot the wavefunction
plt.plot(xj, np.real(wf), label="Real part (wf)")
plt.plot(xj, np.imag(wf), label="Imaginary part (wf)")
# plt.plot(xj, np.abs(wf)**2, label="Probability Density")
plt.plot(xj, np.real(hwf), label="Real part (hwf)")
plt.plot(xj, np.imag(hwf), label="Imaginary part (hwf)")


plt.xlabel("x")
plt.ylabel("$\psi$(x)")
plt.legend()
plt.show()
