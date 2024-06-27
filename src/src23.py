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
    v0 = 0.735
    sigma_v = 0.5
    return v0 * np.exp(-0.5 * (xj / sigma_v) ** 2)


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


# initial wavefunction parameters
x0 = -25.0
k0 = 0.85
sigma0 = 5.0

# time propagation parameters
Tprop = 80.0
dt = 0.005
nt = int(Tprop / dt) + 1

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
# initialize the potential
vpot = initialize_vpot(xj)


wavefunctions = []
for it in range(nt + 1):
    if it % (nt // 100) == 0:
        wavefunctions.append(wf.copy())

    wf = time_propagation(wf, vpot, dx, dt)
    print(it, nt)


# Define function to update plot for each frame of the animation
def update_plot(frame):
    plt.cla()
    plt.xlim([-100, 100])
    plt.ylim([-1.2, 1.2])
    plt.plot(xj, np.real(wavefunctions[frame]), label="Real part of $\psi(x)$")
    plt.plot(xj, np.imag(wavefunctions[frame]), label="Imaginary part of $\psi(x)$")
    plt.plot(xj, vpot, label="$V(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$\psi(x)$")
    plt.legend()


# Create the animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_plot, frames=len(wavefunctions), interval=50)
ani.save("src/src23.gif", writer="pillow")
