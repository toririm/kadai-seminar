import numpy as np
from matplotlib import pyplot as plt

# Constants
mass = 1.0
hbar = 1.0
kconst = 1.0
omega = np.sqrt(kconst / mass)
length = 15.0

num_basis = 4

ham_mat = np.zeros((num_basis, num_basis))

for m in range(1, num_basis + 1):
    for n in range(1, num_basis + 1):
        if m == n:
            ham_mat[m - 1, n - 1] = (
                length**2 * kconst * (1 / 24.0 - 1 / (4 * n**2 * np.pi**2))
            )
        else:
            ham_mat[m - 1, n - 1] = (
                (-1) ** (m % 2 + (m + n) // 2)
                * length**2
                * kconst
                * 4
                * m
                * n
                / ((m - n) ** 2 * (m + n) ** 2 * np.pi**2)
            )

num_grid = 512
dx = length / (num_grid + 1)
xj = np.linspace(-length / 2 + dx, length / 2 - dx, num_grid)

eigenvalues, eigenvectors = np.linalg.eigh(ham_mat)

wf = np.zeros((num_grid, num_basis))

for m in range(num_basis):
    for n in range(num_basis):
        nb = n + 1
        if nb % 2 == 0:
            wf[:, m] += (
                np.sqrt(2.0 / length)
                * np.sin(nb * np.pi * xj[:] / length)
                * eigenvectors[n, m]
            )
        else:
            wf[:, m] += (
                np.sqrt(2.0 / length)
                * np.cos(nb * np.pi * xj[:] / length)
                * eigenvectors[n, m]
            )


def exact_eigenvalue(n):
    return hbar * np.sqrt(kconst / mass) * (n + 0.5)


for i in range(3):
    print(f"{i}-th eigenvalue = {eigenvalues[i]:.5f}")
    print(f"{i}-th eigenvalue Error = {(eigenvalues[i] - exact_eigenvalue(i)):.5f}")
    print()

plt.plot(xj, wf[:, 0], label="Ground state wf (calc.)")
plt.plot(
    xj,
    (mass * omega / (hbar * np.pi)) ** 0.25
    * np.exp(-0.5 * mass * omega * xj**2 / hbar),
    label="Ground state wf (exact)",
    linestyle="dashed",
)
plt.xlabel("x")
plt.ylabel("wf")
plt.legend()
plt.savefig("src32.png")
plt.show()
