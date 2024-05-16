from matplotlib import pyplot as plt
import numpy as np

x = 1.0
step_sizes = np.array([10**(-i) for i in range(15)])

forward_diff = (np.exp(x + step_sizes) - np.exp(x)) / step_sizes
central_diff = (np.exp(x + step_sizes) - np.exp(x - step_sizes)) / (2 * step_sizes)

error_forward = np.abs(forward_diff - np.exp(x))
error_central = np.abs(central_diff - np.exp(x))

plt.plot(step_sizes, error_forward, label="Forward Difference", marker="o")
plt.plot(step_sizes, error_central, label="Central Difference", marker="x")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Step Size (h)")
plt.ylabel("Error")
plt.title("Error in Numerical Differentiation of exp(x) at x = 1.0")
plt.legend()
plt.grid(True)

plt.show()
