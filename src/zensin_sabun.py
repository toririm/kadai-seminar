import numpy as np


def derivative(x, h):
    fx = np.exp(x)
    fxph = np.exp(x + h)
    num_dfdx = (fxph - fx) / h
    ana_dfdx = np.exp(x)
    error = np.abs(num_dfdx - ana_dfdx)
    return ana_dfdx, num_dfdx, error


print("Derivative of exp(x) at x = 1.0 using finite difference method:")
ana_dfdx, _, _ = derivative(1.0, 0.1)
print(f"Analytical df/dx = {ana_dfdx}")

for i in range(1, 10):
    h = 10 ** (-i)
    _, num_dfdx, error = derivative(1.0, h)
    print("----------------")
    print(f"For h = {h}")
    print(f"Numerical df/dx = {num_dfdx}")
    print(f"Error = {error}")
