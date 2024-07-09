import numpy as np

matrix = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("First eigenvalue:", eigenvalues[0])
print("Second eigenvalue:", eigenvalues[1])
print("Third eigenvalue:", eigenvalues[2])
print()

print("First eigenvector:", eigenvectors[:, 0])
print("Second eigenvector:", eigenvectors[:, 1])
print("Third eigenvector:", eigenvectors[:, 2])
