from matplotlib import pyplot
import numpy as np

x = np.linspace(0.0, 10.0, 64)
y = np.cos(x)

pyplot.plot(x, y)
pyplot.show()
