import numpy as np

def halfnhalf(x):
    values = np.zeros(x.shape[1])
    # Set value 1 for the right half of the domain (x >= 0.5)
    values[x[0] >= 0.5] = 1.0
    return values
