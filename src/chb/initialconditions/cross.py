import numpy as np


class Cross:
    def __init__(self, width=0.1):
        self.width = width

    def __call__(self, x):
        values = np.zeros(x.shape[1])
        # Define the width of the cross
        cross_width = self.width
        # Set value 1 for the cross embedded inside the unit square
        values[
            np.logical_or(
                np.logical_and(
                    (np.abs(x[0] - 0.5) <= cross_width / 2),
                    (np.abs(x[1] - 0.5) <= cross_width),
                ),
                np.logical_and(
                    (np.abs(x[1] - 0.5) <= cross_width / 2),
                    (np.abs(x[0] - 0.5) <= cross_width),
                ),
            )
        ] = 1.0
        return values
