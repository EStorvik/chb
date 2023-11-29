from dolfin import Identity


class Swelling:
    def __init__(
        self,
        swelling_parameter: float,
        dim: int = 2,
    ):
        self.swelling_parameter = swelling_parameter
        self.dim = dim

    def __call__(self, pf):
        return 2 * self.swelling_parameter * (pf - 0.5) * Identity(self.dim)

    def prime(self):
        return 2 * self.swelling_parameter * Identity(self.dim)
