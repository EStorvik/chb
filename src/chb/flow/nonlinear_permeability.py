import chb


class NonlinearPermeability:
    def __init__(
        self,
        k0: float,
        k1: float,
        interpolator: chb.StandardInterpolator = chb.StandardInterpolator(),
    ):
        self.k0 = k0
        self.k1 = k1
        self.interpolator = interpolator

    def __call__(self, phasefield):
        return self.k0 + self.interpolator(phasefield) * (self.k1 - self.k0)

    def prime(self, phasefield):
        return self.interpolator.prime(phasefield) * (self.k1 - self.k0)

    def doubleprime(self, phasefield):
        return self.interpolator.doubleprime(phasefield) * (self.k1 - self.k0)
