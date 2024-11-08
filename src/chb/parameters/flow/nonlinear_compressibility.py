import chb


class NonlinearCompressibility:
    def __init__(
        self,
        M0: float,
        M1: float,
        interpolator: chb.interpolate.StandardInterpolator = chb.interpolate.StandardInterpolator(),
    ):
        self.M0 = M0
        self.M1 = M1
        self.interpolator = interpolator

    def __call__(self, phasefield):
        return self.M0 + self.interpolator(phasefield) * (self.M1 - self.M0)

    def prime(self, phasefield):
        return self.interpolator.prime(phasefield) * (self.M1 - self.M0)

    def doubleprime(self, phasefield):
        return self.interpolator.doubleprime(phasefield) * (self.M1 - self.M0)
