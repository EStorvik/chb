import chb


class NonlinearBiotCoupling:
    def __init__(
        self,
        alpha0: float,
        alpha1: float,
        interpolator: chb.StandardInterpolator = chb.StandardInterpolator(),
    ):
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.interpolator = interpolator

    def __call__(self, phasefield):
        return self.alpha0 + self.interpolator(phasefield) * (self.alpha1 - self.alpha0)

    def prime(self, phasefield):
        return self.interpolator.prime(phasefield) * (self.alpha1 - self.alpha0)

    def doubleprime(self, phasefield):
        return self.interpolator.doubleprime(phasefield) * (self.alpha1 - self.alpha0)
