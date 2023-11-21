class UnboundedInterpolator:
    def __init__(self):
        pass

    def __call__(self, phasefield):
        return -2 * phasefield**3 + 3 * phasefield**2

    def prime(self, phasefield):
        return -6 * phasefield**2 + 6 * phasefield

    def doubleprime(self, phasefield):
        return -12 * phasefield + 6
