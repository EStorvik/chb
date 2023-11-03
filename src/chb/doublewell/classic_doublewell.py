class DoubleWellPotential:
    def __init__(self):
        pass

    def __call__(self, pf):
        return pf**2 * (1 - pf) ** 2

    def prime(self, pf):
        return 2 * pf * (1 - pf) * (1 - 2 * pf)

    def doubleprime(self, pf):
        return 2 * (1 - 6 * pf + 6 * pf**2)

    def c(self, pf):
        return (pf - 0.5) ** 4 + 0.0625

    def cprime(self, pf):
        return 4 * (pf - 0.5) ** 3

    def cdoubleprime(self, pf):
        return 12 * (pf - 0.5) ** 2

    def e(self, pf):
        return 0.5 * (pf - 0.5) ** 2

    def eprime(self, pf):
        return pf - 0.5

    def edoubleprime(self, pf):
        return 1
