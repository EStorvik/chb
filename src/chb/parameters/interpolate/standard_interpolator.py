import ufl


class StandardInterpolator:
    def __init__(self):
        pass

    def __call__(self, pf):
        return ufl.conditional(
            ufl.le(pf, 0),
            0,
            ufl.conditional(
                ufl.ge(pf, 1), 1, -2 * pf**3 + 3 * pf**2
            ),
        )

    def prime(self, pf):
        return ufl.conditional(
            ufl.le(pf, 0),
            0,
            ufl.conditional(
                ufl.ge(pf, 1), 0, -6 * pf**2 + 6 * pf
            ),
        )

    def doubleprime(self, pf):
        return ufl.conditional(
            ufl.le(pf, 0),
            0,
            ufl.conditional(ufl.ge(pf, 1), 0, -12 * pf + 6),
        )

class SymmetricStandardInterpolator:
    def __init__(self):
        pass

    def __call__(self, pf):
        return ufl.conditional(
            ufl.le(pf, -1),
            0,
            ufl.conditional(
                ufl.ge(pf, 1), 1, 0.25 *(- pf**3 + 3 * pf + 2)
            ),
        )

    def prime(self, pf):
        return ufl.conditional(
            ufl.le(pf, -1),
            0,
            ufl.conditional(
                ufl.ge(pf, 1), 0, 0.25 * (-3*pf**2 + 3)
            ),
        )

    def doubleprime(self, pf):
        return ufl.conditional(
            ufl.le(pf, -1),
            0,
            ufl.conditional(ufl.ge(pf, 1), 0, 0.25* (-6) * pf),
        )
