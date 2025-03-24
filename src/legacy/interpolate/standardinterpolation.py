import ufl_legacy as ufl


class StandardInterpolator:
    def __init__(self):
        pass

    def __call__(self, phasefield):
        return ufl.conditional(
            ufl.le(phasefield, 0),
            0,
            ufl.conditional(
                ufl.ge(phasefield, 1), 1, -2 * phasefield**3 + 3 * phasefield**2
            ),
        )

    def prime(self, phasefield):
        return ufl.conditional(
            ufl.le(phasefield, 0),
            0,
            ufl.conditional(
                ufl.ge(phasefield, 1), 0, -6 * phasefield**2 + 6 * phasefield
            ),
        )

    def doubleprime(self, phasefield):
        return ufl.conditional(
            ufl.le(phasefield, 0),
            0,
            ufl.conditional(ufl.ge(phasefield, 1), 0, -12 * phasefield + 6),
        )
