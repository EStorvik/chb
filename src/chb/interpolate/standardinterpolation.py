class StandardInterpolator:
    def __init__(self):
        pass

    def __call__(self, phasefield):
        if phasefield < 0:
            out = 0
        elif phasefield > 1:
            out = 1
        else:
            out = -2 * phasefield**3 + 3 * phasefield**2
        return out

    def prime(self, phasefield):
        if phasefield < 0:
            out = 0
        elif phasefield > 1:
            out = 0
        else:
            out = -6 * phasefield**2 + 6 * phasefield
        return out

    def doubleprime(self, phasefield):
        if phasefield < 0:
            out = 0
        elif phasefield > 1:
            out = 0
        else:
            out = -12 * phasefield + 6
        return out
