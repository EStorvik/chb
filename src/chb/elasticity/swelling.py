

class Swelling:
    def __init__(
        self,
        swelling_parameter: float,
    ):
        self.swelling_parameter = swelling_parameter

    def __call__(self, phasefield):
        return 2*self.swelling_parameter*(phasefield-0.5)

    def prime(self):
        return 2*self.swelling_parameter
