import random

from dolfin import Expression, UserExpression


class HalfnhalfInitialConditions(UserExpression):
    def __init__(self, variables: int = 2, **kwargs):
        self.variables = variables
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.half(x)
        for i in range(1, self.variables):
            values[i] = 0.0

    def value_shape(self):
        return (self.variables,)

    def circles(self, x):
        if x[0] < 0.5:
            return 0.0
        else:
            return 1.0
