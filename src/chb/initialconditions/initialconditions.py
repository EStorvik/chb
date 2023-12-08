import random

from dolfin import UserExpression


class InitialConditions(UserExpression):
    def __init__(self, variables: int = 2, **kwargs):
        self.variables = variables
        super().__init__(**kwargs)

    def eval(self, values, x):
        pass

    def value_shape(self):
        return (self.variables,)

    def half(self, x):
        pass