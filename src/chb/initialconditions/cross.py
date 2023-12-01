
from dolfin import UserExpression


class CrossInitialConditions(UserExpression):
    def __init__(self, delta: float = 0.15, variables: int = 2, **kwargs):
        self.variables = variables
        self.delta = delta
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.cross(x)
        for i in range(1, self.variables):
            values[i] = 0.0

    def value_shape(self):
        return (self.variables,)

    def cross(self, x):
        if (
            (x[0] < (0.5 + self.delta) and x[0] > (0.5 - self.delta))
            and (x[1] < (0.5 + 2 * self.delta) and x[1] > (0.5 - 2 * self.delta))
        ) or (
            (x[1] < (0.5 + self.delta) and x[1] > (0.5 - self.delta))
            and (x[0] < (0.5 + 2 * self.delta) and x[0] > (0.5 - 2 * self.delta))
        ):
            return 1.0
        else:
            return 0.0


# CrossInitialCondition = Expression(
#     (
#         "((x[0]<(0.5+delta) && x[0]>(0.5-delta)) && (x[1]<(0.5+2*delta) && x[1]>(0.5-2*delta))) || ((x[1]<(0.5+delta) && x[1]>(0.5-delta)) && (x[0]<(0.5+2*delta) && x[0]>(0.5-2*delta))) ? 1.0 : 0.0",
#         "0.0",
#     ),
#     degree=0,
#     delta=0.15,
# )
