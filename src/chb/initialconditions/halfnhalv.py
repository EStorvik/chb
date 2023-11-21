import random

from dolfin import UserExpression, Expression


HalfnHalfInitialCondition = Expression(("x[0] < 0.5 ? 0.0 : 1.0", "0.0"), degree=1)
