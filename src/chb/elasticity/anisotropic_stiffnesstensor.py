# Make anisotropic stiffness tensor here.


import dolfin as df
from dolfin import sym, grad, dx
from ufl_legacy import indices
import numpy as np

# Create four free indices for defining the scalar product
i, j, k, l = indices(4)

# Fourth-order stiffness tensor
C = df.Constant(np.ones((3, 3, 3, 3)))

# Variational form
mesh = df.UnitCubeMesh(4, 4, 4)
V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
u = df.TrialFunction(V)
v = df.TestFunction(V)

a = (C[i, j, k, l] * sym(grad(u))[k, l] * sym(grad(v))[i, j]) * dx
