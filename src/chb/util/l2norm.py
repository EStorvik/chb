import numpy as np

from dolfinx.fem import form, assemble_scalar

from ufl import inner, dx

def l2norm(u):
    _sq = inner(u,u)*dx
    _int_diff = assemble_scalar(form(_sq))
    return np.sqrt(_int_diff)