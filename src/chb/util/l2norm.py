import numpy as np
from dolfinx.fem import assemble_scalar, form
from ufl import dx, inner


def l2norm(u):
    _sq = inner(u, u) * dx
    _int_diff = assemble_scalar(form(_sq))
    return np.sqrt(_int_diff)


def l2norm_2(u, v):
    _sq = (inner(u, u) + inner(v, v)) * dx
    _int_diff = assemble_scalar(form(_sq))
    return np.sqrt(_int_diff)


def l2norm_3(u, v, w):
    _sq = (inner(u, u) + inner(v, v) + inner(w, w)) * dx
    _int_diff = assemble_scalar(form(_sq))
    return np.sqrt(_int_diff)
