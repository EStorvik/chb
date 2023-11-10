# Make anisotropic stiffness tensor here.


from typing import Optional
from warnings import warn

import dolfin as df
import numpy as np
from dolfin import dx, grad, sym
from ufl_legacy import indices

import chb


class StiffnessTensor:
    def __init__(self, stiffness_tensor: Optional[np.array] = None, dim: int = 2):
        if stiffness_tensor is None:
            self.stiffness_tensor = np.zeros((dim, dim, dim, dim))
            if dim == 2:
                self.stiffness_tensor[0, 0, 0, 0] = 1
                self.stiffness_tensor[0, 0, 0, 1] = 1
                self.stiffness_tensor[0, 0, 1, 0] = 1
                self.stiffness_tensor[0, 0, 1, 1] = 1

                self.stiffness_tensor[0, 1, 0, 0] = 1
                self.stiffness_tensor[0, 1, 0, 1] = 1
                self.stiffness_tensor[0, 1, 1, 0] = 1
                self.stiffness_tensor[0, 1, 1, 1] = 1

                self.stiffness_tensor[1, 0, 0, 0] = 1
                self.stiffness_tensor[1, 0, 0, 1] = 1
                self.stiffness_tensor[1, 0, 1, 0] = 1
                self.stiffness_tensor[1, 0, 1, 1] = 1

                self.stiffness_tensor[1, 1, 0, 0] = 1
                self.stiffness_tensor[1, 1, 0, 1] = 1
                self.stiffness_tensor[1, 1, 1, 0] = 1
                self.stiffness_tensor[1, 1, 1, 1] = 1

            else:
                warn("Please provide a 3d stiffness tensor")
                raise NotImplementedError

    def __call__(self, u, v):
        i, j, k, l = indices(4)
        return (
            self.stiffness_tensor[i, j, k, l] * sym(grad(u))[k, l] * sym(grad(v))[i, j]
        )
