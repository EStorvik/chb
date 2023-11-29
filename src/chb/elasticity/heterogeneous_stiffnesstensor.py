from typing import Optional

import dolfin as df
import numpy as np
from dolfin import Identity, dx, grad, sym
from ufl_legacy import indices

import chb


class HeterogeneousStiffnessTensor:
    def __init__(
        self,
        stiffness0: Optional[np.ndarray] = None,
        stiffness1: Optional[np.ndarray] = None,
        dim: int = 2,
        interpolator: chb.StandardInterpolator = chb.StandardInterpolator(),
    ) -> None:
        self.interpolator = interpolator
        if stiffness0 is None:
            self.stiffness0np = np.zeros((dim, dim, dim, dim))

            self.stiffness0np[0, 0, 0, 0] = 100
            self.stiffness0np[0, 0, 1, 1] = 20
            self.stiffness0np[0, 0, 1, 0] = 0
            self.stiffness0np[0, 0, 0, 1] = 0

            self.stiffness0np[1, 1, 0, 0] = 20
            self.stiffness0np[1, 1, 1, 1] = 100
            self.stiffness0np[1, 1, 1, 0] = 0
            self.stiffness0np[1, 1, 0, 1] = 0

            self.stiffness0np[0, 1, 0, 0] = 0
            self.stiffness0np[0, 1, 1, 1] = 0
            self.stiffness0np[0, 1, 1, 0] = 100
            self.stiffness0np[0, 1, 0, 1] = 100

            self.stiffness0np[1, 0, 0, 0] = 0
            self.stiffness0np[1, 0, 1, 1] = 0
            self.stiffness0np[1, 0, 1, 0] = 100
            self.stiffness0np[1, 0, 0, 1] = 100

            self.stiffness0 = df.Constant(self.stiffness0np)

        else:
            self.stiffness0 = stiffness0

        if stiffness1 is None:
            self.stiffness1np = np.zeros((dim, dim, dim, dim))

            self.stiffness1np[0, 0, 0, 0] = 1
            self.stiffness1np[0, 0, 1, 1] = 0.1
            self.stiffness1np[0, 0, 1, 0] = 0
            self.stiffness1np[0, 0, 0, 1] = 0

            self.stiffness1np[1, 1, 0, 0] = 0.1
            self.stiffness1np[1, 1, 1, 1] = 1
            self.stiffness1np[1, 1, 1, 0] = 0
            self.stiffness1np[1, 1, 0, 1] = 0

            self.stiffness1np[0, 1, 0, 0] = 0
            self.stiffness1np[0, 1, 1, 1] = 0
            self.stiffness1np[0, 1, 1, 0] = 1
            self.stiffness1np[0, 1, 0, 1] = 1

            self.stiffness1np[1, 0, 0, 0] = 0
            self.stiffness1np[1, 0, 1, 1] = 0
            self.stiffness1np[1, 0, 1, 0] = 1
            self.stiffness1np[1, 0, 0, 1] = 1

            self.stiffness1 = df.Constant(self.stiffness1np)

        else:
            self.stiffness1 = stiffness1

    def manual(self, pf):
        return self.stiffness0np + self.interpolator(pf) * (
            self.stiffness1np - self.stiffness0np
        )

    def manual_prime(self, pf):
        return self.interpolator.prime(pf) * (self.stiffness1np - self.stiffness0np)

    def __call__(self, epsu, epsv, pf):
        i, j, k, l = indices(4)
        return (
            (
                self.stiffness0[i, j, k, l]
                + self.interpolator(pf)
                * (self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l])
            )
            * epsu[k, l]
            * epsv[i, j]
        )

    def prime(self, epsu, epsv, pf):
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.prime(pf)
                * (self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l])
            )
            * epsu[k, l]
            * epsv[i, j]
        )

    def doubleprime(self, epsu, epsv, pf):
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.doubleprime(pf)
                * (self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l])
            )
            * epsu[k, l]
            * epsv[i, j]
        )
