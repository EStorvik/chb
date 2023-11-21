from typing import Optional

import numpy as np
import dolfin as df
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

            self.stiffness0np[0, 0, 0, 0] = 4
            self.stiffness0np[0, 0, 1, 1] = 2
            self.stiffness0np[0, 0, 1, 0] = 0
            self.stiffness0np[0, 0, 0, 1] = 0

            self.stiffness0np[1, 1, 0, 0] = 2
            self.stiffness0np[1, 1, 1, 1] = 4
            self.stiffness0np[1, 1, 1, 0] = 0
            self.stiffness0np[1, 1, 0, 1] = 0

            self.stiffness0np[0, 1, 0, 0] = 0
            self.stiffness0np[0, 1, 1, 1] = 0
            self.stiffness0np[0, 1, 1, 0] = 4
            self.stiffness0np[0, 1, 0, 1] = 4

            self.stiffness0np[1, 0, 0, 0] = 0
            self.stiffness0np[1, 0, 1, 1] = 0
            self.stiffness0np[1, 0, 1, 0] = 4
            self.stiffness0np[1, 0, 0, 1] = 4

            self.stiffness0 = df.Constant(self.stiffness0np)

        else:
            self.stiffness0 = stiffness0

        if stiffness1 is None:
            self.stiffness1np = np.zeros((dim, dim, dim, dim))

            self.stiffness1np[0, 0, 0, 0] = 1
            self.stiffness1np[0, 0, 1, 1] = 0.5
            self.stiffness1np[0, 0, 1, 0] = 0
            self.stiffness1np[0, 0, 0, 1] = 0

            self.stiffness1np[1, 1, 0, 0] = 0.5
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

    def __call__(self, u, v, phasefield, swelling=0):
        i, j, k, l = indices(4)
        return (
            (
                self.stiffness0[i, j, k, l]
                + self.interpolator(phasefield)(
                    self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
                )
            )
            * (sym(grad(u))[k, l] - swelling * phasefield * Identity(2)[k, l])
            * (sym(grad(v))[i, j] - swelling * phasefield * Identity(2)[i, j])
        )

    def deps(self, u, v, phasefield, swelling=0):
        i, j, k, l = indices(4)
        return (
            (
                self.stiffness0[i, j, k, l]
                + self.interpolator(phasefield)*(
                    self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
                )
            )
            * (sym(grad(u))[k, l] - swelling * phasefield * Identity(2)[k, l])
            * (sym(grad(v))[i, j])
        )

    def prime(self, u, v, phasefield, swelling=0):
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.prime(phasefield)*(
                    self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
                )
            )
            * (sym(grad(u))[k, l] - swelling * phasefield * Identity(2)[k, l])
            * (sym(grad(v))[i, j] - swelling * phasefield * Identity(2)[i, j])
        )

    def doubleprime(self, u, v, phasefield, swelling=0):
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.doubleprime(phasefield)*(
                    self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
                )
            )
            * (sym(grad(u))[k, l] - swelling * phasefield * Identity(2)[k, l])
            * (sym(grad(v))[i, j] - swelling * phasefield * Identity(2)[i, j])
        )

    def idStress(self, u, pf, swelling=0):
        i, j, k, l = indices(4)
        return (
            (
                self.stiffness0[i, j, k, l]
                + self.interpolator(pf)*(
                    self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
                )
            )
            * (sym(grad(u))[k, l] - swelling * pf * Identity(2)[k, l])
            * (swelling * Identity(2)[i, j])
        )

    def idStressPrime(self, u, pf, swelling=0):
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.prime(pf)*(
                    self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
                )
            )
            * (sym(grad(u))[k, l] - swelling * pf * Identity(2)[k, l])
            * (swelling * Identity(2)[i, j])
        )

    def idIdStiffness(self, pf, swelling):
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.prime(pf)*(
                    self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
                )
            )
            * (swelling * Identity(2)[k, l])
            * (swelling * Identity(2)[i, j])
        )

    def primeU(self, u, u_prev, pf, swelling):
        i, j, k, l = indices(4)
        return (
            self.interpolator.prime(pf)*(
                self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
            )
        ) * (sym(grad(u_prev))[k, l] - swelling * pf * Identity(2)[k, l]) * (
            sym(grad(u - u_prev))[i, j]
        ) - (
            self.stiffness0[i, j, k, l]
            + self.interpolator(pf)*(
                self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
            )
        ) * (
            swelling * Identity(2)[k, l]
        ) * (
            sym(grad(u - u_prev))[i, j]
        )

    def dpfdeps(self, pf_prev, u_prev, v, swelling):
        i, j, k, l = indices(4)
        return (
            self.interpolator.prime(pf_prev)*(
                self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
            )
        ) * (sym(grad(u_prev))[k, l] - swelling * pf_prev * Identity(2)[k, l]) * (
            sym(grad(v))[i, j]
        ) - (
            self.stiffness0[i, j, k, l]
            + self.interpolator(pf_prev)*(
                self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l]
            )
        ) * (
            swelling * pf_prev * Identity(2)[k, l]
        ) * (
            sym(grad(v))[i, j]
        )
