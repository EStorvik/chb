from typing import Optional

import dolfin as df
import numpy as np
from ufl_legacy import indices
import sympy as sym

import chb

"""Heterogeneous and anisotropic general stiffness tensor."""

class HeterogeneousStiffnessTensor:
    """Heterogeneous and anisotropic general stiffness tensor."""
    def __init__(
        self,
        stiffness0: Optional[np.ndarray] = None,
        stiffness1: Optional[np.ndarray] = None,
        dim: int = 2,
        interpolator: chb.StandardInterpolator = chb.StandardInterpolator(),
        voigt: bool = False,
    ) -> None:
        """Initialize the heterogeneous and anisotropic general stiffness tensor.

        Args:
            stiffness0 (np.ndarray, optional): Stiffness tensor for phasefield = 0. Defaults to None.
            stiffness1 (np.ndarray, optional): Stiffness tensor for phasefield = 1. Defaults to None.
            dim (int, optional): Dimension. Defaults to 2.
            interpolator (chb.StandardInterpolator, optional): Interpolator for the phasefield. Defaults to chb.StandardInterpolator().
            voigt (bool, optional): input is given in Voigt notation. Defaults to False.

        Attributes:
            stiffness0 (df.Constant): Stiffness tensor for phasefield = 0
            stiffness1 (df.Constant): Stiffness tensor for phasefield = 1
            dim (int): Dimension. As of now, the only acceptable input is 2.
            interpolator (chb.StandardInterpolator): Interpolator for the phasefield
        """

        # Define interpolator
        self.interpolator = interpolator

        assert dim == 2, "Only 2D is supported as of now."

        # Define stiffness tensors
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

        elif voigt:
            self.stiffness0np = np.zeros((dim, dim, dim, dim))

            self.stiffness0np[0, 0, 0, 0] = stiffness0[0, 0]
            self.stiffness0np[0, 0, 1, 1] = stiffness0[0, 1]
            self.stiffness0np[0, 0, 1, 0] = stiffness0[0, 2]
            self.stiffness0np[0, 0, 0, 1] = stiffness0[0, 2]

            self.stiffness0np[1, 1, 0, 0] = stiffness0[1, 0]
            self.stiffness0np[1, 1, 1, 1] = stiffness0[1, 1]
            self.stiffness0np[1, 1, 1, 0] = stiffness0[1, 2]
            self.stiffness0np[1, 1, 0, 1] = stiffness0[1, 2]

            self.stiffness0np[0, 1, 0, 0] = stiffness0[2, 0]
            self.stiffness0np[0, 1, 1, 1] = stiffness0[2, 1]
            self.stiffness0np[0, 1, 1, 0] = stiffness0[2, 2]
            self.stiffness0np[0, 1, 0, 1] = stiffness0[2, 2]

            self.stiffness0np[1, 0, 0, 0] = stiffness0[2, 0]
            self.stiffness0np[1, 0, 1, 1] = stiffness0[2, 1]
            self.stiffness0np[1, 0, 1, 0] = stiffness0[2, 2]
            self.stiffness0np[1, 0, 0, 1] = stiffness0[2, 2]

            self.stiffness0 = df.Constant(self.stiffness0np)
        
        else:
            self.stiffness0 = df.Constant(stiffness0)


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

        elif voigt:
            self.stiffness1np = np.zeros((dim, dim, dim, dim))

            self.stiffness1np[0, 0, 0, 0] = stiffness1[0, 0]
            self.stiffness1np[0, 0, 1, 1] = stiffness1[0, 1]
            self.stiffness1np[0, 0, 1, 0] = stiffness1[0, 2]
            self.stiffness1np[0, 0, 0, 1] = stiffness1[0, 2]

            self.stiffness1np[1, 1, 0, 0] = stiffness1[1, 0]
            self.stiffness1np[1, 1, 1, 1] = stiffness1[1, 1]
            self.stiffness1np[1, 1, 1, 0] = stiffness1[1, 2]
            self.stiffness1np[1, 1, 0, 1] = stiffness1[1, 2]

            self.stiffness1np[0, 1, 0, 0] = stiffness1[2, 0]
            self.stiffness1np[0, 1, 1, 1] = stiffness1[2, 1]
            self.stiffness1np[0, 1, 1, 0] = stiffness1[2, 2]
            self.stiffness1np[0, 1, 0, 1] = stiffness1[2, 2]

            self.stiffness1np[1, 0, 0, 0] = stiffness1[2, 0]
            self.stiffness1np[1, 0, 1, 1] = stiffness1[2, 1]
            self.stiffness1np[1, 0, 1, 0] = stiffness1[2, 2]
            self.stiffness1np[1, 0, 0, 1] = stiffness1[2, 2]
            
            self.stiffness1 = df.Constant(self.stiffness1np)

        else:
            self.stiffness1 = df.Constant(stiffness1)

    def manual(self, pf):
        """Evaluate the heterogeneous and anisotropic general stiffness tensor. For use in Sympy.
        
        Args:
            pf (sympy.Symbol): Phasefield
            
        Returns:
            sympy.Symbol: Heterogeneous and anisotropic general stiffness tensor
        """
        return self.stiffness0np + self.interpolator(pf) * (
            self.stiffness1np - self.stiffness0np
        )

    def manual_prime(self, pf):
        """Evaluate the derivative of the heterogeneous and anisotropic general stiffness tensor. For use in Sympy.
        
        Args:
            pf (sympy.Symbol): Phasefield
            
        Returns:
            sympy.Symbol: Derivative of the heterogeneous and anisotropic general stiffness tensor
        """
        return self.interpolator.prime(pf) * (self.stiffness1np - self.stiffness0np)

    def __call__(self, epsu: df.Function, epsv: df.Function, pf: df.Function) -> df.Function:
        """Evaluate the heterogeneous and anisotropic general stiffness tensor.
        
        Args:
            epsu (df.Function): Displacement
            epsv (df.Function): Testfunction for displacement
            pf (df.Function): Phasefield
            
        Returns:
            df.Function: Heterogeneous and anisotropic general stiffness tensor
        """
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

    def prime(self, epsu: df.Function, epsv: df.Function, pf: df.Function) -> df.Function:
        """Evaluate the derivative of the heterogeneous and anisotropic general stiffness tensor.
        
        Args:
            epsu (df.Function): Displacement
            epsv (df.Function): Testfunction for displacement
            pf (df.Function): Phasefield
            
        Returns:
            df.Function: Derivative of the heterogeneous and anisotropic general stiffness tensor
        """
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.prime(pf)
                * (self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l])
            )
            * epsu[k, l]
            * epsv[i, j]
        )

    def doubleprime(self, epsu: df.Function, epsv: df.Function, pf: df.Function) -> df.Function:
        """Evaluate the second derivative of the heterogeneous and anisotropic general stiffness tensor.

        Args:
            epsu (df.Function): Displacement
            epsv (df.Function): Testfunction for displacement
            pf (df.Function): Phasefield

        Returns:
            df.Function: Second derivative of the heterogeneous and anisotropic general stiffness tensor
        """
        i, j, k, l = indices(4)
        return (
            (
                self.interpolator.doubleprime(pf)
                * (self.stiffness1[i, j, k, l] - self.stiffness0[i, j, k, l])
            )
            * epsu[k, l]
            * epsv[i, j]
        )
