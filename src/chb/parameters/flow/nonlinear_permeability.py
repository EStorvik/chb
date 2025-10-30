"""Nonlinear permeability for Biot simulations."""

import chb


class NonlinearPermeability:
    """
    Nonlinear permeability interpolated between two values.

    The permeability is interpolated between k0 and k1 based on the
    phase field value using a specified interpolation function.
    """

    def __init__(
        self,
        k0: float,
        k1: float,
        interpolator: chb.interpolate.StandardInterpolator = (
            chb.interpolate.StandardInterpolator()
        ),
    ):
        """Initialize the nonlinear permeability.

        Args:
            k0 (float): Permeability value at phasefield = 0
            k1 (float): Permeability value at phasefield = 1
            interpolator (chb.interpolate.StandardInterpolator, optional):
                Interpolator for the phasefield. Defaults to
                StandardInterpolator().

        Attributes:
            k0 (float): Permeability value at phasefield = 0
            k1 (float): Permeability value at phasefield = 1
            interpolator: Interpolation function for the phasefield
        """
        self.k0 = k0
        self.k1 = k1
        self.interpolator = interpolator

    def __call__(self, phasefield):
        """
        Evaluate the nonlinear permeability.

        k(pf) = k0 + I(pf) * (k1 - k0).

        Args:
            phasefield: Phasefield value

        Returns:
            Permeability value interpolated based on phasefield
        """
        return self.k0 + self.interpolator(phasefield) * (self.k1 - self.k0)

    def prime(self, phasefield):
        """
        Evaluate the derivative of the nonlinear permeability.

        k'(pf) = I'(pf) * (k1 - k0).

        Args:
            phasefield: Phasefield value

        Returns:
            Derivative of permeability with respect to phasefield
        """
        return self.interpolator.prime(phasefield) * (self.k1 - self.k0)

    def doubleprime(self, phasefield):
        """
        Evaluate the second derivative of the nonlinear permeability.

        k''(pf) = I''(pf) * (k1 - k0).

        Args:
            phasefield: Phasefield value

        Returns:
            Second derivative of permeability with respect to phasefield
        """
        return self.interpolator.doubleprime(phasefield) * (self.k1 - self.k0)
