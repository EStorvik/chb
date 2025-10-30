"""Nonlinear compressibility for Biot simulations."""

import chb


class NonlinearCompressibility:
    """
    Nonlinear compressibility interpolated between two values.

    The compressibility is interpolated between M0 and M1 based on the
    phase field value using a specified interpolation function.
    """

    def __init__(
        self,
        M0: float,
        M1: float,
        interpolator: chb.interpolate.StandardInterpolator = (
            chb.interpolate.StandardInterpolator()
        ),
    ):
        """Initialize the nonlinear compressibility.

        Args:
            M0 (float): Compressibility value at phasefield = 0
            M1 (float): Compressibility value at phasefield = 1
            interpolator (chb.interpolate.StandardInterpolator, optional):
                Interpolator for the phasefield. Defaults to
                StandardInterpolator().

        Attributes:
            M0 (float): Compressibility value at phasefield = 0
            M1 (float): Compressibility value at phasefield = 1
            interpolator: Interpolation function for the phasefield
        """
        self.M0 = M0
        self.M1 = M1
        self.interpolator = interpolator

    def __call__(self, phasefield):
        """
        Evaluate the nonlinear compressibility.

        M(pf) = M0 + I(pf) * (M1 - M0).

        Args:
            phasefield: Phasefield value

        Returns:
            Compressibility value interpolated based on phasefield
        """
        return self.M0 + self.interpolator(phasefield) * (self.M1 - self.M0)

    def prime(self, phasefield):
        """
        Evaluate the derivative of the nonlinear compressibility.

        M'(pf) = I'(pf) * (M1 - M0).

        Args:
            phasefield: Phasefield value

        Returns:
            Derivative of compressibility with respect to phasefield
        """
        return self.interpolator.prime(phasefield) * (self.M1 - self.M0)

    def doubleprime(self, phasefield):
        """
        Evaluate the second derivative of the nonlinear compressibility.

        M''(pf) = I''(pf) * (M1 - M0).

        Args:
            phasefield: Phasefield value

        Returns:
            Second derivative of compressibility with respect to
                phasefield
        """
        return self.interpolator.doubleprime(phasefield) * (self.M1 - self.M0)
