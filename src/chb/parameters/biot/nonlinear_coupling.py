from dolfinx.fem import Function

import chb


class NonlinearBiotCoupling:
    """Non-linear Biot-Coupling coefficient."""

    def __init__(
        self,
        alpha0: float,
        alpha1: float,
        interpolator: chb.interpolate.StandardInterpolator = (
            chb.interpolate.StandardInterpolator()
        ),
    ) -> None:
        """
        Initialize the non-linear Biot-Coupling coefficient.

        Args:
            alpha0 (float): Coupling coefficient for phasefield = 0
            alpha1 (float): Coupling coefficient for phasefield = 1
            interpolator (chb.StandardInterpolator, optional): Interpolator for the phasefield.
                Defaults to chb.StandardInterpolator().
        """
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.interpolator = interpolator

    def __call__(self, phasefield: Function) -> Function:
        """
        Evaluate the non-linear Biot-Coupling coefficient.

        Args:
            phasefield (Function): Phasefield.

        Returns:
            Function: Non-linear Biot-Coupling coefficient.
        """
        return self.alpha0 + self.interpolator(phasefield) * (self.alpha1 - self.alpha0)

    def prime(self, phasefield: Function) -> Function:
        """
        Evaluate the derivative of the non-linear Biot-Coupling coefficient.

        Args:
            phasefield (Function): Phasefield.

        Returns:
            Function: Derivative of the non-linear Biot-Coupling coefficient.
        """
        return self.interpolator.prime(phasefield) * (self.alpha1 - self.alpha0)

    def doubleprime(self, phasefield: Function) -> Function:
        """
        Evaluate the second derivative of the non-linear Biot-Coupling coefficient.

        Args:
            phasefield (Function): Phasefield.

        Returns:
            Function: Second derivative of the non-linear Biot-Coupling coefficient.
        """
        return self.interpolator.doubleprime(phasefield) * (self.alpha1 - self.alpha0)
