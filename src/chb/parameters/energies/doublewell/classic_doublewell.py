"""Classic double well potential."""

from dolfinx.fem import Function

class DoubleWellPotential:
    """Classic double well potential. Psi = pf^2 (1 - pf)^2."""

    def __init__(self, scaling: float = 1.0) -> None:
        """Initialize the double well potential.

        Args:
            scaling (float, optional): Scaling factor. Defaults to 1.0.

        Attributes:
            scaling (float): Scaling factor.
        """
        self.scaling = scaling

    def __call__(self, pf: Function) -> Function:
        """Evaluate the double well potential.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Double well potential
        """
        return self.scaling * pf**2 * (1 - pf) ** 2

    def prime(self, pf: Function) -> Function:
        """Evaluate the derivative of the double well potential.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the double well potential
        """
        return self.scaling * 2 * pf * (1 - pf) * (1 - 2 * pf)

    def doubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the double well potential.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the double well potential
        """
        return self.scaling * 2 * (1 - 6 * pf + 6 * pf**2)

    def c(self, pf: Function) -> Function:
        """Evaluate the convex part of the double well potential. Psi_c = (pf- 0.5)^4+ 0.0625.

        args:
            pf (Function): Phasefield

        returns:
            Function: Convex part of the double well potential
        """
        return self.scaling * ((pf - 0.5) ** 4 + 0.0625)

    def cprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the convex part of the double well potential. Psi_c' = 4(pf- 0.5)^3.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the convex part of the double well potential
        """
        return self.scaling * 4 * (pf - 0.5) ** 3

    def cdoubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the convex part of the double well potential. Psi_c'' = 12(pf- 0.5)^2.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the convex part of the double well potential
        """
        return self.scaling * 12 * (pf - 0.5) ** 2

    def e(self, pf: Function) -> Function:
        """Evaluate the expansive part of the double well potential. Psi_e = 0.5(pf- 0.5)^2.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Expansive part of the double well potential
        """
        return self.scaling * 0.5 * (pf - 0.5) ** 2

    def eprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the expansive part of the double well potential. Psi_e' = (pf- 0.5).

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the expansive part of the double well potential
        """
        return self.scaling * (pf - 0.5)

    def edoubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the expansive part of the double well potential. Psi_e'' = 1.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the expansive part of the double well potential
        """
        return self.scaling * 1.0
