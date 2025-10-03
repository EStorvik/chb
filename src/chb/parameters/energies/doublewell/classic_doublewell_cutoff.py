"""Classic double well potential with cutoff."""

from dolfinx.fem import Function
import ufl

class DoubleWellPotential_cutoff:
    """Classic double well potential Psi = pf^2 (1 - pf)^2 with cutoff (for the convex part)."""

    def __init__(self, scaling: float = 1.0, beta: float = 1.0) -> None:
        """Initialize the double well potential.

        Args:
            scaling (float, optional): Scaling factor. Defaults to 1.0.
            beta (float, optional): Cutoff parameter. Defaults to 1.0.

        Attributes:
            scaling (float): Scaling factor.
            beta (float): Cutoff parameter.
        """
        self.scaling = scaling
        self.beta = beta

    def __call__(self, pf: Function) -> Function:
        """Evaluate the double well potential with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta + 0.5),ufl.ge(pf, self.beta + 0.5)), 
            self.scaling * (2 * (self.beta ** 2 - 0.25) * (pf - 0.5) ** 2 - (self.beta ** 4 - 0.0625)), 
            self.scaling * pf**2 * (1 - pf) ** 2
        )

    def prime(self, pf: Function) -> Function:
        """Evaluate the derivative of the double well potential with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta + 0.5),ufl.ge(pf, self.beta + 0.5)), 
            self.scaling * 4 * (self.beta ** 2 - 0.25) * (pf - 0.5), 
            self.scaling * 2 * pf * (1 - pf) * (1 - 2 * pf)
        )

    def doubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the double well potential with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta + 0.5),ufl.ge(pf, self.beta + 0.5)), 
            self.scaling * 4 * (self.beta ** 2 - 0.25), 
            self.scaling * 2 * (1 - 6 * pf + 6 * pf**2)
        )

    def c(self, pf: Function) -> Function:
        """Evaluate the convex part of the double well potential Psi_c = (pf-0.5)^4+0.0625 with cutoff.

        args:
            pf (Function): Phasefield

        returns:
            Function: Convex part of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta + 0.5),ufl.ge(pf, self.beta + 0.5)), 
            self.scaling * (2 * self.beta ** 2 * (pf - 0.5) ** 2 - (self.beta ** 4 - 0.0625)), 
            self.scaling * ((pf - 0.5) ** 4 + 0.0625)
        )

    def cprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the convex part of the double well potential Psi_c' = 4(pf-0.5)^3 with cutoff (Lipschitz continuous).

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the convex part of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta + 0.5),ufl.ge(pf, self.beta + 0.5)), 
            self.scaling * 4 * self.beta ** 2 * (pf - 0.5), 
            self.scaling * 4 * (pf - 0.5) ** 3
        )
        
    def cdoubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the convex part of the double well potential Psi_c'' = 12(pf-0.5)^2 with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the convex part of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta + 0.5),ufl.ge(pf, self.beta + 0.5)), 
            self.scaling * 4 * self.beta ** 2, 
            self.scaling * 12 * (pf - 0.5) ** 2
        )

    def e(self, pf: Function) -> Function:
        """Evaluate the expansive part of the double well potential. Psi_e = 0.5(pf-0.5)^2.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Expansive part of the double well potential
        """
        return self.scaling * 0.5 * (pf - 0.5) ** 2

    def eprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the expansive part of the double well potential. Psi_e' = (pf-0.5).

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

"""Classic double well potential with cutoff for -1/1 CH model."""

class SymmetricDoubleWellPotential_cutoff:
    """Classic double well potential Psi = (1 - pf^2)^2 with cutoff (for the convex part)."""

    def __init__(self, scaling: float = 1.0, beta: float = 1.5) -> None:
        """Initialize the double well potential.

        Args:
            scaling (float, optional): Scaling factor. Defaults to 1.0.
            beta (float, optional): Cutoff parameter. Defaults to 1.5.

        Attributes:
            scaling (float): Scaling factor.
            beta (float): Cutoff parameter.
        """
        self.scaling = scaling
        self.beta = beta

    def __call__(self, pf: Function) -> Function:
        """Evaluate the double well potential with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta ),ufl.ge(pf, self.beta)), 
            self.scaling * (2 * (self.beta ** 2 - 1.0) * pf ** 2 - (self.beta ** 4 - 1)), 
            self.scaling  * (1 - pf ** 2) ** 2
        )

    def prime(self, pf: Function) -> Function:
        """Evaluate the derivative of the double well potential with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta),ufl.ge(pf, self.beta)), 
            self.scaling * 4 * (self.beta ** 2 - 1.0) * pf, 
            self.scaling * 4 * pf * (1 - pf**2) 
        )

    def doubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the double well potential with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta ),ufl.ge(pf, self.beta)), 
            self.scaling * 4 * (self.beta ** 2 - 1.0), 
            self.scaling * 4 * ((1 - pf**2) - 2 * pf ** 2)
        )

    def c(self, pf: Function) -> Function:
        """Evaluate the convex part of the double well potential Psi_c = (pf-0.5)^4+0.0625 with cutoff.

        args:
            pf (Function): Phasefield

        returns:
            Function: Convex part of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta),ufl.ge(pf, self.beta)), 
            self.scaling * (2 * self.beta ** 2 * (pf) ** 2 - (self.beta ** 4 - 1.0)), 
            self.scaling * ((pf) ** 4 + 1.0)
        )

    def cprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the convex part of the double well potential Psi_c' = 4(pf-0.5)^3 with cutoff (Lipschitz continuous).

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the convex part of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta),ufl.ge(pf, self.beta)), 
            self.scaling * 4 * self.beta ** 2 * pf, 
            self.scaling * 4 * pf ** 3
        )
        
    def cdoubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the convex part of the double well potential Psi_c'' = 12(pf-0.5)^2 with cutoff.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the convex part of the double well potential with cutoff
        """
        return ufl.conditional(
            ufl.classes.OrCondition(ufl.le(pf, - self.beta),ufl.ge(pf, self.beta)), 
            self.scaling * 4 * self.beta ** 2, 
            self.scaling * 12 * pf ** 2
        )

    def e(self, pf: Function) -> Function:
        """Evaluate the expansive part of the double well potential. Psi_e = 0.5(pf-0.5)^2.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Expansive part of the double well potential
        """
        return self.scaling * 2 * pf ** 2

    def eprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the expansive part of the double well potential. Psi_e' = (pf-0.5).

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Derivative of the expansive part of the double well potential
        """
        return self.scaling * 4.0 * pf

    def edoubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the expansive part of the double well potential. Psi_e'' = 1.

        Args:
            pf (Function): Phasefield

        Returns:
            Function: Second derivative of the expansive part of the double well potential
        """
        return self.scaling * 4.0

