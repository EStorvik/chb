import dolfinx as dfx
import ufl

"""Swelling term"""


class Swelling:
    """Swelling term"""

    def __init__(
        self,
        swelling_parameter: float,
        pf_ref: float = 0.0,
        dim: int = 2,
    ) -> None:
        """Initialize the swelling term

        Args:
            swelling_parameter (float): Swelling parameter. Defaults to 0.
            dim (int, optional): Dimension. Defaults to 2.

        Attributes:
            swelling_parameter (float): Swelling parameter.
            dim (int): Dimension.
        """
        self.swelling_parameter = swelling_parameter
        self.pf_ref = pf_ref
        self.dim = dim

    def __call__(self, pf: dfx.fem.Function) -> dfx.fem.Function:
        """Evaluate the swelling term

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Swelling term
        """
        return self.swelling_parameter * (pf - self.pf_ref) * ufl.Identity(self.dim)

    def prime(self):
        """Evaluate the derivative of the swelling term

        Returns:
            df.Function: Derivative of the swelling term
        """
        return self.swelling_parameter * ufl.Identity(self.dim)
