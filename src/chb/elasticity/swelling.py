import dolfin as df

"""Swelling term"""


class Swelling:
    """Swelling term"""

    def __init__(
        self,
        swelling_parameter: float,
        dim: int = 2,
    ) -> None:
        """Initialize the swelling term

        Args:
            swelling_parameter (float): Swelling parameter
            dim (int, optional): Dimension. Defaults to 2.

        Attributes:
            swelling_parameter (float): Swelling parameter
            dim (int): Dimension.
        """
        self.swelling_parameter = swelling_parameter
        self.dim = dim

    def __call__(self, pf: df.Function) -> df.Function:
        """Evaluate the swelling term

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Swelling term
        """
        return 2 * self.swelling_parameter * (pf - 0.5) * df.Identity(self.dim)

    def prime(self):
        """Evaluate the derivative of the swelling term

        Returns:
            df.Function: Derivative of the swelling term
        """
        return 2 * self.swelling_parameter * df.Identity(self.dim)
