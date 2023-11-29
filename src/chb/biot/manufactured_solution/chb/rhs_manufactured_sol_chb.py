import sympy as sym
from dolfin import Expression

import chb

"""Class for the manufactured solution of the CHB model."""


class CHBManufacturedSolution:
    """Class for the manufactured solution of the CHB model."""

    def __init__(
        self,
        doublewell,
        mobility,
        gamma,
        ell,
        M,
        alpha,
        kappa,
        stiffness: chb.HeterogeneousStiffnessTensor,
        swelling,
    ) -> None:
        """Initialize the manufactured solution with material parameters. The manufactured solutions themselves are defined below."""

        # Symbolic variables
        x, y, t = sym.symbols("x[0], x[1], t")

        # Manufactured solutions
        self.p = t * x * (x - 1) * y * (y - 1)
        self.ux = t * x * (x - 1) * y * (y - 1)
        self.uy = t * x * (x - 1) * y * (y - 1)
        self.u = [self.ux, self.uy]
        self.pf = (
            t * (1 / 3 * x**3 - 1 / 2 * x**2) * (1 / 3 * y**3 - 1 / 2 * y**2)
        )

        # Define the linearized strain tensor
        self.epsu = sym.Matrix(
            [
                [
                    sym.diff(self.ux, x),
                    0.5 * (sym.diff(self.ux, y) + sym.diff(self.uy, x)),
                ],
                [
                    0.5 * (sym.diff(self.ux, y) + sym.diff(self.uy, x)),
                    sym.diff(self.uy, y),
                ],
            ]
        )

        # Define the divergence of the displacement
        self.divu = sym.diff(self.ux, x) + sym.diff(self.uy, y)

        # Define the strain tensor with swelling effects
        self.strain = self.epsu - swelling * sym.eye(2) * self.pf

        # Define the stress tensor
        self.stress = sym.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        self.stress[i, j] += (
                            stiffness.manual(self.pf)[i, j, k, l] * self.strain[k, l]
                        )

        # Differentiate the stress tensor wrt the phase-field
        self.stressPrime = sym.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        self.stressPrime[i, j] += (
                            stiffness.manual_prime(self.pf)[i, j, k, l]
                            * self.strain[k, l]
                        )

        # Define the chemical potential
        self.mu = (
            gamma
            * (
                1 / ell * doublewell.prime(self.pf)
                - ell * (sym.diff(self.pf, x, x) + sym.diff(self.pf, y, y))
            )
            + 0.5 * self.mat_prod(self.stressPrime, self.strain)
            - swelling * self.mat_prod(sym.eye(2), self.stress)
            + M.prime(self.pf) * self.p**2 / (M(self.pf) ** 2)
            - self.p * alpha.prime(self.pf) * self.divu
        )

        # Define the gradient of the chemical potential
        self.gradmu = sym.Matrix([sym.diff(self.mu, x), sym.diff(self.mu, y)])

        # Also on each of the four sides of the unit square times outwards pointing normal
        self.gradmu0 = -self.gradmu[0]
        self.gradmu1 = self.gradmu[0]
        self.gradmu2 = -self.gradmu[1]
        self.gradmu3 = self.gradmu[1]

        # Define the gradient of the phase-field
        self.gradpf = sym.Matrix([sym.diff(self.pf, x), sym.diff(self.pf, y)])

        # Also on each of the four sides of the unit square times outwards pointing normal
        self.gradpf0 = -self.gradpf[0]
        self.gradpf1 = self.gradpf[0]
        self.gradpf2 = -self.gradpf[1]
        self.gradpf3 = self.gradpf[1]

        # Define the effective stress
        self.effective_stress = self.stress - alpha(self.pf) * self.p * sym.eye(2)

        # Define right hand side for the phase-field
        self.R = sym.diff(self.pf, t) - mobility * (
            sym.diff(self.mu, x, x) + sym.diff(self.mu, y, y)
        )

        # Define body forces
        self.f = -sym.Matrix(
            [
                sym.diff(self.effective_stress[0, 0], x)
                + sym.diff(self.effective_stress[0, 1], y),
                sym.diff(self.effective_stress[1, 0], x)
                + sym.diff(self.effective_stress[1, 1], y),
            ]
        )

        # Define the source term (for pressure equation)
        self.S_f = sym.diff(self.p / M(self.pf) + alpha(self.pf) * self.divu, t) - (
            sym.diff(kappa(self.pf) * (sym.diff(self.p, x)), x)
            + sym.diff(kappa(self.pf) * sym.diff(self.p, y), y)
        )

    def mat_prod(self, A: sym.Matrix, B: sym.Matrix) -> sym.Matrix:
        """Inner product of two matrices."""
        prod = 0
        for i in range(2):
            for j in range(2):
                prod += A[i, j] * B[i, j]
        return prod

    # Define output functions in cpp code strings.
    def S_f_out(self) -> str:
        return str(sym.ccode(self.S_f))

    def R_out(self) -> str:
        return str(sym.ccode(self.R))

    def f0_out(self) -> str:
        return str(sym.ccode(self.f[0]))

    def f1_out(self) -> str:
        return str(sym.ccode(self.f[1]))

    def mu_out(self) -> str:
        return str(sym.ccode(self.mu))

    def p_out(self) -> str:
        return str(sym.ccode(self.p))

    def ux_out(self) -> str:
        return str(sym.ccode(self.ux))

    def uy_out(self) -> str:
        return str(sym.ccode(self.uy))

    def pf_out(self) -> str:
        return str(sym.ccode(self.pf))

    def epsu_out(self) -> str:
        return str(sym.ccode(self.epsu))

    def divu_out(self) -> str:
        return str(sym.ccode(self.divu))

    def gradmu0_out(self) -> str:
        return str(sym.ccode(self.gradmu0))

    def gradmu1_out(self) -> str:
        return str(sym.ccode(self.gradmu1))

    def gradmu2_out(self) -> str:
        return str(sym.ccode(self.gradmu2))

    def gradmu3_out(self) -> str:
        return str(sym.ccode(self.gradmu3))

    def gradpf0_out(self) -> str:
        return str(sym.ccode(self.gradpf0))

    def gradpf1_out(self) -> str:
        return str(sym.ccode(self.gradpf1))

    def gradpf2_out(self) -> str:
        return str(sym.ccode(self.gradpf2))

    def gradpf3_out(self) -> str:
        return str(sym.ccode(self.gradpf3))
