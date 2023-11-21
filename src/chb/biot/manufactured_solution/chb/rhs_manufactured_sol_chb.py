import sympy as sym
from dolfin import Expression


import chb


class CHBManufacturedSolution:
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
    ):
        
        x, y, t = sym.symbols("x[0], x[1], t")
        self.p  = t * x * (x - 1) * y * (y - 1)
        self.ux = t * x * (x - 1) * y * (y - 1)
        self.uy = t * x * (x - 1) * y * (y - 1)
        self.u = [self.ux, self.uy]
        self.pf = t * x * (x - 1) * y * (y - 1)
        self.t = 0
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
        self.divu = sym.diff(self.ux, x) + sym.diff(self.uy, y)

        self.strain = self.epsu - swelling * sym.eye(2) * self.pf

        self.stress = sym.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        self.stress[i, j] += (
                            stiffness.manual(self.pf)[i, j, k, l] * self.strain[k, l]
                        )

        self.stressPrime = sym.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        self.stressPrime[i, j] += (
                            stiffness.manual_prime(self.pf)[i, j, k, l]
                            * self.strain[k, l]
                        )

        self.mu = (
            gamma*(
                1 / ell * doublewell.prime(self.pf)
                - ell * (sym.diff(self.pf, x, x) + sym.diff(self.pf, y, y))
            )
            + 0.5 * self.mat_prod(self.stressPrime, self.strain)
            - swelling * self.mat_prod(sym.eye(2), self.stress)
            + M.prime(self.pf)*self.p**2/(M(self.pf)**2)
            -self.p*alpha.prime(self.pf)*self.divu
        )

        self.gradmu = sym.Matrix([sym.diff(self.mu, x), sym.diff(self.mu, y)])

        self.gradmu0 = -self.gradmu[0]
        self.gradmu1 = self.gradmu[0]
        self.gradmu2 = -self.gradmu[1]
        self.gradmu3 = self.gradmu[1]

        self.effective_stress = self.stress-alpha(self.pf)*self.p*sym.eye(2)

        self.R = sym.diff(self.pf, t)-mobility*(sym.diff(self.mu, x, x)+sym.diff(self.mu, y, y))

        self.f = -sym.Matrix([sym.diff(self.effective_stress[0,0], x)+sym.diff(self.effective_stress[0,1], y), sym.diff(self.effective_stress[1,0], x)+sym.diff(self.effective_stress[1,1], y)])

        self.S_f = sym.diff(self.p / M(self.pf) + alpha(self.pf) * self.divu, t) - (sym.diff(kappa(self.pf)*(sym.diff(self.p, x)), x) + sym.diff(kappa(self.pf)*sym.diff(self.p, y), y))

    def mat_prod(self, A, B):
        prod = 0
        for i in range(2):
            for j in range(2):
                prod += A[i, j] * B[i, j]
        return prod
    
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
    


   

   
# CH
gamma = 1.0
ell = 1.0e-2
mobility = 1
doublewell = chb.DoubleWellPotential()

# Elasticity
stiffness = chb.HeterogeneousStiffnessTensor(interpolator=chb.UnboundedInterpolator())
swelling = 0.3

# Flow
compressibility0 = 1
compressibility1 = 0.1
M = chb.NonlinearCompressibility(compressibility0, compressibility1, chb.UnboundedInterpolator())

permeability0 = 1
permeability1 = 0.1
k = chb.NonlinearPermeability(permeability0, permeability1, chb.UnboundedInterpolator())

# Coupling
alpha0 = 1
alpha1 = 0.5
alpha = chb.NonlinearBiotCoupling(alpha0, alpha1, chb.UnboundedInterpolator())

# Energies
energy_h = chb.CHBHydraulicEnergy(M, alpha)
energy_e = chb.CHBElasticEnergy(stiffness, swelling)
manufsol = CHBManufacturedSolution(doublewell, mobility, gamma, ell, M, alpha, k, stiffness, swelling)

S_f = Expression(str(sym.ccode(manufsol.S_f)), degree = 2, t = 0)

