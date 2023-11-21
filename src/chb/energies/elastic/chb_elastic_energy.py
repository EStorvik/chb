from dolfin import assemble, dx

import chb
from ufl_legacy import indices


class CHBElasticEnergy:
    def __init__(self, stiffness: chb.HeterogeneousStiffnessTensor, swelling: float):
        self.stiffness = stiffness
        self.swelling = swelling

    def __call__(self, pf, u):
        return assemble(0.5 * self.stiffness(u, pf, self.swelling) * dx)

    def dpf(self, pf, u):
        return 0.5 * self.stiffness.prime(
            u, u, pf, self.swelling
        ) - self.stiffness.idStress(u, pf, self.swelling)

    def dpf_prime(self, pf, u, pf_prev, u_prev):
        return (
            0.5 * self.stiffness.doubleprime(u_prev, u_prev, pf_prev, self.swelling)
            - 2 * self.stiffness.idStressPrime(u_prev, pf_prev, self.swelling)
            + self.stiffness.idIdStiffness(pf_prev, self.swelling)
        ) * (pf - pf_prev) + self.stiffness.primeU(u, u_prev, pf_prev, self.swelling)

    def dpfdpf(self, pf, pf_prev, u_prev):
        return (
            0.5 * self.stiffness.doubleprime(u_prev, u_prev, pf_prev, self.swelling)
            - 2 * self.stiffness.idStressPrime(u_prev, pf_prev, self.swelling)
            + self.stiffness.idIdStiffness(pf_prev, self.swelling)
        ) * (pf - pf_prev)

    def dudpf(self, u, pf_prev, u_prev):
        return self.stiffness.primeU(u, u_prev, pf_prev, self.swelling)

    def du(self, pf, u, eta_u):
        return self.stiffness.deps(u, eta_u, pf, self.swelling)

    def dpfdu(self, pf, pf_prev, u_prev, eta_u):
        return self.stiffness.dpfdeps(pf_prev, u_prev, eta_u, self.swelling) * (
            pf - pf_prev
        )
