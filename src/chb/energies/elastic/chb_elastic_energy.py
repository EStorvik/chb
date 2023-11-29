from dolfin import assemble, dx

import chb


class CHBElasticEnergy:
    def __init__(self, stiffness: chb.HeterogeneousStiffnessTensor):
        self.stiffness = stiffness

    def __call__(self, pf, u):
        return assemble(0.5 * self.stiffness(u, pf) * dx)

    def dpf(self, pf, u):
        return 0.5 * self.stiffness.prime(
            u, u, pf
        ) - self.stiffness.idStress(u, pf)

    def dpf_prime(self, pf, u, pf_prev, u_prev):
        return (
            0.5 * self.stiffness.doubleprime(u_prev, u_prev, pf_prev)
            - 2 * self.stiffness.idStressPrime(u_prev, pf_prev)
            + self.stiffness.idIdStiffness(pf_prev)
        ) * (pf - pf_prev) + self.stiffness.primeU(u, u_prev, pf_prev)

    def dpfdpf(self, pf, pf_prev, u_prev):
        return (
            0.5 * self.stiffness.doubleprime(u_prev, u_prev, pf_prev)
            - 2 * self.stiffness.idStressPrime(u_prev, pf_prev)
            + self.stiffness.idIdStiffness(pf_prev)
        ) * (pf - pf_prev)

    def dudpf(self, u, pf_prev, u_prev):
        return self.stiffness.primeU(u, u_prev, pf_prev)

    def du(self, pf, u, eta_u):
        return self.stiffness.deps(u, eta_u, pf)

    def dpfdu(self, pf, pf_prev, u_prev, eta_u):
        return self.stiffness.dpfdeps(pf_prev, u_prev, eta_u) * (
            pf - pf_prev
        )
