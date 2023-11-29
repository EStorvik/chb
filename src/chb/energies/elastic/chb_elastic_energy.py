from dolfin import Identity, assemble, dx, grad, sym

import chb


class CHBElasticEnergy:
    def __init__(
        self, stiffness: chb.HeterogeneousStiffnessTensor, swelling: chb.Swelling
    ) -> None:
        self.stiffness = stiffness
        self.swelling = swelling

    def __call__(self, pf, u):
        return assemble(
            0.5
            * self.stiffness(
                epsu=sym(grad(u)) - self.swelling(pf),
                epsv=sym(grad(u)) - self.swelling(pf),
                pf=pf,
            )
            * dx
        )

    def dpf(self, pf, u):
        return 0.5 * self.stiffness.prime(
            epsu=sym(grad(u)) - self.swelling(pf),
            epsv=sym(grad(u)) - self.swelling(pf),
            pf=pf,
        ) - self.stiffness(
            epsu=sym(grad(u)) - self.swelling(pf), epsv=self.swelling.prime(), pf=pf
        )

    def dpf_dpf(self, pf, u):
        return (
            0.5
            * self.stiffness.doubleprime(
                epsu=sym(grad(u)) - self.swelling(pf),
                epsv=sym(grad(u)) - self.swelling(pf),
                pf=pf,
            )
            - 2
            * self.stiffness.prime(
                epsu=sym(grad(u)) - self.swelling(pf),
                epsv=self.swelling.prime(),
                pf=pf,
            )
            + self.stiffness(
                epsu=self.swelling.prime(),
                epsv=self.swelling.prime(),
                pf=pf,
            )
        )

    def dpf_deps(self, pf, u, du):
        return self.stiffness.prime(
            epsu=sym(grad(du)),
            epsv=sym(grad(u)) - self.swelling(pf),
            pf=pf,
        ) - self.stiffness(
            epsu=sym(grad(du)),
            epsv=self.swelling.prime(),
            pf=pf,
        )

    def deps(self, pf, u, du):
        return self.stiffness(
            epsu=sym(grad(u)) - self.swelling(pf),
            epsv=sym(grad(du)),
            pf=pf,
        )

    def deps_dpf(self, pf, u, du):
        return self.stiffness.prime(
            epsu=sym(grad(u)) - self.swelling(pf),
            epsv=sym(grad(du)),
            pf=pf,
        ) - self.stiffness(
            epsu=self.swelling.prime(),
            epsv=sym(grad(du)),
            pf=pf,
        )
