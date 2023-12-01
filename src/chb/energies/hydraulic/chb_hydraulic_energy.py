from dolfin import assemble, div, dx


class CHBHydraulicEnergy:
    def __init__(self, compressibility, alpha) -> None:
        self.M = compressibility
        self.alpha = alpha

    def __call__(self, pf, p):
        return assemble((p**2 / (2.0 * self.M(pf))) * dx)

    def dpf(self, pf, u, p):
        return self.M.prime(pf) * p**2 / (self.M(pf) ** 2) - p * self.alpha.prime(
            pf
        ) * div(u)
    
    def dpf_dpf(self, pf, u, p):
        return (
                (
                    self.M.doubleprime(pf) * p**2
                    - 2 * self.M.prime(pf) ** 2 * p**2 * self.M(pf)
                )
                / (self.M(pf) ** 4)
                - p * self.alpha.doubleprime(pf) * div(u)
            )

    def dpf_dp(self, pf, u, p):
        return (
                2 * p * self.M.prime(pf) / (self.M(pf) ** 2)
                - self.alpha.prime(pf) * div(u)
            )

    def dpf_du(self, pf, p, du):
        return  - p * self.alpha.prime(pf) * div(du)


