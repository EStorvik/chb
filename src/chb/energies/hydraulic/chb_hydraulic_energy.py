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

    def dpf_prime(self, pf, u, p, pf_prev, u_prev, p_prev):
        return (
            (
                (
                    self.M.doubleprime(pf_prev) * p_prev**2
                    - 2 * self.M.prime(pf_prev) ** 2 * p_prev**2 * self.M(pf_prev)
                )
                / (self.M(pf_prev) ** 4)
                - p_prev * self.alpha.doubleprime(pf_prev) * div(u_prev)
            )
            * (pf - pf_prev)
            + (
                2 * p_prev * self.M.prime(pf_prev) / (self.M(pf_prev) ** 2)
                - self.alpha.prime(pf_prev) * div(u_prev)
            )
            * (p - p_prev)
            + p_prev * self.alpha.prime(pf_prev) * (div(u) - div(u_prev))
        )

    def dpfdpf(self, pf, pf_prev, u_prev, p_prev):
        return (
            (
                self.M.doubleprime(pf_prev) * p_prev**2
                - 2 * self.M.prime(pf_prev) ** 2 * p_prev**2 * self.M(pf_prev)
            )
            / (self.M(pf_prev) ** 4)
            - p_prev * self.alpha.doubleprime(pf_prev) * div(u_prev)
        ) * (pf - pf_prev)

    def dudpf(self, u, pf_prev, u_prev, p_prev):
        return p_prev * self.alpha.prime(pf_prev) * (div(u) - div(u_prev))

    def dpdpf(self, p, pf_prev, u_prev, p_prev):
        return (
            2 * p_prev * self.M.prime(pf_prev) / (self.M(pf_prev) ** 2)
            - self.alpha.prime(pf_prev) * div(u_prev)
        ) * (p - p_prev)

    def du(self, pf, p):
        return -self.alpha(pf) * p
