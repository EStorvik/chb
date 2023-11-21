import sympy as sym


def main():
    x, y, t = sym.symbols("x[0], x[1], t")

    M, alpha, kappa, lame_mu, lame_lambda, p_ref = sym.symbols(
        "M, alpha, kappa, lame_mu, lame_lambda, p_ref"
    )

    p = t * x * (x - 1) * y * (y - 1) / p_ref
    ux = t * x * (x - 1) * y * (y - 1)
    uy = t * x * (x - 1) * y * (y - 1)

    divu = sym.diff(ux, x) + sym.diff(uy, y)

    divdivuI1 = sym.diff(divu, x)
    divdivuI2 = sym.diff(divu, y)

    epsu11 = sym.diff(ux, x)
    epsu12 = 0.5 * (sym.diff(ux, y) + sym.diff(uy, x))
    epsu21 = epsu12
    epsu22 = sym.diff(uy, y)

    diveps1 = sym.diff(epsu11, x) + sym.diff(epsu12, y)
    diveps2 = sym.diff(epsu21, x) + sym.diff(epsu22, y)

    laplace_p = sym.diff(sym.diff(p, x), x) + sym.diff(sym.diff(p, y), y)

    S_f = sym.diff(p / M + alpha * divu, t) - kappa * laplace_p
    f_1 = -(2 * lame_mu * diveps1 + lame_lambda * divdivuI1) + alpha * sym.diff(p, x)
    f_2 = -(2 * lame_mu * diveps2 + lame_lambda * divdivuI2) + alpha * sym.diff(p, y)
    print(f_2)


# Main execution
if __name__ == "__main__":
    main()
