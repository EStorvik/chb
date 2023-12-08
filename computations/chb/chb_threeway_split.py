from math import sqrt

import dolfin as df
from dolfin import div, dot, grad
from typing import Optional
from ufl_legacy import Measure
import time
import chb


def chb_threeway_split(
    gamma: float,
    ell: float,
    mobility: float,
    doublewell: chb.DoubleWellPotential,
    M: chb.NonlinearCompressibility,
    k: chb.NonlinearPermeability,
    alpha: chb.NonlinearBiotCoupling,
    energy_h: chb.CHBHydraulicEnergy,
    energy_e: chb.CHBElasticEnergy,
    initialconditions: chb.InitialConditions,
    dt: float,
    num_time_steps: int,
    nx: int,
    ny: int,
    max_iter_split: int,
    max_iter_inner_newton: int,
    tol: float,
    output_path: str = "/home/erlend/src/fenics/output/chb/halfnhalf/threewaysplit/",
    output_interval: int = 1,
    log: Optional[str] = None,
    verbose: bool = True,
):
    mesh = df.UnitSquareMesh(nx, ny)
    dx = Measure("dx", domain=mesh)

    # Finite elements
    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    RT0 = df.FiniteElement("RT", mesh.ufl_cell(), 1)
    P0 = df.FiniteElement("DG", mesh.ufl_cell(), 0)
    P2V = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)

    # Function spaces
    ME_ch = df.MixedElement([P1, P1])
    ME_f = df.MixedElement(RT0, P0)
    V_ch = df.FunctionSpace(mesh, ME_ch)
    V_e = df.FunctionSpace(mesh, P2V)
    V_f = df.FunctionSpace(mesh, ME_f)

    # Test and trial functions
    # CH
    ch = df.TrialFunction(V_ch)
    eta_ch = df.TestFunction(V_ch)
    pf, mu = df.split(ch)
    eta_pf, eta_mu = df.split(eta_ch)

    # Elasticity
    u = df.TrialFunction(V_e)
    eta_u = df.TestFunction(V_e)

    # Flow
    fl = df.TrialFunction(V_f)
    eta_f = df.TestFunction(V_f)
    q, p = df.split(fl)
    eta_q, eta_p = df.split(eta_f)

    # Iteration functions
    # CH
    ch_n = df.Function(V_ch)
    pf_n, _ = df.split(ch_n)

    ch_prev = df.Function(V_ch)
    pf_prev, _ = df.split(ch_prev)

    ch_old = df.Function(V_ch)
    pf_old, _ = df.split(ch_old)

    # Elasticity
    u_n = df.Function(V_e)
    u_prev = df.Function(V_e)
    u_old = df.Function(V_e)

    # Flow
    fl_n = df.Function(V_f)
    q_n, p_n = df.split(fl_n)

    fl_prev = df.Function(V_f)
    q_prev, p_prev = df.split(fl_prev)

    fl_old = df.Function(V_f)
    q_old, p_old = df.split(fl_old)

    # Boundary conditions
    def boundary(x, on_boundary):
        return on_boundary

    # Elasticity
    zero_e = df.Constant((0.0, 0.0))
    bc_e = df.DirichletBC(V_e, zero_e, boundary)

    # Flow
    zero_f = df.Constant(0.0)
    bc_f = df.DirichletBC(V_f.sub(1), zero_f, boundary)

    # Initial condtions
    ch_n.interpolate(initialconditions)

    # RHS
    R = df.Constant(0.0)
    f = df.Constant((0.0, 0.0))
    S_f = df.Constant(0.0)

    # Linear variational forms
    F_pf = (
        (pf - pf_old) * eta_pf * dx
        + dt * mobility * dot(grad(mu), grad(eta_pf)) * dx
        - dt * R * eta_pf * dx
    )
    F_mu = (
        mu * eta_mu * dx
        - gamma * ell * dot(grad(pf), grad(eta_mu)) * dx
        - gamma
        / ell
        * (
            doublewell.cprime(pf_prev)
            + doublewell.cdoubleprime(pf_prev) * (pf - pf_prev)
            - doublewell.eprime(pf_old)
        )
        * eta_mu
        * dx
        - (energy_e.dpf(pf_prev, u_n) + energy_e.dpf_dpf(pf_prev, u_n) * (pf - pf_prev))
        * eta_mu
        * dx
        - (
            energy_h.dpf(pf_prev, u_n, p_n)
            + energy_h.dpf_dpf(pf_prev, u_n, p_n) * (pf - pf_prev)
        )
        * eta_mu
        * dx
    )
    F_e = (
        energy_e.deps(pf_n, u, eta_u) * dx
        - (alpha(pf_n) * p_n) * div(eta_u) * dx
        - dot(f, eta_u) * dx
    )
    F_p = (
        (p / M(pf_n) + alpha(pf_n) * div(u_n)) * eta_p * dx
        - (p_old / M(pf_old) + alpha(pf_old) * div(u_old)) * eta_p * dx
        + dt * div(q) * eta_p * dx
        - dt * S_f * eta_p * dx
    )
    F_q = dot(q, eta_q) / k(pf_n) * dx - p * div(eta_q) * dx

    F_ch = F_pf + F_mu
    F_fl = F_p + F_q
    A_ch, L_ch = df.lhs(F_ch), df.rhs(F_ch)
    A_e, L_e = df.lhs(F_e), df.rhs(F_e)
    A_fl, L_fl = df.lhs(F_fl), df.rhs(F_fl)

    # Output
    pf_out, mu_out = ch_n.split()
    path = output_path
    output_file_pf = df.File(
        path + "phasefield.pvd",
        "compressed",
    )
    output_file_pf << pf_out

    output_file_mu = df.File(
        path + "mu.pvd",
        "compressed",
    )
    output_file_mu << mu_out

    _, p_out = fl_n.split()
    output_file_p = df.File(
        path + "pressure.pvd",
        "compressed",
    )
    output_file_p << p_out

    output_file_u = df.File(
        path + "displacement.pvd",
        "compressed",
    )
    output_file_u << u_n

    # Time stepping
    t = 0.0
    if log is not None:
        iter_file = open(output_path + log + "_iter.txt", "w")
        time_file = open(output_path + log + "_time.txt", "w")

    t0 = time.time()
    total_iteration_count = 0
    for i in range(num_time_steps):
        # Set old time-step functions
        ch_old.assign(ch_n)
        u_old.assign(u_n)
        fl_old.assign(fl_n)

        # Update current time
        t += dt
        S_f.t = t
        f.t = t
        R.t = t

        tpre = time.time()
        iteration_count = 0
        for j in range(max_iter_split):
            iteration_count += 1
            u_prev.assign(u_n)
            fl_prev.assign(fl_n)

            # Solve ch
            for k in range(max_iter_inner_newton):
                ch_prev.assign(ch_n)
                df.solve(A_ch == L_ch, ch_n, bcs=[])
                increment_pf = sqrt(df.assemble((pf_n - pf_prev) ** 2 * dx))
                if verbose:
                    print(
                        f"Increment norm at time step {i} iteration {j} inner iteration {k}: {increment_pf}"
                    )
                if increment_pf < tol:
                    break

            # Solve elasticity
            df.solve(A_e == L_e, u_n, bcs=[bc_e])

            # Solve flow
            df.solve(A_fl == L_fl, fl_n, bcs=[bc_f])

            increment_total = sqrt(
                df.assemble(
                    (pf_n - pf_prev) ** 2 * dx
                    + (p_n - p_prev) ** 2 * dx
                    + (u_n - u_prev) ** 2 * dx
                )
            )
            if verbose:
                print(f"Norm at time step {i} iteration {j}: {increment_total}")
            if increment_total < tol:
                break

        tpost = time.time() - tpre
        total_iteration_count += iteration_count
        if log is not None:
            # Write log files
            iter_file.write(f"({i}, {iteration_count})\n")
            time_file.write(f"({i}, {tpost})\n")

        # Output
        if i % output_interval == 0:
            pf_out, mu_out = ch_n.split()
            output_file_pf << pf_out
            output_file_mu << mu_out
            _, p_out = fl_n.split()
            output_file_p << p_out
            output_file_u << u_n
    tfin = time.time() - t0
    if log is not None:
        time_file.write(f"Total time spent: {tfin}")
        iter_file.write(f"Total number of iterations: {total_iteration_count}")
        time_file.close()
        iter_file.close()
