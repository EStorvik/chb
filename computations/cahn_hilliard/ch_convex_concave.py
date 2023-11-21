import random

import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

import time

import chb


def convex_concave_solver(
    dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME, initialcondition
):
    # Define test/trial functions
    u = TrialFunction(ME)
    eta_pf, eta_mu = TestFunctions(ME)

    # Define functions
    un = Function(ME)  # current solution
    uold = Function(ME)  # solution from previous time step
    uprev = Function(ME)  # Solution from previous iterative linearization step
    uprevnest = Function(ME)

    # Split mixed functions
    pf, mu = split(u)
    pfn, mun = split(un)
    pfold, muold = split(uold)
    pfprev, muprev = split(uprev)
    pfprevnest, muprevnest = split(uprevnest)

    # Define initial value
    un.interpolate(initialcondition)
    # uold.interpolate(u_init)
    # uprev.interpolate(u_init)

    # Define variational problem
    Fpf = (
        pf * eta_pf * dx
        - pfold * eta_pf * dx
        + dt * m * dot(grad(mu), grad(eta_pf)) * dx
    )
    Fmu = (
        mu * eta_mu * dx
        - gamma * ell * dot(grad(pf), grad(eta_mu)) * dx
        - (gamma / ell)
        * (
            psi.cprime(pfprevnest) * eta_mu * dx
            + psi.cdoubleprime(pfprevnest) * (pf - pfprevnest) * eta_mu * dx
            - psi.eprime(pfprev) * eta_mu * dx
            - psi.edoubleprime(pfprev) * (pf - pfprev) * eta_mu * dx
        )
    )
    F = Fpf + Fmu

    a, L = lhs(F), rhs(F)

    # Output file
    file = File(f"/home/erlend/src/fenics/output/output_nested_ch.pvd", "compressed")

    # Time-stepping
    t = 0
    tic = time.time()
    for n in range(num_steps):
        # Update current time and previous time-step
        t += dt
        uold.assign(un)

        for i in range(max_iter):
            # Update previous iterate
            uprevnest.assign(un)

            for j in range(max_iter):
                # Update previous iterate
                uprev.assign(un)

                # Compute solution
                solve(a == L, un)

                # Compute increment
                increment_L2 = sqrt(assemble((un - uprev) ** 2 * dx))

                # Break the loop
                if increment_L2 < tol:
                    # print(f"Inner Newton terminated after {j} iterations at outer iterate {i} at timestep {n}.")
                    break

            # Compute increment
            increment_L2 = sqrt(assemble((un - uprevnest) ** 2 * dx))
            # print(f"Time step {n}, iteration {i}, increment: {increment_L2}.")

            # Break the loop
            if increment_L2 < tol:
                print(f"Outer Newton terminated after {i} iterations at timestep {n}.")
                break
        # file << (un.split()[0], t)
    print("Time elapsed: ", time.time() - tic)
    return un
    # Hold plot
    # plot(pfn)
    # plt.show()
    # interactive()


def newton_solver(
    dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME, initialcondition, output
):
    # Define test/trial Function
    u = TrialFunction(ME)
    eta_pf, eta_mu = TestFunctions(ME)

    # Define functions
    un = Function(ME)  # current solution
    uold = Function(ME)  # solution from previous time step
    uprev = Function(ME)  # Solution from previous iterative linearization step

    # Split mixed functions
    pf, mu = split(u)
    pfn, mun = split(un)
    pfold, muold = split(uold)
    pfprev, muprev = split(uprev)

    # Define initial value
    un.interpolate(initialcondition)
    # uold.interpolate(u_init)
    # uprev.interpolate(u_init)

    # Define variational problem
    Fpf = (
        pf * eta_pf * dx
        - pfold * eta_pf * dx
        + dt * m * dot(grad(mu), grad(eta_pf)) * dx
    )
    Fmu = (
        mu * eta_mu * dx
        - gamma * ell * dot(grad(pf), grad(eta_mu)) * dx
        - (gamma / ell)
        * (
            psi.prime(pfprev) * eta_mu * dx
            + psi.doubleprime(pfprev) * (pf - pfprev) * eta_mu * dx
        )
    )
    F = Fpf + Fmu

    a, L = lhs(F), rhs(F)

    # Output file
    file = File(output, "compressed")

    # Time-stepping
    t = 0
    toc = time.time()
    for n in range(num_steps):
        # Update current time and previous time-step
        t += dt
        uold.assign(un)

        for i in range(max_iter):
            # Update previous iterate
            uprev.assign(un)

            # Compute solution
            solve(a == L, un)

            # Compute increment
            increment_L2 = sqrt(assemble((un - uprev) ** 2 * dx))
            # print(f"Time step {n}, iteration {i}, increment: {increment_L2}.")

            # Break the loop
            if increment_L2 < tol:
                print(f"Newton terminated after {i} iterations at timestep {n}.")
                break
        file << (un.split()[0], t)
    print("Time elapsed: ", time.time() - toc)
    return un


# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Verbosity level
set_log_level(50)

dt = 1.0e-4  # time-step size
num_steps = 3  # number of time steps
T = dt * num_steps  # final time
m = 1  # mobility
ell = 5.0e-2  # regularization width
gamma = 1  # surface tension
max_iter = 300  # max Newton iterations
tol = 1e-6  # tolerance for Newton

# Define double-well potential
psi = chb.DoubleWellPotential()

# Create mesh and define function space
nx = ny = 64
mesh = UnitSquareMesh(nx, ny)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1 * P1)

initialcondition = chb.CrossInitialCondition

output = "/home/erlend/src/fenics/output/output_ch.pvd"

unn = convex_concave_solver(
    dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME, initialcondition
)
un = newton_solver(
    dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME, initialcondition, output
)

print(f"Difference in L2 norm of solutions: {sqrt(assemble((unn-un)**2*dx))}")
