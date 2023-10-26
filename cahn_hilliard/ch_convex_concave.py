
import random
from dolfin import *

import numpy as np
import matplotlib.pyplot as plt

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)

class DoubleWellPotential:
    def __init__(self):
        pass

    def __call__(self, pf):
        return pf**2*(1-pf)**2
    
    def prime(self, pf):
        return 2*pf*(1-pf)*(1-2*pf)
    
    def doubleprime(self, pf):
        return 2*(1-6*pf+6*pf**2)
    
    def c(self, pf):
        return (pf-0.5)**4+0.0625
    
    def cprime(self, pf):
        return 4*(pf-0.5)**3
    
    def cdoubleprime(self,pf):
        return 12*(pf-0.5)**2
    
    def e(self, pf):
        return 0.5*(pf-0.5)**2
    
    def eprime(self, pf):
        return pf-0.5
    
    def edoubleprime(self, pf):
        return 1


def convex_concave_solver(dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME):

    # Define test/trial functions
    u    = TrialFunction(ME)
    eta_pf, eta_mu  = TestFunctions(ME)

    # Define functions
    un   = Function(ME)  # current solution
    uold  = Function(ME)  # solution from previous time step
    uprev = Function(ME) # Solution from previous iterative linearization step
    uprevnest = Function(ME)

    # Split mixed functions
    pf, mu = split(u)
    pfn,  mun  = split(un)
    pfold, muold = split(uold)
    pfprev, muprev = split(uprev)
    pfprevnest, muprevnest = split(uprevnest)

    # Define initial value
    u_init = InitialConditions(degree=1)
    un.interpolate(u_init)
    # uold.interpolate(u_init)
    # uprev.interpolate(u_init)

    # Define variational problem
    Fpf = pf*eta_pf*dx-pfold*eta_pf*dx + dt*m*dot(grad(mu),grad(eta_pf))*dx
    Fmu = mu*eta_mu*dx-gamma*ell*dot(grad(pf),grad(eta_mu))*dx-(gamma/ell)*(psi.cprime(pfprevnest)*eta_mu*dx + psi.cdoubleprime(pfprevnest)*(pf-pfprevnest)*eta_mu*dx-psi.eprime(pfprev)*eta_mu*dx-psi.edoubleprime(pfprev)*(pf-pfprev)*eta_mu*dx)
    F = Fpf+Fmu

    a, L = lhs(F), rhs(F)

    # Output file
    file = File(f"output/output_nested_ch.pvd", "compressed")

    # Time-stepping
    t = 0
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
                increment_L2 = sqrt(assemble((un-uprev)**2*dx))

                # Break the loop
                if increment_L2 < tol:
                    # print(f"Inner Newton terminated after {j} iterations at outer iterate {i} at timestep {n}.")
                    break

            # Compute increment
            increment_L2 = sqrt(assemble((un-uprevnest)**2*dx))
            # print(f"Time step {n}, iteration {i}, increment: {increment_L2}.")

            # Break the loop
            if increment_L2 < tol:
                print(f"Outer Newton terminated after {i} iterations at timestep {n}.")
                break
        file << (un.split()[0], t)
    return un
    # Hold plot
    # plot(pfn)
    # plt.show()
    # interactive()

def newton_solver(dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME):

    # Define test/trial Function
    u    = TrialFunction(ME)
    eta_pf, eta_mu  = TestFunctions(ME)

    # Define functions
    un   = Function(ME)  # current solution
    uold  = Function(ME)  # solution from previous time step
    uprev = Function(ME) # Solution from previous iterative linearization step

    # Split mixed functions
    pf, mu = split(u)
    pfn,  mun  = split(un)
    pfold, muold = split(uold)
    pfprev, muprev = split(uprev)

    # Define initial value
    u_init = InitialConditions(degree=1)
    un.interpolate(u_init)
    # uold.interpolate(u_init)
    # uprev.interpolate(u_init)

    # Define variational problem
    Fpf = pf*eta_pf*dx-pfold*eta_pf*dx + dt*m*dot(grad(mu),grad(eta_pf))*dx
    Fmu = mu*eta_mu*dx-gamma*ell*dot(grad(pf),grad(eta_mu))*dx-(gamma/ell)*(psi.prime(pfprev)*eta_mu*dx + psi.doubleprime(pfprev)*(pf-pfprev)*eta_mu*dx)
    F = Fpf+Fmu

    a, L = lhs(F), rhs(F)

    # Output file
    file = File(f"output/output_ch.pvd", "compressed")

    # Time-stepping
    t = 0
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
            increment_L2 = sqrt(assemble((un-uprev)**2*dx))
            # print(f"Time step {n}, iteration {i}, increment: {increment_L2}.")

            # Break the loop
            if increment_L2 < tol:
                print(f"Newton terminated after {i} iterations at timestep {n}.")
                break
        file << (un.split()[0], t)

    return un

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Verbosity level
set_log_level(50)

dt = 5.0e-6        # time-step size
num_steps = 3     # number of time steps
T = dt* num_steps  # final time
m = 1              # mobility
ell = 5.0e-3       # regularization width
gamma = 1          # surface tension
max_iter = 300     # max Newton iterations
tol = 1e-6         # tolerance for Newton

# Define double-well potential
psi = DoubleWellPotential()

# Create mesh and define function space
nx = ny = 96
mesh = UnitSquareMesh(nx, ny)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1*P1)

unn = convex_concave_solver(dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME)
un = newton_solver(dt, num_steps, m, ell, gamma, max_iter, tol, psi, ME)

print(f"Difference in L2 norm of solutions: {sqrt(assemble((unn-un)**2*dx))}")