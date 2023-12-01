import chb

import dolfin as df

import dolfin as df

from dolfin import grad, dot, div
from math import sqrt

from ufl_legacy import Measure

import chb

df.parameters["form_compiler"]["quadrature_degree"] = 5

# Define material parameters

# CH
gamma = 1.0
ell = 2.0e-3
mobility = 1
doublewell = chb.DoubleWellPotential()

# Time discretization
dt = 1.0e-3
num_time_steps = 100
T = dt * num_time_steps

# Nonlinear iteration parameters
max_iter = 20
tol = 1e-6

# Spatial discretization
nx = ny = 128
mesh = df.UnitSquareMesh(nx, ny)
dx = Measure("dx", domain=mesh)


# Finite elements
P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

# Function spaces
ME = df.MixedElement([P1, P1])
V = df.FunctionSpace(mesh, ME)

# Test and trial functions
xi = df.TrialFunction(V)
eta = df.TestFunction(V)
pf, mu = df.split(xi)
eta_pf, eta_mu = df.split(eta)


# Iteration functions
# CH
xi_n = df.Function(V)
pf_n, mu_n = df.split(xi_n)

xi_prev = df.Function(V)
pf_prev, mu_prev = df.split(xi_prev)

xi_old = df.Function(V)
pf_old, mu_old = df.split(xi_old)


# Initial condtions
initialconditions = chb.CrossInitialConditions(delta=0.15, variables=2)
xi_n.interpolate(initialconditions)

# RHS
R = df.Constant(0.0)

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
    * (doublewell.prime(pf_prev) + doublewell.doubleprime(pf_prev) * (pf - pf_prev))
    * eta_mu
    * dx
)

F = F_pf + F_mu
A, L = df.lhs(F), df.rhs(F)

# Output
(
    pf_out,
    _,
) = xi_n.split()
path = "/home/erlend/src/fenics/output/ch/cross/newton/"
output_file_pf = df.File(
    path + "phasefield.pvd",
    "compressed",
)
output_file_pf << pf_out


# Time stepping
t = 0.0
for i in range(num_time_steps):
    # Set old time-step functions
    xi_old.assign(xi_n)

    # Update current time
    t += dt

    for j in range(max_iter):
        xi_prev.assign(xi_n)

        df.solve(A == L, xi_n)
        print(
            f"Norm at time step {i} iteration {j}: {sqrt(df.assemble((pf_n - pf_prev)**2*dx))}"
        )
        if sqrt(df.assemble((pf_n - pf_prev) ** 2 * dx)) < tol:
            break

    # Output
    pf_out, _ = xi_n.split()
    output_file_pf << pf_out
