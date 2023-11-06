import os
import random

import dolfin as df
from dolfin import assemble, div, dot, dx, grad, inner, sqrt, sym

# import chb


class InitialConditions(df.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + df.MPI.rank(df.MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.63 + 0.02 * (0.5 - random.random())
        values[1] = 0.0

    def value_shape(self):
        return (2,)


class StiffnessTensor:
    def __init__(self):
        self.c1111 = 1
        self.c1112 = 1
        self.c1122
        self.c1121
        pass


class DoubleWellPotential:
    def __init__(self):
        pass

    def __call__(self, pf):
        return pf**2 * (1 - pf) ** 2

    def prime(self, pf):
        return 2 * pf * (1 - pf) * (1 - 2 * pf)

    def doubleprime(self, pf):
        return 2 * (1 - 6 * pf + 6 * pf**2)


# Define material parameters
# CH
m = 1.0
gamma = 1.0
ell = 1.0e-2
psi = DoubleWellPotential()

# Elasticity
stiffness = 1.0

# Coupling coefficient
ksi = 1.0

# Define time discretization
dt = 1.0e-5
num_steps = 10
T = dt * num_steps

# Define Newton solver parameters
max_iter_newton = 20
tol_newton = 1.0e-8

# Define splitting iterations
max_iter_split = 20
tol_split = 1.0e-8

# Define mesh
nx = ny = 32
mesh = df.UnitSquareMesh(nx, ny)

# Define function space
P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P1V = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)


V_ch = df.FunctionSpace(mesh, P1 * P1)
V_e = df.FunctionSpace(mesh, P1V)


# Boundary conditions
zero = df.Constant((0.0, 0.0))


def boundary(x, on_boundary):
    return on_boundary


bc = df.DirichletBC(V_e, zero, boundary)


# Define test and trial functions
ch = df.TrialFunction(V_ch)
eta_pf, eta_mu = df.TestFunction(V_ch)
pf, mu = df.split(ch)

u = df.TrialFunction(V_e)
eta_u = df.TestFunction(V_e)


# Define Hierarchy of functions
ch_n = df.Function(V_ch)  # current solution
ch_old = df.Function(V_ch)  # solution from previous time step
ch_prev = df.Function(V_ch)  # Solution from previous iterative linearization step

pf_n, mu_n = df.split(ch_n)
pf_old, mu_old = df.split(ch_old)
pf_prev, mu_prev = df.split(ch_prev)

# Define initial conditions
initial_conditions = InitialConditions(degree=1)
ch_n.interpolate(initial_conditions)

u_n = df.Function(V_e)  # current solution
u_n.interpolate(zero)

u_prev = df.Function(V_e)


Fpf = (
    pf * eta_pf * dx - pf_old * eta_pf * dx + dt * m * dot(grad(mu), grad(eta_pf)) * dx
)
Fmu = (
    mu * eta_mu * dx
    - gamma * ell * dot(grad(pf), grad(eta_mu)) * dx
    - (gamma / ell)
    * (
        psi.prime(pf_prev) * eta_mu * dx
        + psi.doubleprime(pf_prev) * (pf - pf_prev) * eta_mu * dx
    )
    - ksi * stiffness * (div(u_n) - pf) * eta_mu * dx
)

F_ch = Fpf + Fmu
a_ch, L_ch = df.lhs(F_ch), df.rhs(F_ch)

F_e = stiffness * (
    inner(sym(grad(u)), sym(grad(eta_u))) * dx - ksi * pf_n * div(eta_u) * dx
)
a_e, L_e = df.lhs(F_e), df.rhs(F_e)


output_file = df.File(
    "/home/erlend/src/fenics/cdb_output/cl_test_output.pvd", "compressed"
)

# Time loop

for i in range(num_steps):
    ch_old.assign(ch_n)

    for j in range(max_iter_split):
        u_prev.assign(u_n)
        for k in range(max_iter_newton):
            ch_prev.assign(ch_n)
            df.solve(a_ch == L_ch, ch_n)
            increment_L2 = sqrt(assemble((ch_n - ch_prev) ** 2 * dx))
            if increment_L2 < tol_newton:
                print(
                    f"Newton terminated after {k} iterations at timestep {i}, split {j}."
                )
                break

        df.solve(a_e == L_e, u_n, bc)

        increment_u = sqrt(assemble((u_n - u_prev) ** 2 * dx))
        if increment_u < tol_split:
            print(f"Splitting method after {j} iterations at timestep {i}.")
    output_file << ch_n
