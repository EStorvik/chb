from math import sqrt

import dolfin as df
from dolfin import div, dot, grad, inner, sym
from ufl_legacy import Measure

import chb

df.parameters["form_compiler"]["quadrature_degree"] = 5

interpolator = chb.StandardInterpolator()

# Define material parameters

# CH
gamma = 5
ell = 2.0e-2
mobility = 1
doublewell = chb.DoubleWellPotential()

# Elasticity
swelling_parameter = 0.25
swelling = chb.Swelling(swelling_parameter)
stiffness = chb.HeterogeneousStiffnessTensor(interpolator=interpolator)


# Flow
compressibility0 = 1
compressibility1 = 1
M = chb.NonlinearCompressibility(
    compressibility0, compressibility1, interpolator=interpolator
)

permeability0 = 1
permeability1 = 1
k = chb.NonlinearPermeability(permeability0, permeability1, interpolator=interpolator)

# Coupling
alpha0 = 0
alpha1 = 0
alpha = chb.NonlinearBiotCoupling(alpha0, alpha1, interpolator=interpolator)

# Energies
energy_h = chb.CHBHydraulicEnergy(M, alpha)
energy_e = chb.CHBElasticEnergy(stiffness, swelling=swelling)

# Time discretization
dt = 1.0e-5
num_time_steps = 1000
T = dt * num_time_steps

# Nonlinear iteration parameters
max_iter = 50
tol = 1e-6

# Spatial discretization
nx = ny = 64
mesh = df.UnitSquareMesh(nx, ny)
dx = Measure("dx", domain=mesh)


# Finite elements
P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
RT0 = df.FiniteElement("RT", mesh.ufl_cell(), 1)
P0 = df.FiniteElement("DG", mesh.ufl_cell(), 0)
P2V = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)

# Function spaces
ME = df.MixedElement([P1, P1, P2V, RT0, P0])
V = df.FunctionSpace(mesh, ME)

# Test and trial functions
xi = df.TrialFunction(V)
eta = df.TestFunction(V)
pf, mu, u, q, p = df.split(xi)
eta_pf, eta_mu, eta_u, eta_q, eta_p = df.split(eta)


# Iteration functions
# CH
xi_n = df.Function(V)
pf_n, mu_n, u_n, q_n, p_n = df.split(xi_n)

xi_prev = df.Function(V)
pf_prev, mu_prev, u_prev, q_prev, p_prev = df.split(xi_prev)

xi_old = df.Function(V)
pf_old, mu_old, u_old, q_old, p_old = df.split(xi_old)


# Boundary conditions
def boundary(x, on_boundary):
    return on_boundary


# Elasticity
zero_e = df.Constant((0.0, 0.0))
bc_e = df.DirichletBC(V.sub(2), zero_e, boundary)

# Flow
zero_f = df.Constant(0.0)
bc_f = df.DirichletBC(V.sub(4), zero_f, boundary)


# Initial condtions
initialconditions = chb.HalfnhalfInitialConditions(variables=7)
xi_n.interpolate(initialconditions)


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
    - (
        energy_e.dpf(pf_prev, u_prev)
        + energy_e.dpf_dpf(pf_prev, u_prev) * (pf - pf_prev)
        + energy_e.dpf_deps(pf_prev, u_prev, u - u_prev)
    )
    * eta_mu
    * dx
    - (
        energy_h.dpf(pf_prev, u_prev, p_prev)
        + energy_h.dpf_prime(pf, u, p, pf_prev, u_prev, p_prev)
    )
    * eta_mu
    * dx
)
F_e = (
    energy_e.deps(pf_prev, u, eta_u) * dx
    + energy_e.deps_dpf(pf_prev, u_prev, eta_u) * (pf - pf_prev) * dx
    - (alpha(pf_prev) * p + alpha.prime(pf_prev) * p_prev * (pf - pf_prev))
    * div(eta_u)
    * dx
    - dot(f, eta_u) * dx
)
F_p = (
    (p / M(pf_prev) - p_prev * M.prime(pf_prev) * (pf - pf_prev) / (M(pf_prev) ** 2))
    * eta_p
    * dx
    + (alpha(pf_prev) * div(u) + alpha.prime(pf_prev) * div(u_prev) * (pf - pf_prev))
    * eta_p
    * dx
    - (p_old / M(pf_old) + alpha(pf_old) * div(u_old)) * eta_p * dx
    + dt * k(pf_prev) * div(q) * eta_p * dx
    - dt * S_f * eta_p * dx
)
F_q = (
    dot(q, eta_q) / k(pf_prev) * dx
    - (k.prime(pf_prev) * (pf - pf_prev)) / (k(pf_prev) ** 2) * dot(q_prev, eta_q) * dx
    - p * div(eta_q) * dx
)

F = F_pf + F_mu + F_e + F_p + F_q
A, L = df.lhs(F), df.rhs(F)

# Output
pf_out, mu_out, u_out, _, p_out = xi_n.split()
path = "/home/erlend/src/fenics/output/chb/halfnhalf/monolithic/"
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

output_file_p = df.File(
    path + "pressure.pvd",
    "compressed",
)
output_file_p << p_out

output_file_u = df.File(
    path + "displacement.pvd",
    "compressed",
)
output_file_u << u_out

# Time stepping
t = 0.0
for i in range(num_time_steps):
    # Set old time-step functions
    xi_old.assign(xi_n)

    # Update current time
    t += dt
    S_f.t = t
    f.t = t
    R.t = t

    for j in range(max_iter):
        xi_prev.assign(xi_n)

        df.solve(A == L, xi_n, bcs=[bc_f, bc_e])
        print(
            f"Norm at time step {i} iteration {j}: {sqrt(df.assemble((pf_n - pf_prev)**2*dx+ (p_n - p_prev)**2*dx + (u_n - u_prev)**2*dx))}"
        )
        if (
            sqrt(
                df.assemble(
                    (pf_n - pf_prev) ** 2 * dx
                    + (p_n - p_prev) ** 2 * dx
                    + (u_n - u_prev) ** 2 * dx
                )
            )
            < tol
        ):
            break

    # Output
    pf_out, mu_out, u_out, _, p_out = xi_n.split()
    output_file_pf << pf_out
    output_file_mu << mu_out
    output_file_p << p_out
    output_file_u << u_out
