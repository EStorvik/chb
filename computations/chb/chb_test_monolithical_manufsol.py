import dolfin as df

from dolfin import grad, dot, div
from math import sqrt

from ufl_legacy import Measure

import chb

df.parameters["form_compiler"]["quadrature_degree"] = 5

interpolator = chb.StandardInterpolator()

# Define material parameters

# CH
gamma = 1.0
ell = 1.0e-2
mobility = 1
doublewell = chb.DoubleWellPotential()

# Elasticity
stiffness = chb.HeterogeneousStiffnessTensor(interpolator=chb.UnboundedInterpolator())
swelling = 0.3

# Flow
compressibility0 = 1
compressibility1 = 0.5
M = chb.NonlinearCompressibility(
    compressibility0, compressibility1, interpolator=chb.UnboundedInterpolator()
)

permeability0 = 1
permeability1 = 0.5
k = chb.NonlinearPermeability(
    permeability0, permeability1, interpolator=chb.UnboundedInterpolator()
)

# Coupling
alpha0 = 1
alpha1 = 0.5
alpha = chb.NonlinearBiotCoupling(
    alpha0, alpha1, interpolator=chb.UnboundedInterpolator()
)

# Energies
energy_h = chb.CHBHydraulicEnergy(M, alpha)
energy_e = chb.CHBElasticEnergy(stiffness, swelling)

# Manufactured Solution
manufsol = chb.CHBManufacturedSolution(
    doublewell=doublewell,
    mobility=mobility,
    gamma=gamma,
    ell=ell,
    M=M,
    alpha=alpha,
    kappa=k,
    stiffness=stiffness,
    swelling=swelling,
)

# Time discretization
dt = 0.0001
num_time_steps = 10
T = dt * num_time_steps

# Nonlinear iteration parameters
max_iter = 20
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

# CH
gradmu0 = df.Expression(manufsol.gradmu0_out(), degree=4, t=0.0)
gradmu1 = df.Expression(manufsol.gradmu1_out(), degree=4, t=0.0)
gradmu2 = df.Expression(manufsol.gradmu2_out(), degree=4, t=0.0)
gradmu3 = df.Expression(manufsol.gradmu3_out(), degree=4, t=0.0)

gradpf0 = df.Expression(manufsol.gradpf0_out(), degree=4, t=0.0)
gradpf1 = df.Expression(manufsol.gradpf1_out(), degree=4, t=0.0)
gradpf2 = df.Expression(manufsol.gradpf2_out(), degree=4, t=0.0)
gradpf3 = df.Expression(manufsol.gradpf3_out(), degree=4, t=0.0)

# Setup for Neumann data
boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)


class BoundaryX0(df.SubDomain):
    tol = 1e-14

    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[0], 0, tol)


bx0 = BoundaryX0()
bx0.mark(boundary_markers, 0)


class BoundaryX1(df.SubDomain):
    tol = 1e-14

    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[0], 1, tol)


bx1 = BoundaryX1()
bx1.mark(boundary_markers, 1)


class BoundaryY0(df.SubDomain):
    tol = 1e-14

    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[1], 0, tol)


by0 = BoundaryY0()
by0.mark(boundary_markers, 2)


class BoundaryY1(df.SubDomain):
    tol = 1e-14

    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[1], 1, tol)


by1 = BoundaryY1()
by1.mark(boundary_markers, 3)

ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)


# Initial condtions
# CH
# initialconditions = chb.RandomInitialConditions()
# ic = df.Constant()
# xi_n = interpolate(

# # Elasticity
# u_n.interpolate(zero_e)

# # Flow
# p_n.interpolate(zero_f)

# RHS
R = df.Expression(manufsol.R_out(), degree=4, t=0.0)
f = df.Expression((manufsol.f0_out(), manufsol.f1_out()), degree=4, t=0.0)
S_f = df.Expression(manufsol.S_f_out(), degree=4, t=0.0)

# Linear variational forms
F_pf = (
    (pf - pf_old) * eta_pf * dx
    + dt * mobility * dot(grad(mu), grad(eta_pf)) * dx
    - dt * R * eta_pf * dx
    - dt
    * (
        gradmu0 * eta_pf * ds(0)
        + gradmu1 * eta_pf * ds(1)
        + gradmu2 * eta_pf * ds(2)
        + gradmu3 * eta_pf * ds(3)
    )
)
F_mu = (
    mu * eta_mu * dx
    - gamma * ell * dot(grad(pf), grad(eta_mu)) * dx
    - gamma
    / ell
    * (doublewell.prime(pf_prev) + doublewell.doubleprime(pf_prev) * (pf - pf_prev))
    * eta_mu
    * dx
    - (energy_e.dpf(pf_prev, u_prev) + energy_e.dpf_prime(pf, u, pf_prev, u_prev))
    * eta_mu
    * dx
    - (
        energy_h.dpf(pf_prev, u_prev, p_prev)
        + energy_h.dpf_prime(pf, u, p, pf_prev, u_prev, p_prev)
    )
    * eta_mu
    * dx
    + gamma
    * ell
    * (
        gradpf0 * eta_mu * ds(0)
        + gradpf1 * eta_mu * ds(1)
        + gradpf2 * eta_mu * ds(2)
        + gradpf3 * eta_mu * ds(3)
    )
)
F_e = (
    energy_e.du(pf_prev, u, eta_u) * dx
    + energy_e.dpfdu(pf, pf_prev, u_prev, eta_u) * dx
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
pf_out, _, u_out, _, p_out = xi_n.split()
path = "/home/erlend/src/fenics/output/chb/manufsol/monolithic/"
output_file_pf = df.File(
    path + "phasefield.pvd",
    "compressed",
)
output_file_pf << pf_out

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
    gradmu0.t = t
    gradmu1.t = t
    gradmu2.t = t
    gradmu3.t = t

    for j in range(max_iter):
        xi_prev.assign(xi_n)

        df.solve(A == L, xi_n, bcs=[bc_f, bc_e])
        print(
            f"Norm at time step {i} iteration {j}: {sqrt(df.assemble((pf_n - pf_prev)**2*dx))}"
        )
        if sqrt(df.assemble((pf_n - pf_prev) ** 2 * dx)) < tol:
            break

    # Output
    pf_out, _, u_out, _, p_out = xi_n.split()
    output_file_pf << pf_out
    output_file_p << p_out
    output_file_u << u_out
