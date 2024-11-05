import dolfin as df
from dolfin import assemble, div, dot, dx, grad, inner, sqrt, sym

import chb

# Verbosity level
df.set_log_level(50)

# Flow material parameters
compressibility = 1.0e11
permeability = 1.0e-13

# Elasticity parameters
lame_mu = 41.667e9
lame_lambda = 27.778e9
K_dr = lame_mu + lame_lambda

# Coupling coefficients
alpha = 1

# Time discretization
t = 0
dt = 0.1
num_time_steps = 10
T = dt * num_time_steps

# Splitting parameters
num_split_steps = 50
tol_split = 1e-6
stabilization = alpha**2 / (2 * K_dr)

# Spatial discretization
nx = ny = 64
mesh = df.UnitSquareMesh(nx, ny)

# Finite elements
P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P2V = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)

# Function spaces
V_f = df.FunctionSpace(mesh, P1)
V_e = df.FunctionSpace(mesh, P2V)

# Test and trial functions
p = df.TrialFunction(V_f)
q = df.TestFunction(V_f)

u = df.TrialFunction(V_e)
v = df.TestFunction(V_e)

# Iteration functions
p_n = df.Function(V_f)
p_old = df.Function(V_f)
p_prev = df.Function(V_f)

u_n = df.Function(V_e)
u_old = df.Function(V_e)
u_prev = df.Function(V_e)

# Initial conditions
p_n.interpolate(df.Constant(0.0))
u_n.interpolate(df.Constant((0.0, 0.0)))


# Boundary conditions
def boundary(x, on_boundary):
    return on_boundary


# Pressure
zero_p = df.Constant(0.0)
bc_f = df.DirichletBC(V_f, zero_p, boundary)

# Displacement
zero_u = df.Constant((0.0, 0.0))
bc_e = df.DirichletBC(V_e, zero_u, boundary)

# RHS
p_ref = 1.0e-12
manufsol = chb.RHSManufacturedBiot(
    alpha, compressibility, permeability, lame_mu, lame_lambda, p_ref, t
)
S_f = manufsol.S_f
f = manufsol.f


# Stress tensor
def stress(u):
    return 2 * lame_mu * sym(grad(u)) + lame_lambda * div(u) * df.Identity(2)


# Bilinear forms

# Flow
F_f = (
    (p - p_old) / compressibility * q * dx
    + alpha * div(u_n - u_old) * q * dx
    + dt * permeability * dot(grad(p), grad(q)) * dx
    + stabilization * (p - p_prev) * q * dx
    - dt * S_f * q * dx
)
A_f, L_f = df.lhs(F_f), df.rhs(F_f)

# Elasticity
F_e = (
    2 * lame_mu * inner(sym(grad(u)), sym(grad(v))) * dx
    + lame_lambda * div(u) * div(v) * dx
    - alpha * p_n * div(v) * dx
    - dot(f, v) * dx
)
A_e, L_e = df.lhs(F_e), df.rhs(F_e)

output_file_p = df.File(
    "/home/erlend/src/fenics/output/biot/manufactured_solution_pressure.pvd",
    "compressed",
)
output_file_u = df.File(
    "/home/erlend/src/fenics/output/biot/manufactured_solution_displacement.pvd",
    "compressed",
)

output_file_p << p_n
output_file_u << u_n

# Time loop
for i in range(num_time_steps):
    p_old.assign(p_n)
    u_old.assign(u_n)
    t += dt
    S_f.t = t
    f.t = t

    for j in range(num_split_steps):
        p_prev.assign(p_n)

        # Solve flow subproblem
        df.solve(A_f == L_f, p_n, bc_f)
        df.solve(A_e == L_e, u_n, bc_e)

        increment = sqrt(assemble((p_n - p_prev) ** 2 * dx)) / sqrt(
            assemble(p_n**2 * dx)
        )
        print(f"Solved at splitting {j} time step {i} with increment {increment}:")

        if increment < tol_split:
            print(f"Tolerance reached")
            break

    output_file_p << p_n
    output_file_u << u_n
