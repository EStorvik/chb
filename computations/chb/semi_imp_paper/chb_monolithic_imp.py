import os
from time import time

# Fix MPI/OFI finalization errors on macOS
os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

import numpy as np
import pandas as pd
from basix.ufl import element, mixed_element
from dolfinx import mesh
from dolfinx.fem import (
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from ufl import Identity, Measure, TestFunction, div, dx, grad, inner, split, sym

import chb

# Spatial discretization
nx = ny = 64
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

# Define material parameters

# CH
ell = 0.025
gamma = 1
mobility = 1
doublewell = chb.energies.SymmetricDoubleWellPotential_cutoff()

# Elasticity
# isotropic stiffness tensor
# stiffness_tensor = chb.elasticity.IsotropicStiffnessTensor(
#    lame_lambda_0=20, lame_mu_0=100, lame_lambda_1=0.1, lame_mu_1=1
# )
# heterogeneous and anisotropic stiffness tensor
interpolator = chb.interpolate.SymmetricStandardInterpolator()
stiffness_tensor = chb.elasticity.HeterogeneousStiffnessTensor(
    interpolator=interpolator
)
swelling = chb.elasticity.Swelling(swelling_parameter=1, pf_ref=0)

# Biot
alpha = chb.biot.NonlinearBiotCoupling(alpha0=1, alpha1=0.1, interpolator=interpolator)

# Flow
permeability = 1
compressibility = chb.flow.NonlinearCompressibility(
    M0=1, M1=0.1, interpolator=interpolator
)

# Time discretization
dt = 1.0e-3
num_time_steps = 100
T = dt * num_time_steps

# Nonlinear iteration parameters
max_iter = 20
tol = 1e-6


# Finite elements
P1 = element("Lagrange", msh.basix_cell(), 1)
P1U = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
ME = mixed_element([P1, P1, P1U, P1, P1])

# Function spaces
V = functionspace(msh, ME)

# Solution function
xi = Function(V)
pf, mu, u, theta, p = split(xi)

# Test function
eta = TestFunction(V)
eta_pf, eta_mu, eta_u, eta_theta, eta_p = split(eta)

# Solution function at previous time step
xi_old = Function(V)
pf_old, mu_old, u_old, theta_old, p_old = split(xi_old)


# Initial condtions
initialcondition_cross = chb.initialconditions.Cross(width=0.3)
initialcondition = chb.initialconditions.symmetrichalfnhalf
xi.sub(0).interpolate(initialcondition)
xi.sub(1).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi.sub(2).interpolate(lambda x: np.zeros((2, x.shape[1])))
xi.sub(3).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi.sub(4).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi.x.scatter_forward()


# Boundary conditions
def boundary(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)),
    )


def boundary_left(x):
    return np.isclose(x[0], 0.0)


def boundary_right(x):
    return np.isclose(x[0], 1.0)


V_u = V.sub(2)
V_p = V.sub(4)
facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary)
# facets_left = mesh.locate_entities_boundary(msh, msh.topology.dim -1, boundary_left)
# facets_right = mesh.locate_entities_boundary(msh, msh.topology.dim -1, boundary_right)
dofs_u = locate_dofs_topological(V_u, msh.topology.dim - 1, facets)
# dofs_p = locate_dofs_topological(V_p, msh.topology.dim - 1, facets)
# dofs_p_left = locate_dofs_topological(V_p, msh.topology.dim - 1, facets_left)
# dofs_p_right = locate_dofs_topological(V_p, msh.topology.dim - 1, facets_right)

_, _, u_bc, _, _ = Function(V).split()
# _, _, u_bc, _, p_bc_left = Function(V).split()
# _, _, _, _, p_bc_right = Function(V).split()

u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
bc_u = dirichletbc(u_bc, dofs_u)

# p_bc.interpolate(lambda x: np.zeros((1, x.shape[1])))
# bc_p = dirichletbc(p_bc, dofs_p)

# p_bc_left.interpolate(lambda x: np.ones((1, x.shape[1])))
# p_bc_right.interpolate(lambda x: np.zeros((1, x.shape[1])))

# bc_p_left = dirichletbc(p_bc_left, dofs_p_left)
# bc_p_right = dirichletbc(p_bc_right, dofs_p_right)

# Linear variational forms
F_pf = (
    inner(pf - pf_old, eta_pf) * dx + dt * mobility * inner(grad(mu), grad(eta_pf)) * dx
)

F_mu = (
    inner(mu, eta_mu) * dx
    - gamma * ell * inner(grad(pf), grad(eta_mu)) * dx
    - gamma
    / ell
    * inner((doublewell.cprime(pf) - doublewell.eprime(pf_old)), eta_mu)
    * dx
    - (
        inner(
            0.5
            * inner(
                stiffness_tensor.stress_prime(
                    strain=sym(grad(u)) - swelling(pf), pf=pf
                ),
                sym(grad(u)) - swelling(pf),
            )
            - inner(
                stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf),
                swelling.prime(),
            ),
            eta_mu,
        )
    )
    * dx
    - (
        inner(
            0.5 * compressibility.prime(pf) * (theta - alpha(pf) * div(u)) ** 2
            - alpha.prime(pf)
            * compressibility(pf)
            * (theta - alpha(pf) * div(u))
            * div(u),
            eta_mu,
        )
        * dx
    )
)

F_u = (
    inner(
        stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf)
        - alpha(pf) * compressibility(pf) * (theta - alpha(pf) * div(u)) * Identity(2),
        sym(grad(eta_u)),
    )
    * dx
)

F_theta = (
    inner(theta - theta_old, eta_theta) * dx
    + dt * permeability * inner(grad(p), grad(eta_theta)) * dx
)

F_p = (
    inner(p, eta_p) * dx
    - inner(compressibility(pf) * (theta - alpha(pf) * div(u)), eta_p) * dx
)

F = F_pf + F_mu + F_u + F_theta + F_p


# Set up nonlinear problem
problem = NonlinearProblem(F, xi, bcs=[bc_u])

# Set up Newton solver
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.max_it = 100
solver.rtol = 1e-6
# solver.convergence_criterion = "incremental"

# Pyvista plot
# viz = chb.visualization.PyvistaVizualization(V.sub(0), xi.sub(0), 0.0)

# Output file
filenamepath = "../output/chb_monolithic_imp_"
output_file_pf = XDMFFile(MPI.COMM_WORLD, filenamepath + f"{ell}ell_pf.xdmf", "w")
output_file_p = XDMFFile(MPI.COMM_WORLD, filenamepath + f"{ell}ell_p.xdmf", "w")

output_file_pf.write_mesh(msh)
output_file_p.write_mesh(msh)


# Energy
def energy_i(pf, dx):
    return gamma * (1 / ell * doublewell(pf) + ell / 2 * inner(grad(pf), grad(pf))) * dx


def energy_e(pf, u, dx):
    return (
        0.5
        * inner(
            stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf),
            sym(grad(u)) - swelling(pf),
        )
        * dx
    )


def energy_f(pf, u, theta, dx):
    return 0.5 * compressibility(pf) * (theta - alpha(pf) * div(u)) ** 2 * dx


def energyTotal(pf, u, theta, dx):
    return energy_i(pf, dx) + energy_e(pf, u, dx) + energy_f(pf, u, theta, dx)


t_vec = []
energy_int_vec = []
energy_el_vec = []
energy_fl_vec = []
energy_vec = []
iterations = []
times = []

# Time stepping
t = 0.0

for i in range(num_time_steps):
    # Set old time-step functions
    xi_old.x.array[:] = xi.x.array
    xi_old.x.scatter_forward()

    # Update current time
    t += dt

    tpre = time()
    # Solve the nonlinear problem
    n, converged = solver.solve(xi)

    print(f"Used {n} newton iteratons to converge at time step {i}.")

    tpost = time() - tpre
    # Update the plot window
    # viz.update(xi.sub(0), t)

    energy_int = assemble_scalar(form(energy_i(pf, dx=Measure("dx", domain=msh))))
    energy_el = assemble_scalar(form(energy_e(pf, u, dx=Measure("dx", domain=msh))))
    energy_fl = assemble_scalar(
        form(energy_f(pf, u, theta, dx=Measure("dx", domain=msh)))
    )
    energy_total = energyTotal(pf, u, theta, dx=Measure("dx", domain=msh))
    energy = assemble_scalar(form(energy_total))
    print(f"Energy at time step {i}: {energy}")

    t_vec.append(tpost)
    energy_int_vec.append(energy_int)
    energy_el_vec.append(energy_el)
    energy_fl_vec.append(energy_fl)
    energy_vec.append(energy)
    iterations.append(n)
    times.append(tpost)

    # Output
    pf_out, _, _, _, p_out = xi.split()
    output_file_pf.write_function(pf_out, t)
    output_file_p.write_function(p_out, t)


# viz.final_plot(xi.sub(0))

# Create log DataFrame and save to Excel
log_data = {
    "Time_Step": range(1, len(t_vec) + 1),
    "Iterations": iterations,
    "Time": t_vec,
    "Total_Energy": energy_vec,
    "Interface_Energy": energy_int_vec,
    "Elastic_Energy": energy_el_vec,
    "Fluid_Energy": energy_fl_vec,
}

log_df = pd.DataFrame(log_data)
log_filenamepath = "../output/log/chb_monolithic_imp.xlsx"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(log_filenamepath), exist_ok=True)

try:
    # Try to save as Excel file (requires openpyxl)
    log_df.to_excel(log_filenamepath, index=False, sheet_name="Simulation_Log")
    print(f"Log data saved to Excel file: {log_filenamepath}")
except ImportError:
    # Fallback to CSV if openpyxl is not available
    csv_path = log_filenamepath.replace(".xlsx", ".csv")
    log_df.to_csv(csv_path, index=False)
    print(f"Excel writer not available, log data saved to CSV: {csv_path}")
    print("Install openpyxl with: pip install openpyxl")

# Plot Total Energy
# plt.figure(figsize=(10, 6))
# plt.plot(t_vec, energy_vec, 'r-', linewidth=2, label="Total energy")
# plt.xlabel('Time')
# plt.ylabel('Total Energy')
# plt.title('Total Energy Evolution')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# Plot Interface Energy
# plt.figure(figsize=(10, 6))
# plt.plot(t_vec, energy_int_vec, 'b-', linewidth=2, label="Interface energy")
# plt.xlabel('Time')
# plt.ylabel('Interface Energy')
# plt.title('Interface Energy Evolution')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# Plot Elastic Energy
# plt.figure(figsize=(10, 6))
# plt.plot(t_vec, energy_el_vec, 'g-', linewidth=2, label="Elastic energy")
# plt.xlabel('Time')
# plt.ylabel('Elastic Energy')
# plt.title('Elastic Energy Evolution')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# Plot Fluid Energy
# plt.figure(figsize=(10, 6))
# plt.plot(t_vec, energy_fl_vec, 'm-', linewidth=2, label="Fluid energy")
# plt.xlabel('Time')
# plt.ylabel('Fluid Energy')
# plt.title('Fluid Energy Evolution')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# def plot_along_line(u, msh, y=0.5, filename="line_data.npy"):
#     # Create an array of x-coordinates along the line y=0.5
#     x_coords = np.linspace(msh.geometry.x.min(), msh.geometry.x.max(), 100)
#     y_coord = y
#     points = np.array([[x, y_coord, 0] for x in x_coords])

#     tree = bb_tree(msh, msh.geometry.dim)
#     values = []

#     for point in points:
#         cell_candidates = compute_collisions_points(tree, point.T)
#         cell = compute_colliding_cells(msh, cell_candidates, point).array
#         assert len(cell) > 0
#         first_cell = cell[0]
#         values.append(u.eval(point, first_cell))

#     # Save the x-coordinates and values to a numpy file
#     np.save(filename, {"x_coords": x_coords, "values": values})

#     plt.figure()
#     plt.plot(x_coords, values, label=f"Solution at y={0.5}")

#     plt.show()

# plot_along_line(xi.sub(0), msh=msh, filename=f"../output/line_data_{ell}ell.npy")

output_file_pf.close()
output_file_p.close()
