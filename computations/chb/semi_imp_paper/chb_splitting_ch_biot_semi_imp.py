import os
from time import time

# Fix MPI/OFI finalization errors on macOS
os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

import matplotlib.pyplot as plt

import numpy as np
import pandas
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
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from ufl import (
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    div,
    dx,
    grad,
    inner,
    lhs,
    rhs,
    split,
    sym,
)

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


# dim = 2
# stiffness0np = np.zeros((dim, dim, dim, dim))

# stiffness0np[0, 0, 0, 0] = 100
# stiffness0np[0, 0, 1, 1] = 20
# stiffness0np[0, 0, 1, 0] = 0
# stiffness0np[0, 0, 0, 1] = 0

# stiffness0np[1, 1, 0, 0] = 20
# stiffness0np[1, 1, 1, 1] = 100
# stiffness0np[1, 1, 1, 0] = 0
# stiffness0np[1, 1, 0, 1] = 0

# stiffness0np[0, 1, 0, 0] = 0
# stiffness0np[0, 1, 1, 1] = 0
# stiffness0np[0, 1, 1, 0] = 100
# stiffness0np[0, 1, 0, 1] = 100

# stiffness0np[1, 0, 0, 0] = 0
# stiffness0np[1, 0, 1, 1] = 0
# stiffness0np[1, 0, 1, 0] = 100
# stiffness0np[1, 0, 0, 1] = 100

# stiffness_tensor = chb.elasticity.HeterogeneousStiffnessTensor(stiffness1=stiffness0np, stiffness0=stiffness0np,
#     interpolator=interpolator
# )

stiffness_tensor = chb.elasticity.HeterogeneousStiffnessTensor(
    interpolator=interpolator
)

swelling = chb.elasticity.Swelling(swelling_parameter=1, pf_ref=0.0)

# Biot
alpha = chb.biot.NonlinearBiotCoupling(alpha0=1, alpha1=0.1, interpolator=interpolator)

# Flow
permeability = 1
compressibility = chb.flow.NonlinearCompressibility(
    M0=1, M1=0.1, interpolator=interpolator
)

# Time discretization
dt = 1.0e-6
num_time_steps = 100
T = dt * num_time_steps

# Splitting iteration parameters
max_iter_split = 20
tol_split = 1e-6


# Finite elements
P1 = element("Lagrange", msh.basix_cell(), 1)
P1U = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
MEch = mixed_element([P1, P1])
MEb = mixed_element([P1U, P1, P1])

# Function spaces
Vch = functionspace(msh, MEch)
Vb = functionspace(msh, MEb)

# Solution and test functions
xiCH = Function(Vch)
etaCH = TestFunction(Vch)
pf, mu = split(xiCH)
eta_pf, eta_mu = split(etaCH)

xiB = TrialFunction(Vb)
etaB = TestFunction(Vb)
u, theta, p = split(xiB)
eta_u, eta_theta, eta_p = split(etaB)
xiB_n = Function(Vb)

# Solution function at previous time step
xiCH_old = Function(Vch)
pf_old, mu_old = split(xiCH_old)
xiB_old = Function(Vb)
u_old, theta_old, p_old = split(xiB_old)

# Solution function at previous iteration step
xiCH_prev = Function(Vch)
pf_prev, mu_prev = split(xiCH_prev)
xiB_prev = Function(Vb)
u_prev, theta_prev, p_prev = split(xiB_prev)


# Initial condtions
initialcondition_cross = chb.initialconditions.Cross(width=0.3)
initialcondition = chb.initialconditions.symmetrichalfnhalf
xiCH.sub(0).interpolate(initialcondition)
xiCH.sub(1).interpolate(lambda x: np.zeros((1, x.shape[1])))
xiCH.x.scatter_forward()

xiB_n.sub(0).interpolate(lambda x: np.zeros((2, x.shape[1])))
xiB_n.sub(1).interpolate(lambda x: np.zeros((1, x.shape[1])))
xiB_n.sub(2).interpolate(lambda x: np.zeros((1, x.shape[1])))
xiB_n.x.scatter_forward()
u_n, theta_n, p_n = split(xiB_n)


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


V_u = Vb.sub(0)
# V_p = Vb.sub(2)
facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary)
# facets_left = mesh.locate_entities_boundary(msh, msh.topology.dim -1, boundary_left)
# facets_right = mesh.locate_entities_boundary(msh, msh.topology.dim -1, boundary_right)
dofs_u = locate_dofs_topological(V_u, msh.topology.dim - 1, facets)
# dofs_p = locate_dofs_topological(V_p, msh.topology.dim - 1, facets)
# dofs_p_left = locate_dofs_topological(V_p, msh.topology.dim - 1, facets_left)
# dofs_p_right = locate_dofs_topological(V_p, msh.topology.dim - 1, facets_right)

u_bc, _, _ = Function(Vb).split()
# u_bc, _, p_bc_left = Function(Vb).split()
# _, _, p_bc_right = Function(Vb).split()

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
                    strain=sym(grad(u_old)) - swelling(pf_old), pf=pf_old
                ),
                sym(grad(u_old)) - swelling(pf_old),
            )
            - inner(
                stiffness_tensor.stress(
                    strain=sym(grad(u_prev)) - swelling(pf), pf=pf_old
                ),
                swelling.prime(),
            ),
            eta_mu,
        )
    )
    * dx
    - (
        inner(
            0.5
            * compressibility.prime(pf_old)
            * (theta_old - alpha(pf_old) * div(u_old)) ** 2
            - alpha.prime(pf)
            * compressibility(pf_old)
            * (theta_prev - alpha(pf) * div(u_prev))
            * div(u_prev),
            eta_mu,
        )
        * dx
    )
)

F_u = (
    inner(
        stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf_old)
        - alpha(pf)
        * compressibility(pf_old)
        * (theta - alpha(pf) * div(u))
        * Identity(2),
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
    - inner(compressibility(pf_old) * (theta - alpha(pf) * div(u)), eta_p) * dx
)

Fch = F_pf + F_mu
Fb = F_u + F_theta + F_p


# Set up non-linear problems
problemCH = NonlinearProblem(Fch, xiCH, bcs=[])

aB = lhs(Fb)
lB = rhs(Fb)
problemB = LinearProblem(
    aB, lB, bcs=[bc_u]
)  # bcs=[bc_u, bc_p_left, bc_p_right], bcs=[bc_u]

# Set up Newton solver
solverCH = NewtonSolver(MPI.COMM_WORLD, problemCH)
solverCH.max_it = 100
solverCH.rtol = 1e-6
# solver.convergence_criterion = "incremental"

# Pyvista plot
# viz = chb.visualization.PyvistaVizualization(Vch.sub(0), xiCH.sub(0), 0.0)
# vizP = chb.visualization.PyvistaVizualization(Vb.sub(2), xiB_n.sub(2), 0.0, "pressure")

# Output file
filenamepath = "../output/chb_splitting_ch_biot_semi_imp_"
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
energy_vec = []
energy_int_vec = []  # Interface energy
energy_el_vec = []  # Elastic energy
energy_fl_vec = []  # Fluid energy
iterations = []
times = []
# Time stepping
t = 0.0

for i in range(num_time_steps):
    # Set old time-step functions
    xiCH_old.x.array[:] = xiCH.x.array
    xiCH_old.x.scatter_forward()
    xiB_old.x.array[:] = xiB_n.x.array
    xiB_old.x.scatter_forward()

    # Update current time
    t += dt
    iteration = 0
    tpre = time()
    for j in range(max_iter_split):
        iteration += 1
        # Set previous iteration functions
        xiCH_prev.x.array[:] = xiCH.x.array
        xiCH_prev.x.scatter_forward()
        xiB_prev.x.array[:] = xiB_n.x.array
        xiB_prev.x.scatter_forward()

        # Solve the non-linear problems
        nCH, convergedCH = solverCH.solve(xiCH)
        xiB_n = problemB.solve()
        xiB_n.x.scatter_forward()
        u_n, theta_n, p_n = xiB_n.split()

        increment_split = chb.util.l2norm_3(pf - pf_prev, u_n - u_prev, p_n - p_prev)
        # print(f"Increment norm at time step {i} splitting step {j}: {increment_split}")

        if increment_split < tol_split:
            break

    tpost = time() - tpre
    # Update the plot window
    # viz.update(xiCH.sub(0), t)
    # vizP.update(xiB_n.sub(2), t)

    energy_total = energyTotal(pf, u_n, theta_n, dx=Measure("dx", domain=msh))
    energy = assemble_scalar(form(energy_total))

    # Calculate individual energy components
    energy_int_form = energy_i(pf, dx=Measure("dx", domain=msh))
    energy_el_form = energy_e(pf, u_n, dx=Measure("dx", domain=msh))
    energy_fl_form = energy_f(pf, u_n, theta_n, dx=Measure("dx", domain=msh))

    energy_int = assemble_scalar(form(energy_int_form))
    energy_el = assemble_scalar(form(energy_el_form))
    energy_fl = assemble_scalar(form(energy_fl_form))

    print(
        f"Total energy at time step {i} is {energy:.2f}. Interface: {energy_int:.2f}. Fluid: {energy_fl:.2f}. Elastic: {energy_el:.2f}."
    )

    t_vec.append(i)
    energy_vec.append(energy)
    energy_int_vec.append(energy_int)
    energy_el_vec.append(energy_el)
    energy_fl_vec.append(energy_fl)
    iterations.append(iteration)
    times.append(tpost)

    # Output
    pf_out, _ = xiCH.split()
    output_file_pf.write_function(pf_out, t)
    _, _, p_out = xiB_n.split()
    output_file_p.write_function(p_out, t)


# viz.final_plot(xiCH.sub(0))
# vizP.final_plot(xiB_n.sub(2))

# Create log DataFrame and save to Excel
log_data = {
    "Time_Step": range(1, len(t_vec) + 1),  # Enumerate time steps starting from 1
    "Iterations": iterations,
    "Time": t_vec,
    "Total_Energy": energy_vec,
    "Interface_Energy": energy_int_vec,
    "Elastic_Energy": energy_el_vec,
    "Fluid_Energy": energy_fl_vec,
}

log_df = pandas.DataFrame(log_data)
log_filenamepath = "../output/log/chb_splitting_ch_biot_semi_imp.xlsx"


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
plt.figure(figsize=(10, 6))
plt.plot(t_vec, energy_vec, "r-", linewidth=2, label="Total energy")
plt.plot(t_vec, energy_el_vec, "b-", linewidth=2, label="Elastic energy")
plt.plot(t_vec, energy_int_vec, "g-", linewidth=2, label="interface energy")
plt.plot(t_vec, energy_fl_vec, "o-", linewidth=2, label="fluid energy")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.title("Total Energy Evolution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

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

# plot_along_line(xiCH.sub(0), msh=msh, filename=f"../output/line_data_{ell}ell.npy")
# plot_along_line(xiB_n.sub(2), msh=msh, filename=f"../output/line_data_{ell}ell.npy")

output_file_pf.close()
output_file_p.close()
