import chb

import numpy as np

from mpi4py import MPI

from petsc4py import PETSc

from basix.ufl import element, mixed_element

import matplotlib.pyplot as plt

from dolfinx import mesh
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    functionspace,
    Function,
    Expression,
    assemble_scalar,
    form,
    locate_dofs_topological,
    dirichletbc,
)

from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points


from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)

from ufl import (
    Measure,
    TestFunction,
    TrialFunction,
    split,
    Constant,
    Identity,
    inner,
    grad,
    sym,
    dx,
    rhs,
    lhs,
    nabla_div,
)

# Spatial discretization
nx = ny = 32
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

# Define material parameters

# CH
ell = 0.1
gamma = 1
mobility = 1
doublewell = chb.energies.DoubleWellPotential_cutoff()

# Elasticity
# isotropic stiffness tensor
#stiffness_tensor = chb.elasticity.IsotropicStiffnessTensor(
#    lame_lambda_0=20, lame_mu_0=100, lame_lambda_1=0.1, lame_mu_1=1
#)
# heterogeneous and anisotropic stiffness tensor
stiffness_tensor = chb.elasticity.HeterogeneousStiffnessTensor()
swelling = chb.elasticity.Swelling(swelling_parameter=0.1, pf_ref=0)

# Biot
alpha = chb.biot.NonlinearBiotCoupling(alpha0=1, alpha1=0.1)

# Flow
permeability = 1
compressibility = chb.flow.NonlinearCompressibility(M0=1, M1=0.1)

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

# Test and trial functions
xi = TrialFunction(V)
eta = TestFunction(V)
pf, mu, u, theta, p = split(xi)
eta_pf, eta_mu, eta_u, eta_theta, eta_p = split(eta)


# Iteration functions
xi_n = Function(V)

xi_prev = Function(V)
pf_prev, mu_prev, u_prev, theta_prev, p_prev = xi_prev.split()

xi_old = Function(V)
pf_old, mu_old, u_old, theta_old, p_old = xi_old.split()


# Initial condtions
initialcondition_cross = chb.initialconditions.Cross(width = 0.3)
initialcondition = chb.initialconditions.halfnhalf
xi_n.sub(0).interpolate(initialcondition)
xi_n.sub(1).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi_n.sub(2).interpolate(lambda x: np.zeros((2, x.shape[1])))
xi_n.sub(3).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi_n.sub(4).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi_n.x.scatter_forward()

pf_n, mu_n, u_n, theta_n, p_n = xi_n.split()


# Boundary conditions
def boundary(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))

def boundary_left(x):
    return np.isclose(x[0], 0.0)

def boundary_right(x):
    return np.isclose(x[0], 1.0)

V_u = V.sub(2)
V_theta = V.sub(3)
facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary)
facets_left = mesh.locate_entities_boundary(msh, msh.topology.dim -1, boundary_left)
facets_right = mesh.locate_entities_boundary(msh, msh.topology.dim -1, boundary_right)
dofs_u = locate_dofs_topological(V_u, msh.topology.dim - 1, facets)
dofs_theta_left = locate_dofs_topological(V_theta, msh.topology.dim - 1, facets_left)
dofs_theta_right = locate_dofs_topological(V_theta, msh.topology.dim - 1, facets_right)

_, _, u_bc, theta_bc_left, _ = Function(V).split()
_, _, _, theta_bc_right, _ = Function(V).split()

u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
bc_u = dirichletbc(u_bc, dofs_u)

theta_bc_left.interpolate(lambda x: np.ones((1, x.shape[1])))
theta_bc_right.interpolate(lambda x: np.zeros((1, x.shape[1])))

bc_theta_left = dirichletbc(theta_bc_left, dofs_theta_left)
bc_theta_right = dirichletbc(theta_bc_right, dofs_theta_right)

# Linear variational forms
F_pf = (
    inner(pf - pf_old, eta_pf) * dx + dt * mobility * inner(grad(mu), grad(eta_pf)) * dx
)

F_mu = (
    inner(mu, eta_mu) * dx
    - gamma * ell * inner(grad(pf), grad(eta_mu)) * dx
    - gamma / ell
    * inner(
        (
            doublewell.cprime(pf_prev) + doublewell.cdoubleprime(pf_prev) * (pf - pf_prev)
            - doublewell.eprime(pf_old)
        ),
        eta_mu
    )
    * dx
    - (
        inner(
            0.5 *
            inner(
                stiffness_tensor.stress_prime(strain=sym(grad(u_old)) - swelling(pf_old), pf=pf_old),
                sym(grad(u_old)) - swelling(pf_old)
            )
            - inner(
                stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf_old),
                swelling.prime()
            ),
            eta_mu
        )
    )
    * dx
    - (
        inner(
            0.5 * compressibility.prime(pf_old) * (theta_old - alpha(pf_old) * nabla_div(u_old))**2
            - alpha.prime(pf_old) * compressibility(pf_old) * (theta - alpha(pf_old) * nabla_div(u)) * nabla_div(u_old),
            eta_mu
        )
        * dx
    )
)

F_u = (
    inner(
        stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf_old)
        - alpha(pf_old) * compressibility(pf_old) * (theta - alpha(pf_old) * nabla_div(u)) * Identity(2),
        sym(grad(eta_u))
    )
    * dx
)

F_theta = (
    inner(theta - theta_old, eta_theta) * dx + dt * permeability * inner(grad(p), grad(eta_theta)) * dx
)

F_p = (
    inner(p, eta_p) * dx - inner(compressibility(pf_old) * (theta - alpha(pf_old) * nabla_div(u)), eta_p) * dx              
)

F = F_pf + F_mu + F_u + F_theta + F_p


# Set up problem
a = lhs(F)
L = rhs(F)

problem = LinearProblem(a, L, bcs=[bc_u, bc_theta_left, bc_theta_right])


# Pyvista plot
viz = chb.visualization.PyvistaVizualization(V.sub(0), xi_n.sub(0), 0.0)

# Output file
output_file_pf = XDMFFile(MPI.COMM_WORLD, f"../output/chb_{ell}ell_pf.xdmf", "w")
output_file_p = XDMFFile(MPI.COMM_WORLD, f"../output/chb_{ell}ell_p.xdmf", "w")

output_file_pf.write_mesh(msh)
output_file_p.write_mesh(msh)


# Time stepping
t = 0.0


for i in range(num_time_steps):
    # Set old time-step functions
    xi_old.x.array[:] = xi_n.x.array
    xi_old.x.scatter_forward()
    pf_old, mu_old, u_old, theta_old, p_old = xi_old.split()
    # Update current time
    t += dt

    for j in range(max_iter):
        xi_prev.x.array[:] = xi_n.x.array
        xi_prev.x.scatter_forward()
        pf_prev, mu_prev, u_prev, theta_prev, p_prev = (
            xi_prev.split()
        )  # This seem only to be necessary for the computation of the L2-norm

        # Define the problem
        xi_n = problem.solve()
        xi_n.x.scatter_forward()
        pf_n, mu_n, u_n, theta_n, p_n = (
            xi_n.split()
        )  # This seem only to be necessary for the computation of the L2-norm

        increment = chb.util.l2norm(pf_n - pf_prev)
        print(f"Norm at time step {i} iteration {j}: {increment}")
        # print(f"Norms for pf, mu, u are: {chb.util.l2norm(pf_n-pf_prev)}, {chb.util.l2norm(mu_n-mu_prev)}, {chb.util.l2norm(u_n-u_prev)}")
        viz.update(xi_n.sub(0), t)
        if increment < tol:
            break

        # Update the plot window

    # Output
    pf_out, _, _, _, p_out = xi_n.split()
    output_file_pf.write_function(pf_out, t)
    output_file_p.write_function(p_out, t)


viz.final_plot(xi_n.sub(0))


def plot_along_line(u, msh, y=0.5, filename="line_data.npy"):
    # Create an array of x-coordinates along the line y=0.5
    x_coords = np.linspace(msh.geometry.x.min(), msh.geometry.x.max(), 100)
    y_coord = y
    points = np.array([[x, y_coord, 0] for x in x_coords])

    tree = bb_tree(msh, msh.geometry.dim)
    values = []

    for point in points:
        cell_candidates = compute_collisions_points(tree, point.T)
        cell = compute_colliding_cells(msh, cell_candidates, point).array
        assert len(cell) > 0
        first_cell = cell[0]
        values.append(u.eval(point, first_cell))

    # Save the x-coordinates and values to a numpy file
    np.save(filename, {"x_coords": x_coords, "values": values})

    plt.figure()
    plt.plot(x_coords, values, label=f"Solution at y={0.5}")

    plt.show()

plot_along_line(pf_n, msh=msh, filename=f"../output/line_data_{ell}ell.npy")

output_file_pf.close()
output_file_p.close()
