import chb

import numpy as np

from mpi4py import MPI

from petsc4py import PETSc

from basix.ufl import element, mixed_element

from dolfinx import mesh
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    functionspace,
    Function,
    assemble_scalar,
    form,
    locate_dofs_topological,
    dirichletbc,
)
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
nx = ny = 64
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

# Define material parameters

# CH
ell = 0.05
gamma = 1# / ell
mobility = 1 #* ell
doublewell = chb.energies.DoubleWellPotential()

# Elasticity
stiffness_tensor = chb.elasticity.IsotropicStiffnessTensor(
    lame_lambda_0=20, lame_mu_0=100, lame_lambda_1=0.1, lame_mu_1=1
)
swelling = chb.elasticity.Swelling(swelling_parameter=0.25)

# Biot
alpha = chb.biot.NonlinearBiotCoupling(alpha0=1, alpha1=1)

# Flow
permeability = chb.flow.NonlinearPermeability(k0=0.5, k1=1)
compressibility = chb.flow.NonlinearCompressibility(M0=1, M1=1)

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
P0 = element("DG", msh.basix_cell(), 0)
Q1 = element("RT", msh.basix_cell(), 1)
ME = mixed_element([P1, P1, P1U, P0, Q1])

# Function spaces
V = functionspace(msh, ME)

# Test and trial functions
xi = TrialFunction(V)
eta = TestFunction(V)
pf, mu, u, p, q = split(xi)
eta_pf, eta_mu, eta_u, eta_p, eta_q = split(eta)


# Iteration functions
xi_n = Function(V)

xi_prev = Function(V)
pf_prev, mu_prev, u_prev, p_prev, q_prev = xi_prev.split()

xi_old = Function(V)
pf_old, mu_old, u_old, p_old, q_old = xi_old.split()


# Initial condtions
initialcondition = chb.initialconditions.halfnhalf
xi_n.sub(0).interpolate(initialcondition)
xi_n.sub(1).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi_n.sub(2).interpolate(lambda x: np.zeros((2, x.shape[1])))
xi_n.sub(3).interpolate(lambda x: np.zeros((1, x.shape[1])))
xi_n.sub(4).interpolate(lambda x: np.zeros((2, x.shape[1])))
xi_n.x.scatter_forward()

pf_n, mu_n, u_n, p_n, q_n = xi_n.split()


# Boundary conditions
def boundary(x):
    return np.isclose(x[0], 0.0)


V_u = V.sub(2)
V_p = V.sub(3)
facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary)
dofs_u = locate_dofs_topological(V_u, msh.topology.dim - 1, facets)
dofs_p = locate_dofs_topological(V_p, msh.topology.dim - 1, facets)

_, _, u_bc, p_bc, _ = Function(V).split()
u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
bc_u = dirichletbc(u_bc, dofs_u)

p_bc.interpolate(lambda x: np.ones((1, x.shape[1])))

bc_p = dirichletbc(p_bc, dofs_p)


# Linear variational forms
F_pf = (
    inner(pf - pf_old, eta_pf) * dx + dt * mobility * inner(grad(mu), grad(eta_pf)) * dx
)

F_mu = (
    inner(mu, eta_mu) * dx
    - gamma * ell * inner(grad(pf), grad(eta_mu)) * dx
    - gamma
    / ell
    * inner(
        (
            doublewell.cprime(pf_prev)
            + doublewell.cdoubleprime(pf_prev) * (pf - pf_prev)
            - doublewell.eprime(pf_old)
        ),
        eta_mu,
    )
    * dx
    - (
        0.5
        * inner(
            inner(
                stiffness_tensor.stress_prime(
                    strain=sym(grad(u_old)) - swelling(pf_old),
                    pf=pf_old,
                ),
                sym(grad(u_old)) - swelling(pf_old),
            )
            - inner(
                stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf_old),
                swelling.prime(),
            ),
            eta_mu,
        )
    )
    * dx
    # - (
    #     inner(
    #         compressibility.prime(pf_old)
    #         * p_old**2
    #         / (2 * compressibility(pf_old) ** 2)
    #         - p_old * alpha.prime(pf_old) * nabla_div(u_old),
    #         eta_mu,
    #     )
    #     * dx
    # )
)

F_u = (
    inner(
        stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf_old)
        - alpha(pf_old) * p * Identity(2),
        sym(grad(eta_u)),
    )
    * dx
)

F_p = (
    inner(
        (
            p / compressibility(pf_old)
            + alpha(pf_old) * nabla_div(u)
            - p_old / compressibility(pf_old)
            - alpha(pf_old) * nabla_div(u_old)
        )
        / dt
        + nabla_div(q),
        eta_p,
    )
    * dx
)

F_q = inner(q / permeability(pf_old), eta_q) * dx - inner(p, nabla_div(eta_q)) * dx

F = F_pf + F_mu + F_u + F_p + F_q


# Set up problem
a = lhs(F)
L = rhs(F)

problem = LinearProblem(a, L, bcs=[bc_u, bc_p])


# Pyvista plot
viz = chb.visualization.PyvistaVizualization(V.sub(0), xi_n.sub(0), 0.0)

# Output file
output_file_pf = XDMFFile(MPI.COMM_WORLD, "../output/ch.xdmf", "w")
output_file_pf.write_mesh(msh)


# Time stepping
t = 0.0


for i in range(num_time_steps):
    # Set old time-step functions
    xi_old.x.array[:] = xi_n.x.array
    xi_old.x.scatter_forward()
    pf_old, mu_old, u_old, p_old, q_old = xi_old.split()
    # Update current time
    t += dt

    for j in range(max_iter):
        xi_prev.x.array[:] = xi_n.x.array
        xi_prev.x.scatter_forward()
        pf_prev, mu_prev, u_prev, p_old, q_old = (
            xi_prev.split()
        )  # This seem only to be necessary for the computation of the L2-norm

        # Define the problem
        xi_n = problem.solve()
        xi_n.x.scatter_forward()
        pf_n, mu_n, u_n, p_old, q_old = (
            xi_n.split()
        )  # This seem only to be necessary for the computation of the L2-norm

        increment = chb.util.l2norm(pf_n - pf_prev)
        print(f"Norm at time step {i} iteration {j}: {increment}")
        # print(f"Norms for pf, mu, u are: {chb.util.l2norm(pf_n-pf_prev)} {chb.util.l2norm(mu_n-mu_prev)}, {chb.util.l2norm(u_n-u_prev)}")
        viz.update(xi_n.sub(0), t)
        if increment < tol:
            break

        # Update the plot window

    # Output
    pf_out, _, _, _, _ = xi_n.split()
    output_file_pf.write_function(pf_out, t)


viz.final_plot(xi_n.sub(0))

output_file_pf.close()
