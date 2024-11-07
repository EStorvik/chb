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
from dolfinx.fem.petsc import LinearProblem

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
)

import chb.elasticity.swelling


# Spatial discretization
nx = ny = 64
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

# Define material parameters

# CH
ell = 0.02
gamma = 1 / ell
mobility = ell
doublewell = chb.DoubleWellPotential()

# Elasticity
stiffness_tensor = chb.elasticity.IsotropicStiffnessTensor(
    lame_lambda_0=20, lame_lambda_1=0.1, lame_mu_0=100, lame_mu_1=1
)
swelling = chb.elasticity.swelling.Swelling(swelling_parameter=0.25)

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
ME = mixed_element([P1, P1, P1U])

# Function spaces
V = functionspace(msh, ME)

# Test and trial functions
xi = TrialFunction(V)
eta = TestFunction(V)
pf, mu, u = split(xi)
eta_pf, eta_mu, eta_u = split(eta)


# Iteration functions
xi_n = Function(V)
pf_n, mu_n, u_n = xi_n.split()

xi_prev = Function(V)
pf_prev, mu_prev, u_prev = xi_prev.split()

xi_old = Function(V)
pf_old, mu_old, u_old = xi_old.split()


# Initial condtions
initialcondition = chb.initialconditions.halfnhalf
xi_n.sub(0).interpolate(initialcondition)
xi_n.x.scatter_forward()


# Boundary conditions
def boundary(x):
    return np.isclose(x[0], 0.0)


V_u = V.sub(2)
facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary)
dofs = locate_dofs_topological(V_u, msh.topology.dim - 1, facets)

_, _, u_bc = Function(V).split()
u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
bc = dirichletbc(u_bc, dofs)


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
)

F_u = (
    inner(
        stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf_old),
        sym(grad(eta_u)),
    )
    * dx
)
F = F_pf + F_mu + F_u

a = lhs(F)
L = rhs(F)


# Pyvista plot
viz = chb.visualization.PyvistaVizualization(V.sub(0), xi_n.sub(0), 0.0)

# Output file
output_file_pf = XDMFFile(MPI.COMM_WORLD, "../output/ch.xdmf", "w")
output_file_pf.write_mesh(msh)


# Time stepping
t = 0.0

# Prepare viewer for plotting the solution during the computation


for i in range(num_time_steps):
    # Set old time-step functions
    xi_old.x.array[:] = xi_n.x.array
    xi_old.x.scatter_forward()

    # Update current time
    t += dt

    for j in range(max_iter):
        xi_prev.x.array[:] = xi_n.x.array
        xi_prev.x.scatter_forward()
        pf_prev, _, _ = (
            xi_prev.split()
        )  # This seem only to be necessary for the computation of the L2-norm

        # Define the problem
        problem = LinearProblem(a, L, bcs=[bc])
        xi_n = problem.solve()
        pf_n, _, _ = (
            xi_n.split()
        )  # This seem only to be necessary for the computation of the L2-norm
        xi_n.x.scatter_forward()
        increment = chb.util.l2norm(pf_n - pf_prev)
        print(f"Norm at time step {i} iteration {j}: {increment}")
        # print(f"Norms for pf, mu, u are: {chb.util.l2norm(pf_n-pf_prev)} {chb.util.l2norm(mu_n-mu_prev)}, {chb.util.l2norm(u_n-u_prev)}")
        viz.update(xi_n.sub(0), t)
        if increment < tol:
            break

        # Update the plot window

    # Output
    pf_out, _, _ = xi_n.split()
    output_file_pf.write_function(pf_out, t)


viz.final_plot(xi_n.sub(0))

output_file_pf.close()
