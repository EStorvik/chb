import dolfinx
from basix.ufl import element, mixed_element
from dolfinx import mesh
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from mpi4py import MPI
from ufl import TestFunction, TrialFunction, dx, grad, inner, lhs, rhs, split

import chb

print(dolfinx.__version__)


# Define material parameters

# CH
gamma = 1.0
ell = 5.0e-2
mobility = 1
doublewell = chb.energies.DoubleWellPotential()

# Time discretization
dt = 1.0e-3
num_time_steps = 10
T = dt * num_time_steps

# Nonlinear iteration parameters
max_iter = 20
tol = 1e-6

# Spatial discretization
nx = ny = 64
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)


# Finite elements
P1 = element("Lagrange", msh.basix_cell(), 1)
ME = mixed_element([P1, P1])

# Function spaces
V = functionspace(msh, ME)

# Test and trial functions
xi = TrialFunction(V)
eta = TestFunction(V)
pf, mu = split(xi)
eta_pf, eta_mu = split(eta)


# Iteration functions
# CH
xi_n = Function(V)
pf_n, mu_n = xi_n.split()

xi_prev = Function(V)
pf_prev, mu_prev = xi_prev.split()

xi_old = Function(V)
pf_old, mu_old = xi_old.split()


# Initial condtions
initialcondition = chb.initialconditions.halfnhalf
xi_n.sub(0).interpolate(initialcondition)
xi_n.x.scatter_forward()

# Linear variational forms
F_pf = (
    inner(pf - pf_old, eta_pf) * dx + dt * mobility * inner(grad(mu), grad(eta_pf)) * dx
)
F_mu = (
    inner(mu, eta_mu) * dx
    - gamma * ell * inner(grad(pf), grad(eta_mu)) * dx
    - gamma
    / ell
    * (doublewell.prime(pf_prev) + doublewell.doubleprime(pf_prev) * (pf - pf_prev))
    * eta_mu
    * dx
)
F = F_pf + F_mu

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
        pf_prev, _ = (
            xi_prev.split()
        )  # This seem only to be necessary for the computation of the L2-norm

        # Define the problem
        problem = LinearProblem(a, L)
        xi_n = problem.solve()
        pf_n, _ = (
            xi_n.split()
        )  # This seem only to be necessary for the computation of the L2-norm
        xi_n.x.scatter_forward()
        increment = chb.util.l2norm(pf_n - pf_prev)
        print(f"Norm at time step {i} iteration {j}: {increment}")

        viz.update(xi_n.sub(0), t)
        if increment < tol:
            break

        # Update the plot window

    # Output
    pf_out, _ = xi_n.split()
    output_file_pf.write_function(pf_out, t)


viz.final_plot(xi_n.sub(0))

output_file_pf.close()
