import chb

import numpy as np

from mpi4py import MPI

from petsc4py import PETSc

from basix.ufl import element, mixed_element

from dolfinx import mesh, plot
from dolfinx.io import XDMFFile
from dolfinx.fem import functionspace, Function, assemble_scalar, form
from dolfinx.fem.petsc import LinearProblem

from ufl import Measure, TestFunction, TrialFunction, split, Constant, inner, grad, dx, rhs, lhs


# Pyvista
try:
    import pyvista as pv
    import pyvistaqt as pvqt

    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

have_pyvista = False
# Define material parameters

# CH
gamma = 1.0
ell = 1.0e-1
mobility = 1
doublewell = chb.DoubleWellPotential()

# Time discretization
dt = 1.0e-3
num_time_steps = 10
T = dt * num_time_steps

# Nonlinear iteration parameters
max_iter = 20
tol = 1e-6

# Spatial discretization
nx = ny = 32
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
#dx = Measure("dx", domain=msh)


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
F_pf = inner(pf-pf_old, eta_pf) * dx + dt * mobility * inner(grad(mu), grad(eta_pf)) * dx
F_mu = inner(mu, eta_mu) * dx - gamma * ell * inner(grad(pf),grad(eta_mu)) * dx - gamma/ell * (doublewell.prime(pf_prev) + doublewell.doubleprime(pf_prev) * (pf-pf_prev))* eta_mu * dx
F = F_pf+F_mu

a = lhs(F)
L = rhs(F)



# Output file
output_file_pf = XDMFFile(MPI.COMM_WORLD, "../output/ch.xdmf", "w")
output_file_pf.write_mesh(msh)


# Time stepping
t = 0.0

# Prepare viewer for plotting the solution during the computation
if have_pyvista:
    V0, dofs = V.sub(0).collapse()
    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = xi_n.x.array[dofs].real
    grid.set_active_scalars("c")

    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[0, 1])
    p.view_xy(True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

for i in range(num_time_steps):
    # Set old time-step functions
    xi_old.x.array[:] = xi_n.x.array
    xi_old.x.scatter_forward()

    # Update current time
    t += dt

    for j in range(max_iter):
        xi_prev.x.array[:] = xi_n.x.array
        xi_prev.x.scatter_forward()
        pf_prev,_ = xi_prev.split()

        # Define the problem
        problem = LinearProblem(a, L)
        xi_n = problem.solve()
        pf_n, _ = xi_n.split()
        print(np.sum(xi_n.x.array[:]-xi_prev.x.array[:]))
        xi_n.x.scatter_forward()
        increment = chb.util.l2norm(pf_n-pf_prev)
        # print(
        #     f"Norm at time step {i} iteration {j}: {increment}"
        # )

        
        # Update the plot window
        if have_pyvista:
            p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
            grid.point_data["c"] = xi_n.x.array[dofs].real
            p.app.processEvents()
        if increment < tol:
            break

    # Output
    pf_out, _ = xi_n.split()
    output_file_pf.write_function(pf_out, t)


# Update ghost entries and plot
if have_pyvista:
    xi_n.x.scatter_forward()
    grid.point_data["c"] = xi_n.x.array[dofs].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "c.png"
    pv.plot(grid, show_edges=True, screenshot=screenshot)

output_file_pf.close()
