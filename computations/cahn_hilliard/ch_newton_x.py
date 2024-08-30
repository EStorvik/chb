# import chb

import numpy as np

from mpi4py import MPI

from petsc4py import PETSc

from basix.ufl import element, mixed_element

from dolfinx import mesh, plot
from dolfinx.io import XDMFFile
from dolfinx.fem import functionspace, Function, assemble_scalar, form
from dolfinx.fem.petsc import LinearProblem

from ufl import Measure, TestFunction, TrialFunction, split, Constant, inner, grad, dx

try:
    import pyvista as pv
    import pyvistaqt as pvqt

    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

# doublewell
"""Classic double well potential."""


class DoubleWellPotential:
    """Classic double well potential. Psi = pf^2 (1 - pf)^2."""

    def __init__(self, scaling: float = 1.0) -> None:
        """Initialize the double well potential.

        Args:
            scaling (float, optional): Scaling factor. Defaults to 1.0.

        Attributes:
            scaling (float): Scaling factor.
        """
        self.scaling = scaling

    def __call__(self, pf: Function) -> Function:
        """Evaluate the double well potential.

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Double well potential
        """
        return self.scaling * pf**2 * (1 - pf) ** 2

    def prime(self, pf:Function) -> Function:
        """Evaluate the derivative of the double well potential.

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Derivative of the double well potential
        """
        return self.scaling * 2 * pf * (1 - pf) * (1 - 2 * pf)

    def doubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the double well potential.

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Second derivative of the double well potential
        """
        return self.scaling * 2 * (1 - 6 * pf + 6 * pf**2)

    def c(self, pf: Function) -> Function:
        """Evaluate the convex part of the double well potential. Psi_c = (pf- 0.5)^4+ 0.0625.

        args:
            pf (df.Function): Phasefield

        returns:
            df.Function: Convex part of the double well potential
        """
        return self.scaling * ((pf - 0.5) ** 4 + 0.0625)

    def cprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the convex part of the double well potential. Psi_c' = 4(pf- 0.5)^3.

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Derivative of the convex part of the double well potential
        """
        return self.scaling * 4 * (pf - 0.5) ** 3

    def cdoubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the convex part of the double well potential. Psi_c'' = 12(pf- 0.5)^2.

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Second derivative of the convex part of the double well potential
        """
        return self.scaling * 12 * (pf - 0.5) ** 2

    def e(self, pf: Function) -> Function:
        """Evaluate the expansive part of the double well potential. Psi_e = 0.5(pf- 0.5)^2.

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Expansive part of the double well potential
        """
        return self.scaling * 0.5 * (pf - 0.5) ** 2

    def eprime(self, pf: Function) -> Function:
        """Evaluate the derivative of the expansive part of the double well potential. Psi_e' = (pf- 0.5).

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Derivative of the expansive part of the double well potential
        """
        return self.scaling * (pf - 0.5)

    def edoubleprime(self, pf: Function) -> Function:
        """Evaluate the second derivative of the expansive part of the double well potential. Psi_e'' = 1.

        Args:
            pf (df.Function): Phasefield

        Returns:
            df.Function: Second derivative of the expansive part of the double well potential
        """
        return self.scaling * 1.0




# Define material parameters

# CH
gamma = 1.0
ell = 2.0e-1
mobility = 1
doublewell = DoubleWellPotential()

# Time discretization
dt = 1.0e-3
num_time_steps = 100
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
pf_n, mu_n = split(xi_n)

xi_prev = Function(V)
pf_prev, mu_prev = split(xi_prev)

xi_old = Function(V)
pf_old, mu_old = split(xi_old)


# Initial condtions
rng = np.random.default_rng(42)
#qinitialconditions = lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1]))
def initialconditions(x):
    values = np.zeros(x.shape[1])
    # Set value 1 for the right half of the domain (x >= 0.5)
    values[x[0] >= 0.5] = 1.0
    return values
xi_n.sub(0).interpolate(initialconditions)


# Linear variational formsç
A_pf = (
    inner(pf, eta_pf) * dx
    + dt * mobility * inner(grad(mu), grad(eta_pf)) * dx
)
L_pf =  inner(pf_old,eta_pf)*dx

A_mu = (
    inner(mu, eta_mu) * dx
    - gamma * ell * inner(grad(pf), grad(eta_mu)) * dx
    - gamma
    / ell
    * doublewell.cdoubleprime(pf_prev) * inner(pf, eta_mu)
    * dx
)
L_mu = gamma/ell*inner(doublewell.cprime(pf_prev)-doublewell.cdoubleprime(pf_prev)*pf_prev+doublewell.eprime(pf_old),eta_mu)*dx

a = A_pf + A_mu
L = L_pf + L_mu



# Output file
output_file_pf = XDMFFile(MPI.COMM_WORLD, "../output/ch.xdmf", "w")
output_file_pf.write_mesh(msh)

def l2increment(u,v):
    diff = u-v
    diff_sq = inner(diff,diff)*dx
    int_diff = assemble_scalar(form(diff_sq))
    return np.sqrt(int_diff)



# Time stepping
t = 0.0

V0, dofs = V.sub(0).collapse()

# Prepare viewer for plotting the solution during the computation
if have_pyvista:
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

    # Update current time
    t += dt

    for j in range(max_iter):
        xi_prev.x.array[:] = xi_n.x.array

        # Define the problem
        problem = LinearProblem(a, L)
        xi_n = problem.solve()
        increment = l2increment(pf_n,pf_prev)
        print(
            f"Norm at time step {i} iteration {j}: {increment}"
        )

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
