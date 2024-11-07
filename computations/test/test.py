import numpy as np
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx import mesh, fem
import ufl

# Create mesh
msh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

# Define the function space
element = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
V = fem.functionspace(msh, element)

# Define anisotropic stiffness tensor components (Voigt notation)
C = np.array([[150.0, 50.0, 0.0], [50.0, 150.0, 0.0], [0.0, 0.0, 100.0]])


# Strain tensor
def epsilon(u):
    return ufl.sym(ufl.grad(u))


# Strain tensor in Voigt notation
def epsilon_voigt(u):
    eps = epsilon(u)
    return ufl.as_vector([eps[0, 0], eps[1, 1], eps[0, 1]])


# Stress tensor for anisotropic material
def sigma_anisotropic(u):
    eps_voigt = epsilon_voigt(u)
    sigma_voigt = ufl.as_vector(
        [
            C[0, 0] * eps_voigt[0] + C[0, 1] * eps_voigt[1] + C[0, 2] * eps_voigt[2],
            C[1, 0] * eps_voigt[0] + C[1, 1] * eps_voigt[1] + C[1, 2] * eps_voigt[2],
            C[2, 0] * eps_voigt[0] + C[2, 1] * eps_voigt[1] + C[2, 2] * eps_voigt[2],
        ]
    )
    return sigma_voigt


# Boundary conditions
u_bc = fem.Function(V)
u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))

# Locate the dofs where the boundary condition is to be applied using a geometric condition
facets = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], 0.0))
dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
bc = fem.dirichletbc(u_bc, dofs)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(sigma_anisotropic(u), epsilon_voigt(v)) * ufl.dx
l = ufl.dot(ufl.Constant(msh, PETSc.ScalarType((0, 0))), v) * ufl.dx

# Solve the problem
problem = fem.petsc.LinearProblem(a, l, bcs=[bc])
uh = problem.solve()

# Save the solution to XDMF file for visualization
with dolfinx.io.XDMFFile(msh.comm, "../output/output.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh, "displacement")
