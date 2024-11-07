import dolfinx
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem

# Print versions to verify the installation
print("Dolfinx version: ", dolfinx.__version__)
print("UFL version: ", ufl.__version__)

# Test basic setup
domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
P1U = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V = fem.FunctionSpace(domain, P1U)
u = fem.Function(V)
print("Setup successful")
