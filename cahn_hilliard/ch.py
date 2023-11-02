import chb
import dolfin as df


# Define functions
def main():
    # Define problem properties
    m = 1.0
    gamma = 1.0
    ell = 1.0e-2

    # Define solver properties
    # Time discretization (Implicit Euler)
    dt = 1.0e-5
    num_steps = 10
    T = dt*num_steps

    # Newton solver parameters
    max_iter = 20
    tol = 1.0e-8

    # Define mesh
    nx = ny = 32
    mesh = df.UnitSquareMesh(nx, ny)

    # Define function space
    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = df.FunctionSpace(mesh, P1*P1)

    # Define initial condition
    
     

# Main execution
if __name__ == "__main__":
    main()