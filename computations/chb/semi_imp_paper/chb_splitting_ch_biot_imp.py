import os
from time import time

# Fix MPI/OFI finalization errors on macOS
os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

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


def splitting_ch_biot_imp(parameters):
    nx = parameters.nx
    ny = parameters.ny
    msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

    # Define material parameters

    # CH
    ell = parameters.ell
    gamma = parameters.gamma
    mobility = parameters.mobility
    doublewell = chb.energies.SymmetricDoubleWellPotential_cutoff()

    # Elasticity
    # isotropic stiffness tensor
    # stiffness_tensor = chb.elasticity.IsotropicStiffnessTensor(
    #    lame_lambda_0=20, lame_mu_0=100, lame_lambda_1=0.1, lame_mu_1=1
    # )
    # heterogeneous and anisotropic stiffness tensor
    interpolator = chb.interpolate.SymmetricStandardInterpolator()
    stiffness_tensor = chb.elasticity.HeterogeneousStiffnessTensor(
        interpolator=interpolator
    )
    swelling = chb.elasticity.Swelling(swelling_parameter=parameters.swelling, pf_ref=0)

    # Biot
    alpha = chb.biot.NonlinearBiotCoupling(alpha0=parameters.alpha_0, alpha1=parameters.alpha_1, interpolator=interpolator)

    # Flow
    permeability = parameters.permeability
    compressibility = chb.flow.NonlinearCompressibility(
        M0=parameters.compressibility_0, M1=parameters.compressibility_1, interpolator=interpolator
    )

    # Time discretization
    dt = parameters.dt
    num_time_steps = parameters.num_time_steps
    T = dt * num_time_steps

    # Nonlinear iteration parameters
    max_iter_split = parameters.max_iter
    tol = parameters.tol



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
                        strain=sym(grad(u_prev)) - swelling(pf), pf=pf
                    ),
                    sym(grad(u_prev)) - swelling(pf),
                )
                - inner(
                    stiffness_tensor.stress(strain=sym(grad(u_prev)) - swelling(pf), pf=pf),
                    swelling.prime(),
                ),
                eta_mu,
            )
        )
        * dx
        - (
            inner(
                0.5
                * compressibility.prime(pf)
                * (theta_prev - alpha(pf) * div(u_prev)) ** 2
                - alpha.prime(pf)
                * compressibility(pf)
                * (theta_prev - alpha(pf) * div(u_prev))
                * div(u_prev),
                eta_mu,
            )
            * dx
        )
    )

    F_u = (
        inner(
            stiffness_tensor.stress(strain=sym(grad(u)) - swelling(pf), pf=pf)
            - alpha(pf) * compressibility(pf) * (theta - alpha(pf) * div(u)) * Identity(2),
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
        - inner(compressibility(pf) * (theta - alpha(pf) * div(u)), eta_p) * dx
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
    filenamepath = "../output/chb_splitting_ch_biot_imp_"


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

            if increment_split < tol:
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

        t_vec.append(tpost)
        energy_vec.append(energy)
        energy_int_vec.append(energy_int)
        energy_el_vec.append(energy_el)
        energy_fl_vec.append(energy_fl)
        iterations.append(iteration)
        times.append(tpost)



    # viz.final_plot(xiCH.sub(0))
    # vizP.final_plot(xiB_n.sub(2))

    # Create log DataFrame and save to Excel
    log_data = {
        "Time_Step": range(1, len(t_vec) + 1),  # Enumerate time steps starting from 1
        "Iterations": iterations,
        "Computational_Time": t_vec,
        "Total_Energy": energy_vec,
        "Interface_Energy": energy_int_vec,
        "Elastic_Energy": energy_el_vec,
        "Fluid_Energy": energy_fl_vec,
    }

    return log_data
