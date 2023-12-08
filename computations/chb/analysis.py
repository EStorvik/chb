import chb
import dolfin as df

from chb_monolithic import chb_monolithic
from chb_threeway_split import chb_threeway_split


# Run monolithic solver

df.parameters["form_compiler"]["quadrature_degree"] = 5

interpolator = chb.StandardInterpolator()

# Define material parameters

# CH
gamma = [5, 10]
ell = 2.0e-2
mobility = 1
doublewell = chb.DoubleWellPotential()

# Elasticity
swelling_parameter = 0.25
swelling = chb.Swelling(swelling_parameter)
stiffness = chb.HeterogeneousStiffnessTensor(interpolator=interpolator)


# Flow
compressibility0 = 1
compressibility1 = 0.1
M = chb.NonlinearCompressibility(
    compressibility0, compressibility1, interpolator=interpolator
)

permeability0 = 1
permeability1 = 0.1
k = chb.NonlinearPermeability(permeability0, permeability1, interpolator=interpolator)

# Coupling
alpha0 = 1
alpha1 = 0.5
alpha = chb.NonlinearBiotCoupling(alpha0, alpha1, interpolator=interpolator)

# Energies
energy_h = chb.CHBHydraulicEnergy(M, alpha)
energy_e = chb.CHBElasticEnergy(stiffness, swelling=swelling)

# Time discretization
dt = 1.0e-5
num_time_steps = 200
T = dt * num_time_steps

# Nonlinear iteration parameters
max_iter = 50
max_iter_inner_newton = 50
tol = 1e-6

# Spatial discretization
nx = ny = 64

initialConditions = chb.HalfnhalfInitialConditions(variables=2)

output_path_monolithic = "/home/erlend/src/fenics/output/chb/halfnhalf/monolithic/"
output_path_threeway = "/home/erlend/src/fenics/output/chb/halfnhalf/threewaysplit/"
output_interval = 5
log = ["log/gamma5", "log/gamma10"]


for i in range(len(gamma)):
    chb_monolithic(
        gamma = gamma[i],
        ell=ell,
        mobility=mobility,
        doublewell=doublewell,
        M=M,
        k=k,
        alpha=alpha,
        energy_h=energy_h,
        energy_e=energy_e,
        initialconditions=chb.HalfnhalfInitialConditions(variables=7),
        dt=dt,
        num_time_steps=num_time_steps,
        nx=nx,
        ny=ny,
        max_iter=max_iter,
        tol = tol,
        output_path=output_path_monolithic,
        output_interval=output_interval,
        log = log[i],
        verbose = True,
    )

    chb_threeway_split(
        gamma = gamma[i],
        ell=ell,
        mobility=mobility,
        doublewell=doublewell,
        M=M,
        k=k,
        alpha=alpha,
        energy_h=energy_h,
        energy_e=energy_e,
        initialconditions=chb.HalfnhalfInitialConditions(variables=2),
        dt=dt,
        num_time_steps=num_time_steps,
        nx=nx,
        ny=ny,
        max_iter_inner_newton=max_iter_inner_newton,
        max_iter_split=max_iter,
        tol = tol,
        output_path=output_path_threeway,
        output_interval=output_interval,
        log = log[i],
        verbose = True,
    )
