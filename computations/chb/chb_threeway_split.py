import dolfin as df

from dolfin import dx, grad, inner, sym, dot, div

import chb


interpolator = chb.StandardInterpolator()

# Define material parameters

# CH
gamma = 1.0
ell = 1.0e-2
mobility = 1
doublewell = chb.DoubleWellPotential()

# Elasticity
stiffness = chb.HeterogeneousStiffnessTensor()
swelling = 0.3

# Flow
compressibility0 = 1
compressibility1 = 0.1
M = chb.NonlinearCompressibility(compressibility0, compressibility1)

permeability0 = 1
permeability1 = 0.1
k = chb.NonlinearPermeability(permeability0, permeability1)

# Coupling
alpha0 = 1
alpha1 = 0.5
alpha = chb.NonlinearBiotCoupling(alpha0, alpha1)

# Energies
energy_h = chb.CHBHydraulicEnergy(M, alpha)
energy_e = chb.CHBElasticEnergy(stiffness, swelling)

# Time discretization
dt = 0.0001
num_time_steps = 10
T = dt * num_time_steps

# Spatial discretization
nx = ny = 64
mesh = df.UnitSquareMesh(nx, ny)

# Finite elements
P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
RT0 = df.VectorElement("RT", mesh.ufl_cell(), 1)
P0 = df.FiniteElement("DG", mesh.ufl_cell(), 0)
P2V = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)

# Function spaces
V_ch = df.FunctionSpace(mesh, P1 * P1)
V_f = df.FunctionSpace(mesh, RT0 * P0)
V_e = df.FunctionSpace(mesh, P2V)

# Test and trial functions
# CH
ch = df.TrialFunction(V_ch)
eta_pf, eta_mu = df.TestFunction(V_ch)
pf, mu = df.split(ch)

# Elasticity
u = df.TrialFunction(V_e)
eta_u = df.TestFunction(V_e)

# Flow
fl = df.TrialFunction(V_f)
eta_q, eta_p = df.TestFunction(V_f)
q, p = df.split(fl)


# Iteration functions
# CH
ch_n = df.Function(V_ch)
pf_n, _ = df.split(ch_n)
pf_prev, _ = df.Function(V_ch)
pf_old, _ = df.Function(V_ch)
pf_inner_prev, _ = df.Function(V_ch)

# Elasticity
u_n = df.Function(V_e)
u_prev = df.Function(V_e)
u_old = df.Function(V_e)

# Flow
q_n, p_n = df.Function(V_f)
q_old, p_old = df.Function(V_f)
q_prev, p_prev = df.Function(V_f)


# Boundary conditions
def boundary(x, on_boundary):
    return on_boundary


# Elasticity
zero_e = df.Constant((0.0, 0.0))
bc_e = df.DirichletBC(V_e, zero_e, boundary)

# Flow
zero_f = df.Constant(0.0)
bc_f = df.DirichletBC(V_f.sub(1), zero_f, boundary)

# Initial condtions
# CH
initialconditions = chb.RandomInitialConditions()
ch_n.interpolate(initialconditions)

# Elasticity
u_n.interpolate(zero_e)

# Flow
p_n.interpolate(zero_f)