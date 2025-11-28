class CahnHilliardParameters:

    def __init__(self):

        # Model
        self.gamma = 1
        self.ell = 0.025
        self.mobility = 1.0

        # Time Discretization
        self.dt = 1.0e-5
        self.num_time_steps = 5
        self.T = self.dt * self.num_time_steps

        # Spatial Discretization
        self.nx = self.ny = 64

        # Nonlinear iteration parameters
        self.tol = 1.0e-6
        self.max_iter = 200
