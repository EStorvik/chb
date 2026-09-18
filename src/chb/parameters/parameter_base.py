class Parameters:

    def __init__(
        self,
        nx=64,
        ny=64,
        dt=1e-3,
        num_time_steps=100,
        gamma=1,
        ell=0.025,
        mobility=1,
        swelling=0.5,
        permeability=1,
        compressibility_0=1,
        compressibility_1=0.1,
        alpha_0=1,
        alpha_1=0.1,
        max_iter=100,
        tol=1e-6,
        L=1,
    ):

        # Spatial
        self.nx = nx
        self.ny = ny

        # Time stepping
        self.dt = dt
        self.num_time_steps = num_time_steps

        # Cahn-Hilliard
        self.gamma = gamma
        self.ell = ell
        self.mobility = mobility

        # Elasticity
        self.swelling = swelling

        # Flow
        self.permeability = permeability
        self.compressibility_0 = compressibility_0
        self.compressibility_1 = compressibility_1

        # Biot
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1

        # Nonlinear
        self.tol = tol
        self.max_iter = max_iter

        # FS
        self.L = L
