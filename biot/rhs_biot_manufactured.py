import dolfin as df

class RHSManufacturedBiot:
    def __init__(self, alpha, M, kappa, lame_mu, lame_lambda, t0=0):
        self.alpha = alpha
        self.M = M
        self.kappa = kappa
        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda
        self.S_f = df.Expression("alpha*(x[0]*x[1]*(x[0] - 1) + x[0]*x[1]*(x[1] - 1) + x[0]*(x[0] - 1)*(x[1] - 1) + x[1]*(x[0] - 1)*(x[1] - 1)) - kappa*(2*t*x[0]*(x[0] - 1) + 2*t*x[1]*(x[1] - 1)) + x[0]*x[1]*(x[0] - 1)*(x[1] - 1)/M", alpha = self.alpha, M = self.M, t=t0, kappa = self.kappa, degree = 4)
        self.f = df.Expression(("alpha*(t*x[0]*x[1]*(x[1] - 1) + t*x[1]*(x[0] - 1)*(x[1] - 1)) - lame_lambda*(t*x[0]*x[1] + t*x[0]*(x[1] - 1) + t*x[1]*(x[0] - 1) + 2*t*x[1]*(x[1] - 1) + t*(x[0] - 1)*(x[1] - 1)) - 2*lame_mu*(0.5*t*x[0]*x[1] + 1.0*t*x[0]*(x[0] - 1) + 0.5*t*x[0]*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1) + 2*t*x[1]*(x[1] - 1) + 0.5*t*(x[0] - 1)*(x[1] - 1))","alpha*(t*x[0]*x[1]*(x[0] - 1) + t*x[0]*(x[0] - 1)*(x[1] - 1)) - lame_lambda*(t*x[0]*x[1] + 2*t*x[0]*(x[0] - 1) + t*x[0]*(x[1] - 1) + t*x[1]*(x[0] - 1) + t*(x[0] - 1)*(x[1] - 1)) - 2*lame_mu*(0.5*t*x[0]*x[1] + 2*t*x[0]*(x[0] - 1) + 0.5*t*x[0]*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1) + 1.0*t*x[1]*(x[1] - 1) + 0.5*t*(x[0] - 1)*(x[1] - 1))"), t=t0, alpha = self.alpha, lame_mu = self.lame_mu, lame_lambda = self.lame_lambda, degree = 4)
