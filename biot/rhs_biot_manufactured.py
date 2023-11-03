import dolfin as df

class RHSManufacturedBiot:
    def __init__(self, alpha, M, kappa, lame_mu, lame_lambda, p_ref, t0=0):
        self.alpha = alpha
        self.M = M
        self.kappa = kappa
        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda
        self.p_ref = p_ref
        self.S_f = df.Expression("alpha*(x[0]*x[1]*(x[0] - 1) + x[0]*x[1]*(x[1] - 1) + x[0]*(x[0] - 1)*(x[1] - 1) + x[1]*(x[0] - 1)*(x[1] - 1)) - kappa*(2*t*x[0]*(x[0] - 1)/p_ref + 2*t*x[1]*(x[1] - 1)/p_ref) + x[0]*x[1]*(x[0] - 1)*(x[1] - 1)/(M*p_ref)", alpha = self.alpha, M = self.M, t=t0, p_ref = self.p_ref, kappa = self.kappa, degree = 4)
        self.f = df.Expression(("alpha*(t*x[0]*x[1]*(x[1] - 1)/p_ref + t*x[1]*(x[0] - 1)*(x[1] - 1)/p_ref) - lame_lambda*(t*x[0]*x[1] + t*x[0]*(x[1] - 1) + t*x[1]*(x[0] - 1) + 2*t*x[1]*(x[1] - 1) + t*(x[0] - 1)*(x[1] - 1)) - 2*lame_mu*(0.5*t*x[0]*x[1] + 1.0*t*x[0]*(x[0] - 1) + 0.5*t*x[0]*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1) + 2*t*x[1]*(x[1] - 1) + 0.5*t*(x[0] - 1)*(x[1] - 1))","alpha*(t*x[0]*x[1]*(x[0] - 1)/p_ref + t*x[0]*(x[0] - 1)*(x[1] - 1)/p_ref) - lame_lambda*(t*x[0]*x[1] + 2*t*x[0]*(x[0] - 1) + t*x[0]*(x[1] - 1) + t*x[1]*(x[0] - 1) + t*(x[0] - 1)*(x[1] - 1)) - 2*lame_mu*(0.5*t*x[0]*x[1] + 2*t*x[0]*(x[0] - 1) + 0.5*t*x[0]*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1) + 1.0*t*x[1]*(x[1] - 1) + 0.5*t*(x[0] - 1)*(x[1] - 1))"), t=t0, alpha = self.alpha, lame_mu = self.lame_mu, lame_lambda = self.lame_lambda, p_ref = self.p_ref, degree = 4)
