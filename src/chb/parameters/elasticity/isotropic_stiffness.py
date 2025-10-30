import chb
import ufl


class IsotropicStiffnessTensor:


    def __init__(self, lame = True, dim = 2, interpolator = chb.interpolate.StandardInterpolator(), **kwargs):

        if lame:
            self.lame_lambda_0 = kwargs["lame_lambda_0"]
            self.lame_lambda_1 = kwargs["lame_lambda_1"]

            self.lame_mu_0 = kwargs["lame_mu_0"]
            self.lame_mu_1 = kwargs["lame_mu_1"]

        
        else:
            e_0 = kwargs["e_0"]
            e_1 = kwargs["e_1"]

            nu_0 = kwargs["nu_0"]
            nu_1 = kwargs["nu_1"]

            self.lame_lambda_0 = e_0 * nu_0 / ((1 + nu_0) * (1 - 2 * nu_0))
            self.lame_lambda_1 = e_1 * nu_1 / ((1 + nu_1) * (1 - 2 * nu_1))

            self.mu_0 = e_0 / (2 * (1 + nu_0))
            self.mu_1 = e_1 / (2 * (1 + nu_1))

        self.interpolator = interpolator

        self.dim = dim

    def __call__(self, u):
        pass

    def stress(self, strain, pf):
        return self.lame_lambda(pf)*ufl.tr(strain)*ufl.Identity(self.dim) + 2*self.lame_mu(pf)*strain
    
    def stress_prime(self, strain, pf):
        return self.lame_lambda_prime(pf)*ufl.tr(strain)*ufl.Identity(self.dim) + 2*self.lame_mu_prime(pf)*strain
    

    def lame_lambda(self, pf):
        return self.lame_lambda_0 + self.interpolator(pf)*(self.lame_lambda_1 - self.lame_lambda_0)
    
    def lame_lambda_prime(self, pf):
        return self.interpolator.prime(pf)*(self.lame_lambda_1 - self.lame_lambda_0)
    
    def lame_mu(self, pf):
        return self.lame_mu_0 + self.interpolator(pf)*(self.lame_mu_1 - self.lame_mu_0)
    
    def lame_mu_prime(self, pf):
        return self.interpolator.prime(pf)*(self.lame_mu_1 - self.lame_mu_0)
