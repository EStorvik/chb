import chb
from typing import Optional
import numpy as np
from dolfin import sym, grad, dx
from ufl_legacy import indices


class HeterogeneousStiffnessTensor():
    def __init__(self, stiffness0: Optional[np.ndarray] = None, stiffness1: Optional[np.ndarray] = None, dim: int = 2, interpolator: chb.StandardInterpolator = chb.StandardInterpolator()) -> None:
        self.interpolator = interpolator
        
        if stiffness0 is None:
                self.stiffness0 = np.zeros((dim, dim, dim, dim))

                self.stiffness0[0,0,0,0] = 4
                self.stiffness0[0,0,1,1] = 2
                self.stiffness0[0,0,1,0] = 0
                self.stiffness0[0,0,0,1] = 0

                self.stiffness0[1,1,0,0] = 2
                self.stiffness0[1,1,1,1] = 4
                self.stiffness0[1,1,1,0] = 0
                self.stiffness0[1,1,0,1] = 0
                
                self.stiffness0[0,1,0,0] = 0
                self.stiffness0[0,1,1,1] = 0
                self.stiffness0[0,1,1,0] = 4
                self.stiffness0[0,1,0,1] = 4
                
                self.stiffness0[1,0,0,0] = 0
                self.stiffness0[1,0,1,1] = 0
                self.stiffness0[1,0,1,0] = 4
                self.stiffness0[1,0,0,1] = 4
                
        else: 
            self.stiffness0 = stiffness0

        if stiffness1 is None:
                self.stiffness1 = np.zeros((dim, dim, dim, dim))

                self.stiffness1[0,0,0,0] = 1
                self.stiffness1[0,0,1,1] = 0.5
                self.stiffness1[0,0,1,0] = 0
                self.stiffness1[0,0,0,1] = 0

                self.stiffness1[1,1,0,0] = 0.5
                self.stiffness1[1,1,1,1] = 1
                self.stiffness1[1,1,1,0] = 0
                self.stiffness1[1,1,0,1] = 0
                
                self.stiffness1[0,1,0,0] = 0
                self.stiffness1[0,1,1,1] = 0
                self.stiffness1[0,1,1,0] = 1
                self.stiffness1[0,1,0,1] = 1
                
                self.stiffness1[1,0,0,0] = 0
                self.stiffness1[1,0,1,1] = 0
                self.stiffness1[1,0,1,0] = 1
                self.stiffness1[1,0,0,1] = 1
            
        else:
            self.stiffness1 = stiffness1


    def __call__(self, u, v, phasefield):
        i, j, k, l = indices(4)
        return ((self.stiffness0[i, j, k, l]+self.interpolator(phasefield)(self.stiffness1[i,j,k,l]-self.stiffness0[i,j,k,l])) * sym(grad(u))[k, l] * sym(grad(v))[i, j])
    
    def prime(self, u, v, phasefield):
        i, j, k, l = indices(4)
        return ((self.interpolator.prime(phasefield)(self.stiffness1[i,j,k,l]-self.stiffness0[i,j,k,l])) * sym(grad(u))[k, l] * sym(grad(v))[i, j])