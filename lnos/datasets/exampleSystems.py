from scipy import signal
import numpy as np
import math
import torch

def getAutonomousSystem():
    # Define plant dynamics
    def f(x): return torch.cat((torch.reshape(torch.pow(x[1, :], 3), (1, -1)), torch.reshape(-x[0, :], (1, -1))), 0)
    def h(x): return torch.reshape(x[0, :], (1, -1))
    def g(x): return torch.zeros(x.shape[0], x.shape[1])
    def u(x): return 0

    # System dimension
    dim_x = 2
    dim_y = 1

    # Eigenvalues for D 
    b, a = signal.bessel(3, 2*math.pi, 'low', analog=True, norm='phase')
    eigen = np.roots(a)

    return f,h,g,u,dim_x,dim_y,eigen
