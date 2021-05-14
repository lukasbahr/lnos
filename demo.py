from lnos.observer.lueneberger import LuenebergerObserver
from lnos.datasets.plot import plotLogError2D
from scipy import signal
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Define plant dynamics
    def f(x): return torch.cat((torch.reshape(torch.pow(x[1, :], 3), (1, -1)), torch.reshape(-x[0, :], (1, -1))), 0)
    def h(x): return torch.reshape(x[0, :], (1, -1))
    def g(x): return torch.zeros(x.shape[0], x.shape[1])
    def u(x): return 0

    # System dimension
    dim_x = 2
    dim_y = 1

    # Initiate observer with system dimensions
    observer = LuenebergerObserver(dim_x, dim_y, f, g, h, u)

    # Eigenvalues for D 
    b, a = signal.bessel(3, 2*math.pi, 'low', analog=True, norm='phase')
    eigen = np.roots(a)

    # Set system dynamics
    observer.D = observer.tensorDFromEigen(eigen)
    observer.F = torch.tensor([[1.0], [1.0], [1.0]])

    # TODO find nicer solution
    net = np.arange(-1,1,0.05)
    # print(net)
    mesh = np.array(np.meshgrid(net,net))
    combinations = mesh.T.reshape(-1, 2)
    comb = torch.tensor(combinations)

    # Generate training data
    train_data = observer.generateTrainingData(comb)
    print(train_data.shape)

    # Compute transformation T_start
    observer.computeNonlinearLuenbergerTransformation(train_data,False,5,2)

    # Plot 2D log error 
    plotLogError2D(comb, observer)
