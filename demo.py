from lnos.observer.lueneberger import LuenebergerObserver
from lnos.datasets.plot import plotLogError2D, plotTrajectory2D
from scipy import signal
import math
import numpy as np
import torch
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

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

    T_star = observer.computeNonlinearLuenbergerTransformation(train_data, False, 20, 2)
    

    # Simulation parameters for ODE
    tsim = (0,50)
    dt = 1e-2

    # Compute test data
    w0_test = torch.tensor([[np.random.uniform(-1,1)], [np.random.uniform(-1,1)],[0],[0],[0]], dtype=torch.double)
    tq_test, w_test = observer.simulateLueneberger(w0_test, tsim, dt)

    # Data pipeline x_hat
    input = w_test.reshape(w_test.shape[0], w_test.shape[1]).float()
    input = Variable(input[:,2:])

    # Sample data from T*
    x_hat = observer.T_star(input)
    x_hat = x_hat.detach().numpy()

    # Plot x_1
    plt.plot(tq_test, x_hat[:,0])
    plt.plot(tq_test, w_test[:,0])

    # Plot x_2
    plt.plot(tq_test, x_hat[:,1])
    plt.plot(tq_test, w_test[:,1])
    plt.show() 

    net = np.arange(-1,1,0.01)
    # print(net)
    mesh = np.array(np.meshgrid(net,net))
    combinations = mesh.T.reshape(-1, 2)
    comb = torch.tensor(combinations)

    # Plot 2D log error 
    plotLogError2D(comb, observer)
