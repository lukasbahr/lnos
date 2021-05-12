from lnos.observer.lueneberger import LuenebergerObserver
from lnos.datasets.plot import plotLogError2D
from scipy import signal
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Define plant
    def f(x): return torch.cat((torch.reshape(torch.pow(x[1, :], 3), (1, -1)), torch.reshape(-x[0, :], (1, -1))), 0)
    def h(x): return torch.reshape(x[0, :], (1, -1))
    def g(x): return torch.zeros(x.shape[0], x.shape[1])
    def u(x): return 0

    # State dim
    dim_x = 2
    dim_y = 1

    observer = LuenebergerObserver(dim_x, dim_y, f, g, h, u)

    # Lueneberger observer params
    b, a = signal.bessel(3, 2*math.pi, 'low', analog=True, norm='phase')
    eigen = np.roots(a)

    observer.D = observer.tensorDFromEigen(eigen)
    observer.F = torch.tensor([[1.0], [1.0], [1.0]])

    # 1. Sample from a uniform set $x_i^*$
    xv, yv = np.ogrid[-1:1:0.05, -1:1:0.05]

    # 2. From each $x_i^*$ integrate backward in time with intial condition $x(0)=x_i^*$

    # Simulation parameters for ODE
    tsim = (0, -10)
    dt = -1e-2
    nsims = xv.shape[0]*yv.shape[1]
    print(nsims)

    # Initial conditions backward simulation
    w0_array = torch.zeros((dim_x + observer.dim_z, nsims), dtype=torch.double)

    idx = 0
    for i in xv:
        for j in yv[0]:
            w0_array[0, idx] = i[0]
            w0_array[1, idx] = j
            idx += 1

    print(observer.D)
    # Simulate
    tq, data_fw = observer.simulateLueneberger(w0_array, tsim, dt)



    # Get initial condition for forward simulation
    w1_array = torch.zeros((dim_x + observer.dim_z, nsims), dtype=torch.double);

    for idx in range(nsims):
      w1_array[0,idx] = data_fw[-1,0,idx]
      w1_array[1,idx] = data_fw[-1,1,idx]

    # Simulation parameters for ODE
    tsim = (-10.0,0.0)
    dt = 1e-2

    # Simulate 
    tq, data_bw = observer.simulateLueneberger(w1_array, tsim, dt)


    # Create training data
    train_data = torch.zeros((nsims, data_bw.shape[1])) 

    for idx in range(nsims):
        train_data[idx,:] = data_bw[-1,:,idx]

    print("Start training on {} samples.".format(train_data.shape[0]))

    # Train model x(t)=T*(z(t))
    dataSize = ((2,5),(0,2))
    inputSize = (3,2)
    observer.computeNonlinearLuenbergerTransformation(tq,train_data,False,50,2)

    net = np.arange(-1,1,0.05)
    mesh = np.array(np.meshgrid(net,net))
    combinations = mesh.T.reshape(-1, 2)
    comb = torch.tensor(combinations)

    plotLogError2D(comb, observer)
