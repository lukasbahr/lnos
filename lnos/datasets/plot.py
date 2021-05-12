from lnos.observer.lueneberger import LuenebergerObserver
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def rms_error(x, x_hat):
    return np.log(
        np.sqrt(
            (np.power((x[0]-x_hat[0]), 2) + np.power((x[1]-x_hat[1]), 2)) /
            x[0].shape[0]
        )
    )

def plotLogError2D(points, observer: LuenebergerObserver):

    # Simulation parameters for ODE
    gamma = abs(min(observer.eigenD))
    nsims = points.shape[0]
    y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)

    # Simulate backward
    dt = -1e-2
    tsim = (0, -gamma.real/10)
    y_0[:observer.dim_x,:] = torch.transpose(points,0,1)
    tq, data_bw = observer.simulateLueneberger(y_0, tsim, dt)

    # Simulate forward
    dt = 1e-2
    tsim = (-gamma.real/10,0)
    y_0[:observer.dim_x,:] = data_bw[-1,:observer.dim_x,:]
    tq, data_fw = observer.simulateLueneberger(y_0, tsim, dt)


    inputs = torch.tensor(data_fw[-1,observer.dim_x:observer.dim_x+observer.dim_z,:],dtype=torch.float64)
    inputs = Variable(torch.transpose(inputs,0,1), requires_grad=False)

    # Sample data from T*
    x_hat = observer.T_star(inputs)
    x_hat = x_hat.detach().numpy().T
    x = data_fw[-1,:observer.dim_x,:].numpy()

    plt.scatter(points[:,0], points[:,1], cmap='jet',
                c=rms_error(x,x_hat))
    cbar = plt.colorbar()
    cbar.set_label('Log relative error')

    plt.show()
