import numpy as np
import torch


def normalize(self, data):
    """
    Normalize data between [0,1]
    """
    return (data-np.min(data))/(np.max(data) - np.min(data))


def generateTrainingData(observer, options):
    """
    Generate training samples (x,z) by simulating backward in time
    and after forward in time.
    """
    mesh = np.array(np.meshgrid(options['gridSize'], options['gridSize'])).T.reshape(-1, 2)
    mesh = torch.tensor(mesh)

    k = 10
    t_c = k/min(abs(observer.eigenD.real))
    nsims = mesh.shape[0]
    y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)
    y_1 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)

    # Simulate backward
    dt = -1e-2
    tsim = (0, -t_c)
    y_0[:observer.dim_x, :] = torch.transpose(mesh, 0, 1)
    tq, data_bw = observer.simulateLueneberger(y_0, tsim, dt)

    # Simulate forward
    tsim = (-t_c, 0)
    y_1[:observer.dim_x, :] = data_bw[-1, :observer.dim_x, :]
    tq, data_fw = observer.simulateLueneberger(y_1, tsim, -dt)

    return torch.transpose(data_fw[-1, :, :], 0, 1).float()
