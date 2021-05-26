from lnos.net.autoencoder import Autoencoder
from lnos.observer.lueneberger import LuenebergerObserver
from lnos.datasets.exampleSystems import getAutonomousSystem
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import numpy as np


def getParams():
    params = {}

    # settings related to loss function
    params['num_shifts'] = 30
    params['batch_size'] = 5
    params['simulation_time'] = 60
    params['simulation_dt'] = 1e-2

    return params


def createTrainingData(params):

    f, h, g, u, dim_x, dim_y, eigen = getAutonomousSystem()

    # Initiate observer with system dimensions
    observer = LuenebergerObserver(dim_x, dim_y, f, g, h, u)

    # Set system dynamics
    observer.D = observer.tensorDFromEigen(eigen)
    observer.F = torch.Tensor([[1.0], [1.0], [1.0]])

    net = np.arange(0.1, 1, 0.5)
    mesh = np.array(np.meshgrid(net, net))
    combinations = mesh.T.reshape(-1, 2)
    points = torch.tensor(combinations)

    nsims = points.shape[0]
    y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)

    dt = params['simulation_dt']
    tsim = (0, params['simulation_time'])
    y_0[:observer.dim_x, :] = torch.transpose(points, 0, 1)
    tq, data = observer.simulateLueneberger(y_0, tsim, dt)

    # Bin initial data
    k = 5
    t_c = k/min(abs(eigen.real))
    idx = max(np.argwhere(tq < t_c))
    data = data[idx[-1]-1:, :, :]

    params['simulation_time_offset'] = int(params['simulation_time']/params['simulation_dt'] - (idx[-1]-1))

    return data, observer


def splitDataShifts(data, num_shifts):
    #TODO fix that

    n = data.shape[1]

    numTraj = int(data.shape[0] / num_shifts)

    dataTensor = np.zeros([num_shifts + 1, numTraj, n])

    for j in range(num_shifts):
        dataTensor[j, :, :] = data[j*numTraj:(j+1)*numTraj, :]

    return dataTensor


def train():
    params = getParams()
    data, observer = createTrainingData(params)

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model params
    model = Autoencoder(2, 3, observer)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for i in range(data.shape[-1]):

        dataTrain = splitDataShifts(data[:, :, i], params['num_shifts'])
        numSamples = dataTrain.shape[1]
        numBatches = int(np.floor(dataTrain.shape[1] / params['batch_size']))

        # ind = np.arange(numSamples)
        # np.random.shuffle(ind)
        # dataTrain = dataTrain[:, ind, :]

        running_loss = 0.0

        for step in range(numBatches):

            batchDataTrain = dataTrain[:, step*params['batch_size']:(step+1)*params['batch_size'], :]
            batchDataTrain = torch.Tensor(batchDataTrain)

             # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward 
            x, y, z_k, encodedList = model(batchDataTrain[:,:,:2],params, step)

            # Forward + Backward + Optimize
            loss = model.loss(x, y, z_k, encodedList)
            loss.backward()
            optimizer.step()

            print('{} loss: {}'.format(step,loss))

    print('Finished Training')



if __name__ == "__main__":
    train()
