from lnos.net.autoencoder import Autoencoder
from lnos.net.helperfnc import splitDataShifts
from lnos.observer.lueneberger import LuenebergerObserver
from lnos.datasets.exampleSystems import getAutonomousSystem
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def getParams():
    params = {}

    # settings related to loss function
    params['num_shifts'] = 5
    params['batch_size'] = 5
    params['simulation_time'] = 20
    params['simulation_dt'] = 1e-2
    params['isTensorboard'] = True

    return params


def createObserver():
    f, h, g, u, dim_x, dim_y, eigen = getAutonomousSystem()

    # Initiate observer with system dimensions
    observer = LuenebergerObserver(dim_x, dim_y, f, g, h, u)

    # Set system dynamics
    observer.D = observer.tensorDFromEigen(eigen)
    observer.F = torch.Tensor([[1.0], [1.0], [1.0]])

    return observer


def createTrainingData(params):

    observer = createObserver()

    net = np.arange(-1, 1, 0.2)
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
    t_c = k/min(abs(observer.eigenD.real))
    idx = max(np.argwhere(tq < t_c))
    data = data[idx[-1]-1:, :, :]

    params['simulation_time_offset'] = int(params['simulation_time']/params['simulation_dt'] - (idx[-1]-1))

    return data, observer


def train():
    params = getParams()
    data, observer = createTrainingData(params)

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model params
    model = Autoencoder(2, 3, observer)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if params['isTensorboard']:
        writer = SummaryWriter()

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
            x, y, z_k, encodedList = model(batchDataTrain[:, :, :2], params, step, False)

            # Forward + Backward + Optimize
            loss, loss1, loss2 = model.loss(x[2:, :, :], y[2:, :, :], z_k[2:, :, :], encodedList[2:, :, :])
            if params['isTensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                    'loss1': loss1,
                    'loss2': loss2,
                }, step + (i*data.shape[-1]))
                writer.flush()

            loss.backward()
            optimizer.step()

            print('{} loss: {}'.format(step, loss))

        # validate prediction

        

        if params['isTensorboard']:
            with torch.no_grad():
                x, y, z_k, encodedList = model(torch.Tensor(dataTrain[:, :, :2]), params, 0, True)
            y = y.view(y.shape[0]*y.shape[1], 2)
            x = x.view(x.shape[0]*x.shape[1], 2)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(torch.arange(y.shape[0]), y[:,0])
            ax.plot(torch.arange(x.shape[0]), x[:,0])
            writer.add_figure("recon", fig, global_step=i, close=True, walltime=None)
            writer.flush()


    print('Finished Training')

    if params['isTensorboard']:
        writer.close()
