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
    params['batchSize'] = 5
    params['simulation_time'] = 20
    params['simulation_dt'] = 1e-2
    params['isTensorboard'] = True
    params['epochs'] = 50

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


def generateTrainingData():
    observer = createObserver()

    net = np.arange(-1, 1, 0.2)
    mesh = np.array(np.meshgrid(net, net))
    combinations = mesh.T.reshape(-1, 2)
    points = torch.tensor(combinations)

    k = 10
    t_c = k/min(abs(observer.eigenD.real))
    nsims = points.shape[0]
    y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)
    y_1 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)

    # Simulate backward
    dt = -1e-2
    tsim = (0, -t_c)
    y_0[:observer.dim_x, :] = torch.transpose(points, 0, 1)
    tq, data_bw = observer.simulateLueneberger(y_0, tsim, dt)

    # Simulate forward
    dt = 1e-2
    tsim = (-t_c, 0)
    y_1[:observer.dim_x, :] = data_bw[-1, :observer.dim_x, :]
    tq, data_fw = observer.simulateLueneberger(y_1, tsim, dt)

    return torch.transpose(data_fw[-1, :, :], 0, 1).float(), observer


def train():
    params = getParams()
    data, observer = generateTrainingData()

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model params
    model = Autoencoder(2, 3, observer)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if params['isTensorboard']:
        writer = SummaryWriter()

    # Create trainloader
    trainloader = utils.data.DataLoader(data, batch_size=params['batchSize'],
                                        shuffle=True, num_workers=2)

    # Train Transformation
    # Loop over dataset
    for epoch in range(params['epochs']):

        # Track loss
        running_loss = 0.0

        # Train
        for i, batch in enumerate(trainloader, 0):
            # Set input and labels
            inputs = torch.tensor(batch[:, :observer.dim_x], requires_grad=True)
            labels = torch.tensor(batch[:, observer.dim_x:], requires_grad=False)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            z, x_hat = model(inputs, params)

            dTdx = torch.autograd.functional.jacobian(model.encoder, inputs, create_graph=False, strict=False, vectorize=False)
            dTdx = dTdx[dTdx != 0].reshape((params['batchSize'],observer.dim_z,observer.dim_x))

            # Forward + Backward + Optimize
            loss, loss1, loss2 = model.loss(inputs, x_hat, dTdx, z, observer, params)
            if params['isTensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                    'loss1': loss1,
                    'loss2': loss2,
                }, i + (epoch*params['epochs']))
                writer.flush()

            loss.backward()
            optimizer.step()

            print('{} loss: {}'.format(i, loss))

        # validate prediction

        if params['isTensorboard']:
            with torch.no_grad():
                z, x_hat = model(data[0,:observer.dim_x], params)
            
            tsim = (0,50)
            dt = 1e-2

            w_test = torch.cat((x_hat, z)).reshape(5,1)
            w_ground = data[0].reshape(5,1)

            tq, w = observer.simulateLueneberger(w_ground, tsim, dt)
            tq_, w_ = observer.simulateLueneberger(w_test, tsim, dt)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(tq, w[:,:2,0])
            ax.plot(tq_, w_[:,:2,0])
            writer.add_figure("recon", fig, global_step=epoch, close=True, walltime=None)
            writer.flush()

    print('Finished Training')

    if params['isTensorboard']:
        writer.close()
