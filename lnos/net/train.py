
from lnos.net.linear import Model
from lnos.net.autoencoder import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def trainNonlinearLuenbergerTransformation(
        data: torch.Tensor, observer, isForwardTrans: bool, epochs: int, batchSize: int):
    """
    Numerically estimate the
    nonlinear Luenberger transformation of a SISO input-affine nonlinear
    system with static transformation, and the corresponding left-inverse.
    """
    dim_x = observer.dim_x
    dim_z = observer.dim_z

    # Set size according to compute either T or T*
    if isForwardTrans:
        netSize = (dim_x, dim_z)
        dataInput = (0, dim_x)
        dataOutput = (dim_x, dim_x+dim_z)
    else:
        netSize = (dim_z, dim_x)
        dataInput = (dim_x, dim_x+dim_z)
        dataOutput = (0, dim_x)

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model params
    model = Model(netSize[0], netSize[1])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Network params
    criterion = nn.MSELoss()
    epochs = epochs
    batchSize = batchSize

    # Create trainloader
    trainloader = utils.data.DataLoader(data, batch_size=batchSize,
                                        shuffle=True, num_workers=2)

    # Train Transformation
    # Loop over dataset
    for epoch in range(epochs):

        # Track loss
        running_loss = 0.0

        # Train
        for i, data in enumerate(trainloader, 0):
            # Set input and labels
            inputs = data[:, dataInput[0]:dataInput[1]].to(device)
            labels = data[:, dataOutput[0]:dataOutput[1]].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.00

        print('====> Epoch: {} done!'.format(epoch + 1))
    print('Finished Training')

    return model


def trainAutoencoder(data, observer, options):
    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model params
    model = Autoencoder(observer, options)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if options['isTensorboard']:
        writer = SummaryWriter()

    # Create trainloader
    trainloader = utils.data.DataLoader(data, batch_size=options['batchSize'],
                                        shuffle=True, num_workers=2)

    # Train Transformation
    # Loop over dataset
    for epoch in range(options['epochs']):

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
            z, x_hat = model(inputs)

            dTdx = torch.autograd.functional.jacobian(
                model.encoder, inputs, create_graph=False, strict=False, vectorize=False)
            dTdx = dTdx[dTdx != 0].reshape((options['batchSize'], observer.dim_z, observer.dim_x))

            # Forward + Backward + Optimize
            loss, loss1, loss2 = model.loss(inputs, x_hat, dTdx, z)
            if options['isTensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                    'loss1': loss1,
                    'loss2': loss2,
                }, i + (epoch*options['epochs']))

            loss.backward()
            optimizer.step()

            print('{} loss: {}'.format(i, loss))

        # validate prediction

        if options['isTensorboard']:
            with torch.no_grad():
                z, x_hat = model(data[0, :observer.dim_x])

            tsim = (0, 50)
            dt = 1e-2

            w_test = torch.cat((x_hat, z)).reshape(5, 1)
            w_ground = data[0].reshape(5, 1)

            tq, w = observer.simulateLueneberger(w_ground, tsim, dt)
            tq_, w_ = observer.simulateLueneberger(w_test, tsim, dt)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(tq, w[:, :2, 0])
            ax.plot(tq_, w_[:, :2, 0])
            writer.add_figure("recon", fig, global_step=epoch, close=True, walltime=None)

    print('Finished Training')

    if options['isTensorboard']:
        writer.close()

    return model