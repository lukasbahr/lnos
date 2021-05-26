import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils


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


def normalize(self, data):
    """
    Simple normalization function between [0,1].
    """
    return (data-np.min(data))/(np.max(data) - np.min(data))


def generateTrainingData(points, observer) -> torch.tensor:

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

    return torch.transpose(data_fw[-1, :, :], 0, 1).float()


class Model(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, outputSize)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
