import numpy as np
from scipy import linalg, integrate, interpolate
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch


class LuenebergerObserver():

    def __init__(self, dim_x: int, dim_y: int, f: callable, g: callable, h: callable, u: callable):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = self.dim_y * (self.dim_x + 1)

        self.f = f
        self.g = g
        self.h = h
        self.u = u

        self.F = torch.zeros((self.dim_z, 1))
        self.eigenD = torch.zeros((self.dim_z, 1))
        self.D = torch.zeros((self.dim_z, self.dim_z))

        self.T = Model(self.dim_x, self.dim_z)
        self.T_star = Model(self.dim_z, self.dim_x)

    def tensorDFromEigen(self, eigen: torch.tensor) -> torch.tensor:
        """
        Generates the state matrix D to use in a Luenberger
        observer given the desired eigenvalues.
        """
        self.eigenD = eigen
        eig_complex, eig_real = [x for x in eigen if x.imag != 0], [
            x for x in eigen if x.imag == 0]

        if(any(~np.isnan(eig_complex))):
            eig_complex = sorted(eig_complex)
            eigenCell = self.eigenCellFromEigen(eig_complex, eig_real)
            D = linalg.block_diag(*eigenCell[:])

            return torch.tensor(D)

    @staticmethod
    def eigenCellFromEigen(eig_complex: torch.tensor, eig_real: torch.tensor) -> []:
        """
        Generates a cell array containing 2X2 real
        matrices for each pair of complex conjugate eigenvalues passed in the
        arguments, and real scalar for each real eigenvalue.
        """
        eigenCell = []

        for i in range(0, len(eig_complex), 2):
            array = np.zeros(shape=(2, 2))
            array[0, 0] = eig_complex[i].real
            array[0, 1] = eig_complex[i].imag
            array[1, 0] = eig_complex[i+1].imag
            array[1, 1] = eig_complex[i+1].real
            eigenCell.append(array)

        for i in eig_real:
            array = np.zeros(shape=(1, 1))
            array[0, 0] = i.real
            eigenCell.append(array)

        return eigenCell

    def simulateLueneberger(self, y_0: torch.tensor, tsim: tuple, dt) -> [torch.tensor, torch.tensor]:
        """
        Runs and outputs the results from 
        multiple simulations of an input-affine nonlinear system driving a 
        Luenberger observer target system.
        """
        def dydt(t, y):
            x = y[0:self.dim_x]
            z = y[self.dim_x:len(y)]
            x_dot = self.f(x) + self.g(x) * self.u(t)
            z_dot = torch.matmul(self.D, z)+self.F*self.h(x)
            return torch.cat((torch.tensor(x_dot), z_dot))

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Solve
        sol = odeint(dydt, y_0, tq)

        return tq, sol

    # Normalizing function
    def normalize(self, data):
        return (data-np.min(data))/(np.max(data) - np.min(data))

    # Training pipeline
    def computeNonlinearLuenbergerTransformation(
            self, tq: torch.Tensor, data: torch.Tensor, isForwardTrans: bool, epochs: int, batchSize: int):
        """
        Numerically estimate the
        nonlinear Luenberger transformation of a SISO input-affine nonlinear
        system with static transformation, and the corresponding left-inverse.
        """
        # Set size according to compute either T or T*
        if isForwardTrans:
            netSize =  (self.dim_x, self.dim_z)
            dataInput = (0, self.dim_x)
            dataOutput = (self.dim_x, self.dim_x+self.dim_z)
        else:
            netSize =  (self.dim_z, self.dim_x)
            dataInput = (self.dim_x, self.dim_x+self.dim_z)
            dataOutput = (0, self.dim_x)

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

            print('====> Epoch: {} done!'.format(epoch))
        print('Finished Training')

        if isForwardTrans:
            self.T = model.to("cpu")
        else:
            self.T_star = model.to("cpu")
        
        return model


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
