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
        self.D = torch.zeros((self.dim_z, self.dim_z))

    def tensorDFromEigen(self, eigen: torch.tensor) -> torch.tensor:
        """
        Generates the state matrix D to use in a Luenberger
        observer given the desired eigenvalues.
        D = tensorDFromEigen(eigen) returns a real m X m matrix D
        representing the state matrix of a Luenberger observer. 'eigenvals' is
        a complex m X 1 matrix containing the desired eigenvalues of the
        observer. It is assumed all complex eigenvalues are in conjugate pairs.
        """
        eig_complex, eig_real = [x for x in eigen if x.imag != 0], [
            x for x in eigen if x.imag == 0]

        if(any(~np.isnan(eig_complex))):
            eig_complex = sorted(eig_complex)
            eigenCell = self.matrixFromEigen(eig_complex, eig_real)
            D = linalg.block_diag(*eigenCell[:])

            return torch.tensor(D)

    @staticmethod
    def eigenCellFromEigen(eig_complex: torch.tensor, eig_real: torch.tensor) -> []:
        """
        Generates a cell array containing 2X2 real
        matrices for each pair of complex conjugate eigenvalues passed in the
        arguments, and real scalar for each real eigenvalue.
        eig_cell = eigenCellFromEigen(eig_complex, eig_real) returns an
        (m_c/2 + m_r)-dimensional cell array containing a real 2X2 matrix for
        each complex conjugate pair in the complex m_c X 1 matrix 'eig_complex'
        containing only complex conjugate pairs, and a real scalar for each
        eigenvalue in the real m_r X 1 matrix 'eig_real'.
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
        performMultipleLuenbergerSimulations    Runs and outputs the results from 
        multiple simulations of an input-affine nonlinear system driving a 
        Luenberger observer target system.

        [tq, sol] = performMultipleLuenbergerSimulations(f,g,h,dimx,u,D,F,w0_array,...
        nsims,tsim,dt) returns the output of 'nsims' simulations lasting for
        time 'tsim' of the n-dimensional state affine nonlinear system with 
        state function 'f', input function 'g', output function 'g' and input 
        'u', driving a Luenberger observer with m X m real state matrix D and 
        m X 1 real input matrix F driven by output from plant. 'w0_array' is an
        (n+m) X 'nsims' real matrix representing the initial conditions for the
        plant and observer target system for the simulations to be performed.
        'dt' is the time step the simulation results are resampled into before
        getting returned.
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
            self, tq: torch.Tensor, data: torch.Tensor, isForwardTrans: bool, epochs: int, batchSize: int) -> nn.Module:
        """
        performMultipleLuenbergerSimulations    Runs and outputs the results from 
        multiple simulations of an input-affine nonlinear system driving a 
        Luenberger observer target system.

          [tq, output_data] =
          performMultipleLuenbergerSimulations(f,g,h,dimx,u,D,F,w0_array,...
          nsims,tsim,dt) returns the output of 'nsims' simulations lasting for
          time 'tsim' of the n-dimensional state affine nonlinear system with 
          state function 'f', input function 'g', output function 'g' and input 
          'u', driving a Luenberger observer with m X m real state matrix D and 
          m X 1 real input matrix F driven by output from plant. 'w0_array' is an
          (n+m) X 'nsims' real matrix representing the initial conditions for the
          plant and observer target system for the simulations to be performed.
          'dt' is the time step the simulation results are resampled into before
          getting returned.
        """
        # Set size according to compute either T or T*
        if isForwardTrans:
            netSize = (self.dim_x, self.dim_z)
            dataInput = (0, self.dim_x)
            dataOutput = (self.dim_x, self.dim_x+self.dim_z)
        else:
            netSize = (self.dim_z, self.dim_x)
            dataInput = (self.dim_x, self.dim_x+self.dim_z)
            dataOutput = (0, self.dim_x)

        # Make torch use the GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # T_star params
        T = Model(netSize[0], netSize[1])
        T.to(device)
        optimizer = optim.Adam(T.parameters(), lr=0.001)

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
                outputs = T(inputs)
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

        return T

# Define NN model


class Model(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, outputSize)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
