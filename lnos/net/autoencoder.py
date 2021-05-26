from lnos.observer.lueneberger import LuenebergerObserver
import torch
from torch import nn
import numpy as np

def splitDataShifts(data, num_shifts):

    n = data.shape[1]

    numTraj = int(data.shape[0] / num_shifts)

    dataTensor = np.zeros([num_shifts + 1, numTraj, n])

    for j in range(num_shifts):
        dataTensor[j, :, :] = data[j*numTraj:(j+1)*numTraj, :]

    return dataTensor



class Autoencoder(nn.Module):
    def __init__(self, inputSize, outputSize, observer: LuenebergerObserver):
        super(Autoencoder, self).__init__()
        self.observer = observer
        self.fc1 = nn.Linear(inputSize, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, outputSize)

        self.fc5 = nn.Linear(outputSize, 25)
        self.fc6 = nn.Linear(25, 25)
        self.fc7 = nn.Linear(25, 25)
        self.fc8 = nn.Linear(25, inputSize)
        self.tanh = nn.Tanh()

    def encoderShiftForward(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        x = self.fc1(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

    def decoder(self, x):
        x = self.fc5(x)
        x = self.tanh(self.fc6(x))
        x = self.tanh(self.fc7(x))
        x = self.fc8(x)
        return x

    def loss(self, x, y, z_k, encodedList):

        mse = nn.MSELoss()
        loss1 = mse(x,y)

        loss2 = mse(encodedList, z_k)

        loss = loss1 + loss2

        return loss


    def forward(self, x, params, step):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        encodedList = self.encoderShiftForward(x)

        w_0 = torch.reshape(torch.cat((x[0, 0, :], encodedList[0, 0, :])), (-1, 1))

        tq, z_k = self.observer.simulateLueneberger(w_0, (0.0, params['simulation_time']), params['simulation_dt'])
        z_k = z_k[:params['simulation_time_offset'], :]

        z_k = splitDataShifts(z_k[:,:,0].detach().numpy(), params['num_shifts'])
        z_k = z_k[:, step*params['batch_size']:(step+1)*params['batch_size'], :]

        z_k = torch.Tensor(z_k[:,:,2:])

        y = self.decoder(z_k)

        return x, y, z_k, encodedList
