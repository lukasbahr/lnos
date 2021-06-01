from lnos.observer.lueneberger import LuenebergerObserver
from lnos.net.helperfnc import splitDataShifts
import torch
from torch import nn
import numpy as np

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

    def encoder(self, x):
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

    def loss(self, x, x_hat, dTdx, z, observer, params):

        mse = nn.MSELoss()
        loss1 = mse(x,x_hat)

        lhs = torch.zeros((observer.dim_z,params['batchSize']))
        for i in range(params['batchSize']):
            lhs[:,i] = torch.matmul(dTdx[i],observer.f(x.T).T[i]).T

        rhs = torch.matmul(observer.D,z.T)+torch.matmul(observer.F,observer.h(x.T))

        loss2 = mse(lhs, rhs)

        loss = loss1 + loss2

        return loss, loss1, loss2


    def forward(self, x, params):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""

        z = self.encoder(x)

        x_hat = self.decoder(z)

        return z, x_hat
