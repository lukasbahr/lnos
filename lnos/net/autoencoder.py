from lnos.observer.lueneberger import LuenebergerObserver
import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, observer: LuenebergerObserver, options):
        super(Autoencoder, self).__init__()
        self.observer = observer
        self.options = options

        self.fc1 = nn.Linear(self.observer.dim_x, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, self.observer.dim_z)

        self.fc5 = nn.Linear(self.observer.dim_z, 25)
        self.fc6 = nn.Linear(25, 25)
        self.fc7 = nn.Linear(25, 25)
        self.fc8 = nn.Linear(25, self.observer.dim_x)
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

    def loss(self, x, x_hat, dTdx, z):

        mse = nn.MSELoss()
        loss1 = mse(x,x_hat)

        lhs = torch.zeros((self.observer.dim_z,self.options['batchSize']))
        for i in range(self.options['batchSize']):
            lhs[:,i] = torch.matmul(dTdx[i],self.observer.f(x.T).T[i]).T

        rhs = torch.matmul(self.observer.D,z.T)+torch.matmul(self.observer.F,self.observer.h(x.T))

        loss2 = mse(lhs, rhs)

        loss = loss1 + loss2

        return loss, loss1, loss2


    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""

        z = self.encoder(x)

        x_hat = self.decoder(z)

        return z, x_hat
