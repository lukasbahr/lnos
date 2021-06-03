from lnos.observer.lueneberger import LuenebergerObserver
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, observer: LuenebergerObserver, options):
        super(Autoencoder, self).__init__()
        self.observer = observer
        self.options = options

        numHL = options['numHiddenLayers']
        sizeHL = options['sizeHiddenLayer']

        if options['activation'] == "relu":
            self.act = nn.ReLU()
        elif options['activation'] == "tanh":
            self.act = nn.Tanh()
        else:
            print(
                "Activation function {} not found. Available options: ['relu', 'tanh'].".format(
                    options['activation']))

        self.encoderLayers = nn.ModuleList()
        self.decoderLayers = nn.ModuleList()

        self.encoderLayers.append(nn.Linear(self.observer.dim_x, sizeHL))
        self.decoderLayers.append(nn.Linear(self.observer.dim_z, sizeHL))

        for i in range(numHL):
            self.encoderLayers.append(nn.Linear(sizeHL, sizeHL))
            self.decoderLayers.append(nn.Linear(sizeHL, sizeHL))

        self.encoderLayers.append(nn.Linear(sizeHL, self.observer.dim_z))
        self.decoderLayers.append(nn.Linear(sizeHL, self.observer.dim_x))

    def encoder(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        for layer in self.encoderLayers:
            x = self.act(layer(x))
        return x

    def decoder(self, x):
        for layer in self.decoderLayers:
            x = self.act(layer(x))
        return x

    def loss(self, x, x_hat, dTdx, z):

        z = z.to("cpu")
        x = z.to("cpu")
        x_hat = x_hat.to("cpu")
        dTdx = dTdx.to("cpu")

        mse = nn.MSELoss()

        loss1 = self.options['reconLambda'] * mse(x, x_hat)

        lhs = torch.zeros((self.observer.dim_z, self.options['batchSize']))
        for i in range(self.options['batchSize']):
            lhs[:, i] = torch.matmul(dTdx[i], self.observer.f(x.T).T[i]).T

        rhs = torch.matmul(self.observer.D, z.T)+torch.matmul(self.observer.F, self.observer.h(x.T))

        loss2 = mse(lhs, rhs)

        loss = loss1 + loss2

        return loss, loss1, loss2

    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""

        z = self.encoder(x)

        x_hat = self.decoder(z)

        return z, x_hat
