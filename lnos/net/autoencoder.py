from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(self.dim_x, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, self.dim_z)


    def encoder(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        x = x.float()
        x = self.fc1(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def encoderShiftForward(self, x, numShifts):
        """
        Shifts x_k numShifts foward in time
        """

        y = []
        for i in range(len(numShifts+1)):

            # x_k, x_k+1, x_k+2, ...
            y.append(self.encoder(x[i]))

        return y
        



    def decode(self, z):
        """Decode a batch of latent variables"""
        x = x.float()
        x = self.fc5(x)
        x = self.tanh(self.fc6(x))
        x = self.tanh(self.fc7(x))
        x = self.fc8(x)
        return x

    def reparam(self, mu, logvar):
        """Reparameterisation trick to sample z values. 
        This is stochastic during training,  and returns the mode during evaluation."""

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        mu, logvar = self.encode(x.view(-1, input_size))
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, reconstruction, x, mu, logvar):
        """ELBO assuming entries of x are binary variables, with closed form KLD."""

        bce = torch.nn.functional.binary_cross_entropy(reconstruction, x.view(-1, input_size))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.view(-1, input_size).data.shape[0] * input_size

        return bce + KLD

    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""

        mu, logvar = self.encode(x.view(-1, input_size))
        return self.reparam(mu, logvar)
