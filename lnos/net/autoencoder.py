from lnos.observer.lueneberger import LuenebergerObserver
from torch import nn



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

    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        encodedList = self.encoderShiftForward(x)

        z_0 = encodedList[0,:,:]
        # y = []
        # y.append(self.decoder(z_0))

        tq, z_k = self.observer.simulateLueneberger(z_0, (0.0,float(x.shape[0])), 1)

        y = self.decoder(z_k)

        return x, y, z_k, encodedList 