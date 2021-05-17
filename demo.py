from lnos.observer.lueneberger import LuenebergerObserver
from lnos.datasets.plot import plotLogError2D, plotTrajectory2D
from scipy import signal
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchdiffeq import odeint

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

# Normalizing function
def normalize(data):
  return (data-np.min(data))/(np.max(data) - np.min(data))

# Pipeline for observer simulation
def simulateLueneberger(f,g,h,dimx,u,D,F,w0_array,nsims,tsim,dt):
    # PDE
    def dwdt(t,w):
        x = w[0:dimx]
        z = w[dimx:len(w)]
        x_dot = f(x) + g(x) * u(t)
        z_dot = torch.matmul(D,z)+F*h(x)
        return torch.cat((torch.tensor(x_dot),z_dot))

    # Output timestemps of solver
    tq = torch.arange(tsim[0],tsim[1],dt)

    # Initial value
    w0 = w0_array[:,:]

    # Solve
    w = odeint(dwdt,w0,tq)

    return tq, w

# Training pipeline
def computeNonlinearLuenbergerTransformation(tq,data,dataSize,netSize):
    # Make torch use of the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # T_star params
    T = Model(netSize[0], netSize[1])
    T.to(device)
    optimizer = optim.Adam(T.parameters(), lr=0.001) 

    # Network params
    criterion = nn.MSELoss()
    epochs = 20
    batchSize = 1

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
            inputs = data[:,dataSize[0][0]:dataSize[0][1]].to(device)
            labels = data[:,dataSize[1][0]:dataSize[1][1]].to(device)

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


if __name__ == "__main__":
    # Define plant dynamics
    def f(x): return torch.cat((torch.reshape(torch.pow(x[1, :], 3), (1, -1)), torch.reshape(-x[0, :], (1, -1))), 0)
    def h(x): return torch.reshape(x[0, :], (1, -1))
    def g(x): return torch.zeros(x.shape[0], x.shape[1])
    def u(x): return 0

    # System dimension
    dim_x = 2
    dim_y = 1

    # Initiate observer with system dimensions
    observer = LuenebergerObserver(dim_x, dim_y, f, g, h, u)

    # Eigenvalues for D 
    b, a = signal.bessel(3, 2*math.pi, 'low', analog=True, norm='phase')
    eigen = np.roots(a)

    # Set system dynamics
    observer.D = observer.tensorDFromEigen(eigen)
    observer.F = torch.tensor([[1.0], [1.0], [1.0]])
    print(observer.D)

    # TODO find nicer solution
    # net = np.arange(-1,1,0.05)
    # print(net)
    # mesh = np.array(np.meshgrid(net,net))
    # combinations = mesh.T.reshape(-1, 2)
    # comb = torch.tensor(combinations)

    # Generate training data
    # train_data = observer.generateTrainingData(comb)
    # print(train_data.shape)

    xv, yv = np.ogrid[-1:1:0.05, -1:1:0.05]
    # Simulation parameters for ODE
    tsim = (0,-10)
    dt = -1e-2
    nsims = xv.shape[0]*yv.shape[1]
    print(nsims)

    # Initial conditions backward simulation
    w0_array = torch.zeros((observer.dim_x + observer.D.shape[1], nsims), dtype=torch.double);

    idx = 0
    for i in xv:
        for j in yv[0]:
            w0_array[0,idx] = i[0]
            w0_array[1,idx] = j
            idx += 1

    # Simulate 
    tq, data_fw = simulateLueneberger(f,g,h,observer.dim_x,u,observer.D,observer.F, w0_array,nsims,tsim,dt)

    plt.plot(tq, data_fw[:,0,0:10])
    plt.show()


    # Get initial condition for forward simulation
    w1_array = torch.zeros((observer.dim_x + observer.D.shape[1], nsims), dtype=torch.double);

    for idx in range(nsims):
        w1_array[0,idx] = data_fw[-1,0,idx]
        w1_array[1,idx] = data_fw[-1,1,idx]

    # Simulation parameters for ODE
    tsim = (-10.0,0.0)
    dt = 1e-2

    # Simulate 
    tq, data_bw = simulateLueneberger(f,g,h,observer.dim_x,u,observer.D,observer.F, w1_array,nsims,tsim,dt)

    plt.scatter([tq[0] for i in range(10)], data_fw[-1,0,0:10])
    plt.plot(tq, data_bw[:,0,0:10])
    plt.show()

    # Create training data
    train_data = torch.zeros((nsims, data_bw.shape[1])) 

    for idx in range(nsims):
        train_data[idx,:] = data_bw[-1,:,idx]

    print("Start training on {} samples.".format(train_data.shape[0]))

    # Train model x(t)=T*(z(t))
    dataSize = ((2,5),(0,2))
    inputSize = (3,2)
    T_star = computeNonlinearLuenbergerTransformation(tq,train_data,dataSize,inputSize)
    

    # Simulation parameters for ODE
    tsim = (0,50)
    dt = 1e-2

    # Compute test data
    w0_test = torch.tensor([[np.random.uniform(0,1)], [np.random.uniform(0,1)],[0],[0],[0]], dtype=torch.double)
    tq_test,w_test = simulateLueneberger(f,g,h,observer.dim_x,u,observer.D,observer.F,w0_test,1,tsim,dt)

    # Data pipeline x_hat
    input = w_test.reshape(w_test.shape[0], w_test.shape[1])
    input = Variable(input[:,2:]).float()

    # Sample data from T*
    T_star = T_star.to("cpu")
    x_hat = T_star(input)
    x_hat = x_hat.detach().numpy()

    # Plot x_1
    plt.plot(tq_test, x_hat[:,0])
    plt.plot(tq_test, w_test[:,0])

    # Plot x_2
    plt.plot(tq_test, x_hat[:,1])
    plt.plot(tq_test, w_test[:,1])
    plt.show() 

    # Plot 2D log error 
    plotLogError2D(comb, observer)
