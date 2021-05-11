import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, linalg, integrate, interpolate
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.autograd import Variable
import torch

# Pipeline for observer simulation
def performMultipleLuenbergerSimulations(f,g,h,dimx,u,D,F,w0_array,nsims,tsim,dt):
    def dwdt(t,w):
        x = w[0:dimx]
        z = w[dimx:len(w)]
        x_dot = f(x) + g(x) * u(t)
        z_dot = np.matmul(z,np.transpose(D))+F*h(x)
        return np.concatenate((x_dot,z_dot))

    tspan = (0, tsim)
    tq = np.arange(0,tsim,dt)
    output_data = np.zeros(shape=(len(tq), dimx + D.shape[1], nsims))

    for i in range(nsims):
        w0 = w0_array[:, i]
        w = integrate.solve_ivp(dwdt, tspan, w0)
        wq = interpolate.interp1d(w.t, w.y)
        output_data[:,:,i] = np.transpose(wq(tq))

    return tq, output_data

    
# Pipeline for generating D matrix of observer
def generateCellEigComplexReal(eig_complex, eig_real):
    eigenCell = []

    for i in range(0,len(eig_complex), 2):
        array = np.zeros(shape=(2,2))
        array[0,0] = eig_complex[i].real
        array[0,1] = eig_complex[i].imag
        array[1,0] = eig_complex[i+1].imag
        array[1,1] = eig_complex[i+1].real
        eigenCell.append(array)
    
    for i in eig_real:
        array = np.zeros(shape=(1,1))
        array[0,0] = i.real
        eigenCell.append(array)

    return eigenCell

def generateLuenbergerD(eigen):
    eig_complex, eig_real = [x for x in eigen if x.imag!=0], [x for x in eigen if x.imag==0]

    if(any(~np.isnan(eig_complex))):
        eig_complex = sorted(eig_complex)
    
    eigenCell = generateCellEigComplexReal(eig_complex, eig_real)

    D = linalg.block_diag(*eigenCell[:])

    return D

# Define NN model
class Model(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, outputSize)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Normalizing function
def normalize(data):
  return (data-np.min(data))/(np.max(data) - np.min(data))

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
    epochs = 5
    batchSize = 5

    # Shape from (x,y,nsim) -> (x*nsim,y)
    data_frame = np.zeros((nsims*data.shape[0],data.shape[1]))
    for i in range(nsims):
      data_frame[i*data.shape[0]:(i+1)*data.shape[0],:] = data[:,:,i]
    data = data_frame.astype(np.float32)

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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.00

        print('====> Epoch: {} done!'.format(epoch))

    print('Finished Training')

    return T

if __name__ == "__main__":
    # Define plant model
    f = lambda x: np.array([x[1]**3, -x[0]]) 
    h = lambda x: x[0]
    g = lambda x: np.array([0,0])
    u = lambda x: 0

    # State dim
    dimx = 2
    
    # Lueneberger observer params
    b, a = signal.bessel(3,2*math.pi, 'low', analog=True, norm='phase')
    eigen = np.roots(a)

    D = generateLuenbergerD(eigen)
    F = np.array([1, 1, 1])

    # Simulation parameters for ODE
    nsims = 10
    tsim = 50
    dt = 1e-2

    # Initial conditions
    w0_array = np.zeros(shape=(dimx + D.shape[1], nsims));
    for i in range(nsims):
        w0_array[0,i] = 0.1*(i+1)

    # Generate training data
    tq, data = performMultipleLuenbergerSimulations(f,g,h,dimx,u,D,F, w0_array,nsims,tsim,dt)

    # Plot x_1 against x_2
    plt.plot(data[:,0,:], data[:,1,:])
    plt.show()

    # Bin initial data
    k=3
    t_c = 3/min(abs(linalg.eig(D)[0].real))
    idx = max(np.argwhere(tq < t_c))
    data = data[idx[0]-1:,:,:]

    # Train model x(t)=T*(z(t))
    dataSize = ((2,5),(0,2))
    inputSize = (3,2)
    T_star = computeNonlinearLuenbergerTransformation(tq,data,dataSize,inputSize)
    
    # Train z(t)=T(x(t))
    dataSize = ((0,2),(2,5))
    inputSize = (2,3)
    T = computeNonlinearLuenbergerTransformation(tq,data,dataSize,inputSize)
    
    # Compute test data
    w0_test = np.array([[np.random.uniform(0,1)-1], [np.random.uniform(0,1)-1],[0],[0],[0]])
    tq_test,w_test = performMultipleLuenbergerSimulations(f,g,h,dimx,u,D,F,w0_test,1,tsim,dt)
    
    # Data pipeline for z_hat
    data = w_test.reshape(w_test.shape[0], w_test.shape[1])
    inputs = data[:,:2]
    data = torch.from_numpy(inputs.astype(np.float32))
    data = Variable(data, requires_grad=False)
    
    # Sample data from T*
    T = T.to("cpu")
    z_hat = T(data)
    z_hat = z_hat.detach().numpy()
    
    # Plot z_1
    plt.plot(tq_test, z_hat[:,0])
    plt.plot(tq_test, w_test[:,2])
    
    # Plot z_2
    plt.plot(tq_test, z_hat[:,1])
    plt.plot(tq_test, w_test[:,2])
    
    # Plot z_3
    plt.plot(tq_test, z_hat[:,2])
    plt.plot(tq_test, w_test[:,3])
    plt.show()
    
    # Data pipeline x_hat
    data = w_test.reshape(w_test.shape[0], w_test.shape[1])
    inputs = data[:,2:]
    data = torch.from_numpy(inputs.astype(np.float32))
    data = Variable(data, requires_grad=False)
    
    # Sample data from T*
    T_star = T_star.to("cpu")
    x_hat = T_star(data)
    x_hat = x_hat.detach().numpy()
    
    # Plot x_1
    plt.plot(tq_test, x_hat[:,0])
    plt.plot(tq_test, w_test[:,0])
    
    # Plot x_2
    plt.plot(tq_test, x_hat[:,1])
    plt.plot(tq_test, w_test[:,1])
    plt.show()
    
    # Plot x_hat_1 against x_hat_2
    plt.plot(x_hat[:,0], x_hat[:,1])
    plt.show()