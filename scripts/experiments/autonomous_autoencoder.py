import sys ; sys.path.append(sys.path[0]+'/../..')
from lnos.observer.lueneberger import LuenebergerObserver
from lnos.datasets.exampleSystems import getAutonomousSystem
from lnos.net.helperfnc import generateTrainingData
from lnos.net.train import trainAutoencoder
import torch
import numpy as np

def getOptions():
    """
    Configure model options for the experiment
    """
    options = {}

    options['batchSize'] = 5
    options['simulationTime'] = 20
    options['simulationStep'] = 1e-2
    options['isTensorboard'] = True
    options['epochs'] = 50
    # Size of the uniform grid in 2D -> [start,stop,step]
    options['gridSize'] = np.arange(-1, 1, 0.4)

    return options

def createObserver():
    f, h, g, u, dim_x, dim_y, eigen = getAutonomousSystem()

    # Initiate observer with system dimensions
    observer = LuenebergerObserver(dim_x, dim_y, f, g, h, u)

    # Set system dynamics
    observer.D = observer.tensorDFromEigen(eigen)
    observer.F = torch.Tensor([[1.0], [1.0], [1.0]])

    return observer

if __name__ == "__main__":
    options = getOptions()
    observer = createObserver()
    data = generateTrainingData(observer,options)
    trainAutoencoder(data, observer, options)


