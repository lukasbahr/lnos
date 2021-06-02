import sys ; sys.path.append(sys.path[0]+'/../..')
from lnos.datasets.exampleSystems import createDefaultObserver
from lnos.net.helperfnc import generateTrainingData
from lnos.net.train import trainAutoencoder
import torch
import numpy as np

def getOptions():
    """
    Configure model options for the experiment
    """
    options = {}

    options['batchSize'] = 10
    options['simulationTime'] = 20
    options['simulationStep'] = 1e-2
    options['isTensorboard'] = True
    options['reconLambda'] = .1
    options['epochs'] = 100
    options['gridSize'] = np.arange(-1, 1, 0.1)
    options['numHiddenLayers'] = 5
    options['sizeHiddenLayer'] = 30
    options['activation'] = 'tanh'

    return options


if __name__ == "__main__":
    options = getOptions()
    observer = createDefaultObserver()
    data = generateTrainingData(observer,options)
    trainAutoencoder(data, observer, options)


