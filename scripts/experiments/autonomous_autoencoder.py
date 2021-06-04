import numpy as np
import torch
from lnos.net.train import trainAutoencoder
from lnos.net.helperfnc import generateTrainingData
from lnos.datasets.exampleSystems import createDefaultObserver
import sys
sys.path.append(sys.path[0]+'/../..')


def getOptions():
    """
    Configure model options for the experiment
    """
    options = {}

    options['batchSize'] = 10
    options['epochs'] = 100
    options['numHiddenLayers'] = 5
    options['sizeHiddenLayer'] = 30
    options['activation'] = 'tanh'
    options['reconLambda'] = .1
    options['isTensorboard'] = True

    options['simulationTime'] = 20
    options['simulationStep'] = 1e-2

    options['system'] = 'van_der_pohl'
    options['isAutonomous'] = False

    options['dataGen'] = 'pairs'
    options['sampling'] = 'lhs'
    options['gridSize'] = np.arange(-1, 1, 0.1)
    options['lhs_limits'] = np.array([[-1., 1.], [-1., 1.]])
    options['lhs_samples'] = 500

    return options


if __name__ == "__main__":

    options = getOptions()
    observer = createDefaultObserver(options)
    data = generateTrainingData(observer, options)
    trainAutoencoder(data, observer, options)
