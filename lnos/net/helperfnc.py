import numpy as np

def splitDataShifts(data, num_shifts):
    #TODO fix that

    n = data.shape[1]

    numTraj = int(data.shape[0] / num_shifts)

    dataTensor = np.zeros([num_shifts + 1, numTraj, n])

    for j in range(num_shifts):
        dataTensor[j, :, :] = data[j*numTraj:(j+1)*numTraj, :]

    return dataTensor