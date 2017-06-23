import os
import numpy as np
import scipy.io as sio
import time


def register(atlcld_before, atlcld_after, ptcld, loadFromMat=False, filename=None, saveToMat=False, patID=-1, atlasname=None):
    """
    Perform a thin plate spline warping to gain a mapping function
    :param patlcld_before: the atlas point cloud before registration to the patient
    :param atlcld_after: the atlas point cloud after registration to target patient
    :param ptcld: the sector point cloud in atlas coordinate frame
    :param loadFromMat: if True, loads parameters from file
    :param filename: filename to load parameters from
    :param saveToMat: if True, saves parameters to file
    :param patID: patient ID, used for creating filename
    :param atlasname: atlas name used for creating the filename for saving parameters
    :type patcld_before: numpy.array Nx3
    :type atlcld_after: numpy.array Nx3
    :type ptcld: numpy.array Dx3
    :type loadFromMat: bool
    :type filename: str
    :type saveToMat: bool
    :type patID: int
    :type atlasname: str
    :return: the sector point cloud in patient coordinate frame 
    :return: runtime
    :rtype: numpy.array Dx3
    :rtype: tim: float
    """
    print('Starting TPS...')
    starttime = time.clock()
    state = 1
    if loadFromMat:
        if filename:
            x = sio.loadmat(filename)
            param = x['parameters']
            state = 0
        else:
            print('Filename must be specified.')
    npts = atlcld_before.shape[0]  # number of points in the interested points array]
    if state == 1:
        kernel = np.zeros((npts, npts))
        for i in range(npts):
            for j in range(npts):
                kernel[i, j] = np.sum(np.square(atlcld_before[i, :] - atlcld_before[j, :])) # Setting up kernel matrix
                kernel[j, i] = kernel[i, j]
        # Calculate kernel function R
        kernel[kernel < 1e-320] = 1e-320
        k = np.sqrt(kernel)
        p = np.concatenate((np.ones((npts,1)), atlcld_before), axis=1)
        l = np.concatenate((k, p), 1)
        temp = np.concatenate((p.T, np.zeros((4,4))), 1)
        l = np.append(l, temp, 0)
        ctrl = np.append(atlcld_after, np.zeros((4, 3)), 0)
        param = np.dot(np.linalg.pinv(l), ctrl)

    print('Transforming Points...')

    ptnsNum = ptcld.shape[0]
    k = np.zeros((ptnsNum, npts))
    gx = ptcld[:, 0]
    gy = ptcld[:, 1]
    gz = ptcld[:, 2]
    for i in range(npts):
        k[:, i] = (np.square(gx - atlcld_before[i, 0]) + np.square(gy - atlcld_before[i, 1]) + np.square(gz - atlcld_before[i, 2])) # R ^ 2
    k[k < 1e-320] = 1e-320
    k = np.sqrt(k)
    gx = gx.reshape((len(gx), 1))
    gy = gy.reshape((len(gy), 1))
    gz = gz.reshape((len(gz), 1))
    p = np.concatenate((np.ones((ptnsNum, 1)), gx, gy, gz), 1)
    l = np.concatenate((k, p), 1)
    wobject = np.dot(l, param)
    wobject[:, 0] = np.multiply(np.round(np.multiply(wobject[:,0], 10**3)),10**-3)
    wobject[:, 1] = np.multiply(np.round(np.multiply(wobject[:,1], 10**3)),10**-3)
    wobject[:, 2] = np.multiply(np.round(np.multiply(wobject[:,2], 10**3)),10**-3)
    tim = time.clock() - starttime
    print('TPS Runtime is: ' + str(tim))
    if saveToMat:
        if patID == -1 or atlasname == None:
            print('Both patID and atlasname must be specified.')
        else:
            output_dir = './TPS'
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            sio.savemat(output_dir + '/TPS-' + atlasname + '-' + str(patID) + '.mat', {'parameters': param})
    return wobject, tim


