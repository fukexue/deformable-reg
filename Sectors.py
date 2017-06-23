"""
Sectors.py
Authors: Chris Micek, Arun Raghavan, Julie Shade
Updated: May 2017

Functions to define regions in atlas coordinate space, register to patient coordinate space, and return dose inside.
Allows for definition of grid with any spacing/size over regions of atlas coordinate space.
"""

import framework.OncospaceConnect as oc
from framework.Normalization.statistical_atlas import AtlasCalculator
import time, TPS, random, warnings
from framework.Utils import transform as tf
import framework.Normalization.cpd as cpd
from framework.Utils import visualize as vs
from framework.Utils import image, dose_map as dmap
import numpy as np
import copy as cp
import scipy.io as sio
import db_helper
import os

warnings.filterwarnings('ignore')


def get_atlas(filename):
    """
    Read an atlas from a file and return it as an atlas object.
    :param filename: .oaf file containing atlas
    :type filename: str
    :return: AtlasCalculator object of stored atlas
    :rtype: AtlasCalculator
    """
    return AtlasCalculator.read(filename)


def query_test_patient(reg_names, num=1):
    """
    Returns a random patient from the Oncospace db that has all of the regions of interest contoured.
    :param reg_names: List of regions of interest
    :param num: The number of patients to choose
    
    :type reg_names: [str]
    :type num: int
    :return: random patient that has all of regions of interest contoured
    :rtype: list(int)
    """
    db = oc.Database('{SQL Server}', 'rtmw-oncodb.radonc.jhmi.edu', 'OncospaceHeadNeck', 'oncoguest', '0ncosp@ceGuest')
    print "Database connection successful."
    pat_list = db.RegionsOfInterest.GetPatientRepIDWithRois(reg_names)
    return random.sample(pat_list, num)


def query_dose_grid(patientRepID, rad_ID_num=-1):
    """
    Returns the dose grid for a given patient ID, default is first radiotherapy session if no session specified.
    :param patientRepID: Patient ID from Oncospace database
    :param rad_ID_num: Radiotherapy session number
    :type patientRepID: int
    :type rad_ID_num: int
    :return: Dose grid for specified patient/radiotheapy session
    :rtype: Dictionary with:
                :key:   RadiotherapySession.ID (rad_ID_num)
                :value: a dose.dose() instance
    """
    db = oc.Database('{SQL Server}', 'rtmw-oncodb.radonc.jhmi.edu', 'OncospaceHeadNeck', 'oncoguest', '0ncosp@ceGuest')
    print "Database connection successful."
    rad_IDS = db.RadiotherapySessions.GetIDByPatientRepID(patientRepID)
    if rad_ID_num == -1:
        rad_session_ID = random.choice(rad_IDS)
    else:
        rad_session_ID = rad_IDS[rad_ID_num]
    return db.RadiotherapySessions.GetDoseGrid(rad_session_ID)


def register_atlas_to_patient(fixed_patient_cld, patID, roi_list, saveToMat=False, atlasname=None):
    """
    Register an atlas to a patient and return registered point cloud and information from registration method.
    Note: uses registration method from /normalization/cpd/cpd_nonrigid.py where t = y + g*wc
    :param fixed_patient_cld: point cloud representation atlas ("fixed patient")
    :param patientID: ID of patient to register atlas to
    :param roi_list: list of organs from moving patient to include
    :param saveToMat: Flag indicating whether to save the atlas point cloud and registration return values to a .mat 
                      file. The file holds a dict containing:
                      'atlas': The atlas point cloud
                      'transformed': The transformed points (patient transformed to atlas)
                      'g': G matrix from CPD calculation; see framework.Normalization.cpd.cpd_nonrigid for details
                      'wc': matrix of coefficients
                      'errors': Variance of position differences between point correspondences
    :param atlasname: If not None, includes this atlas name in the name of the .mat file saved.
    
    :type fixed_patient_cld: numpy.ndarray(numpy.float64), N x 3
    :type patientID: int
    :type roi_list: list(str)
    :type saveToMat: bool
    :type atlasname: None | str
    
    :return: A tuple containing:
            T: transformed points (patient transformed to atlas)
            g: G matrix from CPD calculation; see framework.Normalization.cpd.cpd_nonrigid for details
            wc: matrix of coefficients
            errors: Variance of position differences between point correspondences
    
    :rtype: tuple(numpy.ndarray(numpy.float64)) N x 3, numpy.ndarray(numpy.float64), numpy.ndarray(numpy.float64), 
            list(numpy.ndarray(numpy.float64)))
    """
    # Calculate the center of mass of the points
    y = fixed_patient_cld
    yc = np.mean(y, axis=0)
    mymask = get_mask(patID, roi_list)
    msk_idx = mymask.data.nonzero()
    msk_idx = np.fliplr(np.asarray(msk_idx).T)
    msk_pts = mymask.transform_index_to_physical_point(msk_idx)
    xc = np.mean(msk_pts, axis=0)
    x = msk_pts
    # And the difference between the two
    cdiff = xc - yc
    # Move the points in Y to be centered at X
    ret = cp.deepcopy(y)
    for idx, yi in enumerate(ret):
        ret[idx] = yi + cdiff
    X = x
    Y = ret
    start_time = time.time()
    T, g, wc, errors = cpd.register_nonrigid(X, Y, 0.0, lamb=3.0, beta=2.0, plateau_thresh=1.0e-3,
                                             plateau_length=5)
    runtime = time.time() - start_time
    print('Runtime is: ')
    print(runtime)
    output_dir = './Registrations'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if saveToMat:
        name = output_dir + '/Reg-'
        if atlasname:
            name += atlasname + '-' + str(patID) + '.mat'
        else:
            name + str(patID) + '.mat'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        sio.savemat(name, {'atlas': fixed_patient_cld, 'transformed': T, 'g': g, 'wc': wc, 'errors': errors})
    return T, g, wc, errors, x


def get_mask(patID, roi_list, sampling=(0.25, 0.25, 1), crop=True):
    """
    Get a combined binary mask with all the desired ROIs for the patient.
    :param patID: The patient representation ID of the patient to get a mask from
    :param roi_list: The list or ROI names to include in the mask
    :param sampling: Tuple (x, y, z) of fraction of points in each dimension to use for downsampling
    :param crop: Boolean indicating whether to crop the resulting mask
    
    :type patID: int
    :type roi_list: list(str)
    :type sampling: tuple(float)
    :type crop: bool
    
    :return: The binary mask representation of the patient
    :rtype: numpy.ndarray(numpy.float64) X x Y x Z
    """
    # Get all ROIs for a patient
    db = oc.Database('{SQL Server}', 'rtmw-oncodb.radonc.jhmi.edu', 'OncospaceHeadNeck', 'oncoguest', '0ncosp@ceGuest')
    print "Database connection successful."
    masks = []
    mymask = None
    for reg in roi_list:
        roiID = db.RegionsOfInterest.GetIDByPatientRepIDName(patID, reg)
        if roiID is None:
            continue
        roi = db.RegionsOfInterest.GetMask(roiID)
        # Get the masks
        masks.append(roi.mask.get_mask_edge_voxels())
        # Get the combined mask for all ROIs
        mymask = tf.combine_masks(masks) if len(masks) > 1 else masks[0]
        # Perform any preprocessing (cropping, sampling)
        if crop:
            mymask = tf.crop(mymask)
        if sampling:
            # if self.sampling and prep_id != self.fixed_patient:
            mymask = tf.downsample(mymask, sampling)
    return mymask


def simple_grid(x, y, z, center, spacing):
    """
    Return a grid with specified center and x, y, and z lengths
    :param x: length along x axis
    :param y: length along y axis
    :param z: length along z axis
    :param center: center of the volume
    :param spacing: spacing between each point
    :type x: float
    :type y: float
    :type z: float
    :type center: list(float)
    :type spacing: tuple(float)
    :return: grid
    :rtype: numpy.ndarray
    """
    xspacing, yspacing, zspacing = spacing[0], spacing[1], spacing[2]

    x = float(x - x % xspacing)
    y = float(y - y % yspacing)
    z = float(z - z % zspacing)
    max_x = float(center[0] + float(x) / 2.0)
    min_x = float(center[0] - float(x) / 2.0)
    max_y = float(center[1] + float(y) / 2.0)
    min_y = float(center[1] - float(y) / 2.0)
    max_z = float(center[2] + float(z) / 2.0)
    min_z = float(center[2] - float(z) / 2.0)
    if ((x / xspacing) + 1) % 2 == 0:
        center[0] = center[0] - xspacing / 2.0
    if (y / yspacing + 1) % 2 == 0:
        center[1] = center[1] - yspacing / 2.0
    grid = np.array(center)
    if type(center) is not list:
        center = [_[0] for _ in center]
    p2 = center
    p2[0] = min_x
    while p2[0] <= max_x:
        grid = np.append(grid, simple_grid_helper(p2, spacing, min_y, max_y, min_z, max_z), 0)
        p2[0] += xspacing
    grid = grid[-((grid.shape[0]) - 3):]
    grid = grid.reshape((grid.shape[0] / 3), 3)
    return grid


def simple_grid_helper(center, spacing, min_y, max_y, min_z, max_z):
    """
    Appends an array around the given center points from the min_y to max_y and min_z to max_z points with the
    given spacing.
    :param center: center point to build array around
    :param spacing: distance between points
    :param min_y: minimum y point
    :param max_y: maximum y point
    :param min_z: minimum z point
    :param max_z: maximum z point

    :type center: list(float)
    :type spacing: tuple(float)
    :type min_y: float
    :type max_y: float
    :type min_z: float
    :type max_z: float

    :return: array around given center point
    :rtype: np.array(float64)
    """
    grid = center

    xspacing, yspacing, zspacing = spacing[0], spacing[1], spacing[2]

    for i in np.arange(min_y, max_y + yspacing, yspacing):
        curr = np.copy(center)
        curr[1] = i
        for j in np.arange(min_z, max_z + zspacing, zspacing):
            curr[2] = j
            grid = np.append(grid, curr)
    return grid[-((grid.shape[0]) - 3):]


def getSectorMap(filename, sector, roi_list, visualize=False, patID=-1, saveToMat=False):
    """
    Get an atlas point cloud registered to a patient with a transformed sector delineation
    :param filename: the filename of the catlas
    :param sector:  a point cloud representing the sector in atlas coordinate frame
    :param roi_list: a list of the regions of interest for the patient
    :param visualize: if True, the method will plot the before and after transformations
    :param patID: patient ID, if left empty, a random patient will be chosen using the roi_list
    :param saveToMat: by default, false, but if true, will save to a .mat file called sectors.mat
    :type filename: str
    :type sector: numpy.array N x 3
    :type roi_list: list[str]
    :type visualize: bool
    :type patID: int
    :type saveToMat: bool
    :return: A point cloud of the atlas and the sector together and as separate objects
    :rtype: numpy.array N x 3
    """
    atlas = get_atlas(filename)
    atlas = atlas.iterations[-1]
    print atlas.shape
    length = sector.shape[0]
    if patID == -1:
        patID = query_test_patient(roi_list)[0]
    fixed_atlas = atlas
    print('Starting Registration...')
    transformed, g, wc, errors, patient = register_atlas_to_patient(fixed_atlas, patID, roi_list)
    print('Registration finished.')
    wobject = TPS.register(fixed_atlas, transformed, sector)
    print('TPS finished.')
    atlas_after = np.append(transformed, wobject, 0)
    if visualize:
        print('Visualizing')
        atlas_before = np.append(atlas, sector, 0)
        vs.visualizePointCloud(atlas_before, 'Atlas with delineated sector', alpha=0.3, num_points=length, state=1)
        vs.visualizePointCloud(atlas_after, 'Atlas with sector in patient coordinate frame', alpha=0.3,
                               num_points=length, state=1)
    if saveToMat:
        sio.savemat('sectors.mat', {'atlas': atlas, 'transformed': transformed, 'sector': sector, 'ptsector': wobject})
    return atlas_after, wobject, transformed


def get_dose_inside_sector(patID, mode='explicit', *args, **kwargs):
    """
    Returns the dose inside a rectangular prism (or tetrahedral prism). This can either be defined implicitly in the 
    patient space, or explicitly after an external deformation.

    :param patID: patient representation ID
    :param mode: Choice of sector grid definition mode. Default is 'explicit', which accepts a dense point cloud 
                 representing the region to be queried as a positional argument. Alternative is 'implicit', 
                 which defines a center point and distances along x, y, and z axes.
    :param args: Positional arguments. If mode is 'explicit', the only argument is the point cloud defining the 
                 queried region:
                    sector_cloud: The dense point cloud representing the sector where dose information is desired.
                                  (likely from SectorUtils.find_points_in_def_cube)
                 If the mode is 'implicit', the accepted arguments are:
                    x: length along x axis
                    y: length along y axis
                    z: length along z axis
                    center: center of the volume
    :param kwargs: Keyword arguments:
        rad_ID_num: Radiation therapy session number to query; default is the first

    :type patID: int
    :type mode: str
    :type args: list(numpy.ndarray([[numpy.float64]])) | list(float, float, float, tuple(float))
    :type kwargs: None | dict(str, int)

    :return: A dose_map.dose_mask object instance containing dose information within the region specified by 
             cornerpoints
    :rtype: framework.Utils.dose_map.dose_mask
    """
    rad_ID_num = -1
    if 'rad_ID_num' in kwargs:
        rad_ID_num = kwargs['rad_ID_num']

    try:
        patient_mask = _get_mask_helper(patID, 'mandible')
    except ValueError:
        patient_mask = None
    if patient_mask is None:
        return None
    if mode == 'implicit':
        x, y, z, center = args[0], args[1], args[2], args[3]
        sector_cloud = simple_grid(x, y, z, center, patient_mask.spacing)

    elif mode == 'explicit':
        sector_cloud = args[0]

    else:
        raise ValueError("Mode requested is undefined")

    dgrid = query_dose_grid(patID, rad_ID_num)
    sector_grid_mask = db_helper.cloud_to_mask(sector_cloud, None, patient_mask)
    dmask = dmap.dose_mask(sector_grid_mask, dgrid)
    vs.visualizeDoseMap(dmask)
    return dmask


def _get_mask_helper(prepID, reg):
    """
    Helper function for get_dose_inside_sector. Returns a mask object at full resolution for a single region of 
    interest in the given patient.
    
    :param prepID: The patient representation ID of the patient to query
    :param reg: The ROI name of the region to get
    
    :type prepID: int
    :type reg: str
    
    :return: A fully-sampled mask of the queried region for the given patient
    :rtype: framework.Utils.image.mask
    """

    # Get the ROI ID by name
    db = oc.Database('{SQL Server}', 'rtmw-oncodb.radonc.jhmi.edu', 'OncospaceHeadNeck', 'oncoguest', '0ncosp@ceGuest')
    roiID = db.RegionsOfInterest.GetIDByPatientRepIDName(prepID, reg)

    # Query the mask

    r = db.RegionsOfInterest.GetMask(roiID)

    return r.mask


# def update_sector_mask(mask):
#     im = mask.data
#     nzz, nzy, nzx = im.nonzero()
#     minz = np.min(nzz)
#     maxz = np.max(nzz)
#     miny = np.min(nzy)
#     maxy = np.max(nzy)
#     minx = np.min(nzx)
#     maxx = np.max(nzx)
#
#     im = im[minz:maxz+1, miny:maxy+1, minx:maxx+1]
#     mask.set_origin(mask.transform_index_to_physical_point([minx, miny, minz]))
#     mask.set_image(im)
#     return mask


def get_corner_points(indices, grid, x, y, z, spacing):
    """
    Returns the 8 corner points for the cube in the grid indicated by cube index. Indices should be given in x, y, z
    order and begin from the -x, -y, -z corner of the grid.
    :param indices: [x, y, z] indices of the desired section of the grid. indexed 0 to n-1
    :param grid: point cloud representing a once-regular grid that has been transformed into patient coordinate space
                or a regular grid
    :param x: length along x axis (BEFORE transformation)
    :param y: length along y axis (BEFORE transformation)
    :param z: length along z axis (BEFORE transformation)
    :param spacing: tuple representing spacing along x, y, z axes. can also be int representing same spacing along each
                    axis.
    :type indices: tuple of ints or np.array(int)
    :type grid: np.array(float64)
    :type x: float
    :type y: float
    :type z: gloat
    :type spacing: tuple of floats (3x1)

    :return: point cloud with the 8 corner points of the desired section
    :rtype: np.array(float64) 3 x 8
    """
    if type(spacing) == int or type(spacing) == float:
        spacing = [spacing, spacing, spacing]

    xpts = int(x / spacing[0]) + 1
    ypts = int(y / spacing[1]) + 1
    zpts = int(z / spacing[2]) + 1

    if indices[0] > (xpts - 2) or indices[1] > (ypts - 2 or indices[2] > (zpts - 2)):
        print "Cube indices exceed grid limits."
        return None

    x_min_index = indices[0] * ypts * zpts
    x_max_index = (indices[0] + 1) * ypts * zpts
    y_min_index = indices[1] * zpts
    y_max_index = (indices[1] + 1) * zpts

    cornerpts = np.array([grid[x_min_index + y_min_index + indices[2]],
                          grid[x_min_index + y_min_index + indices[2] + 1],
                          grid[x_min_index + y_max_index + indices[2]],
                          grid[x_min_index + y_max_index + indices[2] + 1],
                          grid[x_max_index + y_min_index + indices[2]],
                          grid[x_max_index + y_min_index + indices[2] + 1],
                          grid[x_max_index + y_max_index + indices[2]],
                          grid[x_max_index + y_max_index + indices[2] + 1]])

    return cornerpts


def downsample_pc(msk_pts, samplefrac):
    """
    Uniformly down-sample a point cloud
    :param msk_pts: point cloud to be down-sampled
    :param samplefrac: fraction of points to be left in msk_pts, either a single float or a tuple(x, y, z)
    
    :type msk_pts: numpy.ndarray(numpy.float64) N x 3
    :type samplefrac: float | tuple(float)
    
    :return: down-sampled msk_pts
    :rtype: numpy.ndarray(numpy.float64) N x 3
    """

    fracs = np.multiply(np.ones(3), samplefrac)
    if not all(b <= 1 for b in fracs):
        raise ValueError("Fraction parameter(s) must be <= 1.")

    skips = [int(1 / i) for i in fracs]
    return msk_pts[::skips[0], :]


def getRegistrationFile(filename):
    """
    Load registration data from a .mat file
    :param filename: location and name of file
    :type filename: str
    :return: all the data within the file
    """
    data = sio.loadmat(filename)
    return data['atlas'], data['transformed'], data['g'], data['wc'], data['errors']


def getGridDataFromfile(filename):
    """
    Get the grid data from a saved file
    :param filename: the name and location of the file
    :type filename: str
    :return: transformed grid and original grid as well as runtime
    """
    data = sio.loadmat(filename)
    return data['tgrid'], data['orig_grid'], data['runtime']


def main():
    # filename = '.\output\l_eye_r_eye_mandible_l_parotid_r_parotid_brain_brainstem_genioglossus_muscle_geniohyoid_muscle_hyoglossus_muscle_30_pat.oaf'
    # filename = '.\Registrations\Reg-BigAtlas-444.mat'
    # roi_list = ['l_eye', 'r_eye', 'mandible', 'r_parotid', 'l_parotid', 'brain', 'brainstem', 'genioglossus_muscle', 'geniohyoid_muscle', 'hyoglossus_muscle']
    # patID = query_test_patient(roi_list)
    # atlas = get_atlas(filename)
    # results = register_atlas_to_patient(downsample_pc(atlas.iterations[-1], [.5]), patID, roi_list, saveToMat=True, atlasname='BigAtlas')
    # data = getRegistrationFile(filename)
    # atlas = data[0]
    # transformed = data[1]
    # vs.visualizePointCloud(atlas, 'Atlas', alpha=0.3)
    # x = np.absolute(16)
    # y = np.absolute(25)
    # z = np.absolute(20)
    # #grid = simple_grid(x, y, z, np.mean(atlas, axis=0), 2)
    # tgrid, grid, runtime = getGridDataFromfile('transformedgrid-444.mat')
    # vs.visualizePointCloud(np.append(atlas, grid, axis=0), 'atlas with grid', alpha=0.3, num_points=grid.shape[0], state=1, pt_size=0)
    # ntgrid, runtime = TPS.register(atlas, transformed, grid, loadFromMat=True, filename='./TPS/BigAtlas-444.mat')
    # vs.visualizePointCloud(np.append(transformed, tgrid, axis=0), 'transformed atlas with grid', alpha=0.3, num_points=tgrid.shape[0],
    #                        state=1, pt_size=0)
    # sio.savemat('transformedgrid-444.mat', {'tgrid': tgrid, 'orig_grid': grid, 'runtime': runtime})
    # t = downsample_pc(atlas.iterations[-1], [.25])
    # vs.visualizePointCloud(t, 'Downsampled Atlas', alpha=0.3)
    # center = np.array([0,0,0])
    # grid = simple_grid(5, 5, 5, center, .25)
    # num_pts, throw = grid.shape
    # trans = np.append(t, grid, axis=0)
    # vs.visualizePointCloud(trans, 'Atlas with grid', alpha=0.1, num_points=num_pts, state=1, pt_size=0)

    # sgrid = get_sector_grid(msk_pts, spacing=2)
    # atlas_after, wobject, transformed = getSectorMap(filename, sgrid, roi_list, visualize=True, saveToMat=False)
    # get_corner_points([0, 0, 0], wobject)


    # organgrid = np.append(msk_pts, sgrid)
    # organgrid = organgrid.reshape((organgrid.shape[0]/3, 3))
    # vs.visualizePointCloudByOrgan(organgrid, [msk_pts.shape[0], sgrid.shape[0]], alpha=0.3, title='Mask with Grid')

    get_dose_inside_sector(216, 'implicit', 15, 15, 15, [3, -60, 20])


if __name__ == '__main__':
    main()
