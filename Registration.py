
import sys
import os
import pickle
from copy import deepcopy

from framework.Utils import transform as tf

import numpy as np

# SimpleITK package and tools
import SimpleITK as sitk
from framework.Utils.Registration import sitk_tools as st

class RegistrationOutput(dict):
    '''
    Container class for the output of a registration process
    '''
    def __init__(self, transform=None, registration_method=None,  metric_values=[], multires_iterations=[]):
        self.transform = transform
        self.registration_method = registration_method
        self.metric_values = metric_values
        self.multires_iterations = multires_iterations
        pass

    def update_metrics(self, registration_method):
        self.metric_values.append(registration_method.GetMetricValue())

    def update_multires_iterations(self):
        self.multires_iterations.append(len(self.metric_values))

class Registration(object):
    '''
    Registration base class. Contains the general data structure, functions, and utility methods to implement different registration algorithms.

    Positional arguments:

    '''
    def __init__(self, dbconn, fixed_patient, moving_patient, roi_list, use_surfaces=False, sampling=None, crop=False):

        self.registration_type = None

        self.dbconn = dbconn
        self.fixed_patient = fixed_patient
        self.moving_patient = moving_patient

        self.roi_list = roi_list
        # Boolean flag on whether or not to use surface mask
        self.use_surfaces = use_surfaces
        # Sampling rates
        self.sampling = sampling
        # Boolean flag on whether or not to crop to non-zero bounds
        self.crop = crop

        # Lists to store masks, images, and point clouds
        self.masks = {}
        self.images = {}
        self.clouds = {}

        # Store other metrics (runtime, error per iteration)
        self.metrics = {
            "runtime": None,
            "error": []
        }
        self.params = {}
        pass

    @staticmethod
    def read(filename):
        fhandle = open(filename, 'r')
        reg_obj = pickle.load(fhandle)
        fhandle.close()
        return reg_obj

    def write(self, filename, output_dir='./output'):
        # Create the output directory if it isn't already there
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Cannot serialize Database connections, so pull it out...
        tempdb = self.dbconn
        self.dbconn = None
        # Serialize and write to file
        fhandle = open(output_dir + '/' + str(filename) + '.oaf', 'w')
        pickle.dump(self, fhandle)
        # ...and put it back
        self.dbconn = tempdb
        fhandle.close()
        pass

    def get_mask(self, prep_id):
        # If the mask was already queried...
        if prep_id in self.masks.keys():
            return self.masks[prep_id]
        # If the mask was not previously queried
        else:
            # Get all ROIs for a patient
            masks = []
            for reg in self.roi_list:
                roiID = self.dbconn.RegionsOfInterest.GetIDByPatientRepIDName(prep_id, reg)
                roi = self.dbconn.RegionsOfInterest.GetMask(roiID)
                # Get the masks
                if self.use_surfaces:
                    masks.append(roi.mask.get_mask_edge_voxels())
                else:
                    masks.append(roi.mask)
            # Get the combined mask for all ROIs
            mymask = tf.combine_masks(masks) if len(masks) > 1 else masks[0]
            # Perform any preprocessing (cropping, sampling)
            if self.crop:
                mymask = tf.crop(mymask)
            if self.sampling:
            # if self.sampling and prep_id != self.fixed_patient:
                mymask = tf.downsample(mymask, self.sampling)
            return mymask

    def get_masks(self):
        # If both masks have not been queried...
        if len(self.masks.keys()) < 2:
            self.masks = {p: self.get_mask(p) for p in [self.fixed_patient, self.moving_patient]}
        # Return the dictionary of masks
        return self.masks

    def make_image(self, prep_id):
        '''
        For a given patient, get all relevant masks, combine them, and make an SITK image.

        Keyword arguments:
            :prep_id:   patient representation id

        Returns:
            SimpleITK image of the joined masks, with datatype sitkFloat32
        '''
        # Get all ROIs for a patient
        masks = []
        for reg in self.roi_list:
            roiID = self.dbconn.RegionsOfInterest.GetIDByPatientRepIDName(prep_id, reg)
            roi = np.append(self.dbconn.RegionsOfInterest.GetMask(roiID))
            # Get the masks
            if self.use_surfaces:
                masks.append(roi.mask.get_mask_edge_voxels())
            else:
                masks.append(roi.mask)
        # Get the combined mask for all ROIs
        mymask = tf.combine_masks(masks) if len(masks) > 1 else masks[0]
        # Make it an image
        img = st.image_from_mask(mymask)
        return sitk.Cast(img, sitk.sitkFloat32)

    def get_images(self):
        # If both images have not been made...
        if len(self.images.keys()) < 2:
            self.images = {p: self.make_image(p) for p in [self.fixed_patient, self.moving_patient]}
        # Return the dictionary of images
        return self.images

    def make_point_cloud(self, prep_id):
        # If the point cloud was already calculated
        if prep_id in self.clouds.keys():
            return self.clouds[prep_id]
        # If not, calculate and store it
        else:
            mymask = self.get_mask(prep_id)
            msk_idx = mymask.data.nonzero()
            msk_idx = np.fliplr(np.asarray(msk_idx).T)
            msk_pts = mymask.transform_index_to_physical_point(msk_idx)
            self.clouds[prep_id] = msk_pts
            return msk_pts

    def get_point_clouds(self):
        # If both point clouds have not been made...
        if len(self.clouds.keys()) < 2:
            self.clouds = {p: self.make_point_cloud(p) for p in [self.fixed_patient, self.moving_patient]}
        # Return the dictionary of point clouds
        return self.clouds


    # Added by Arun
    def get_point_clouds2(self):
        # Make only the point cloud for the moving patient.
        if len(self.clouds.keys()) < 2:
            self.clouds = {p: self.make_point_cloud(p) for p in [self.moving_patient]}
        # Return the dictionary of point clouds
        return self.clouds

    def preprocess(self):
        raise NotImplementedError("Registration subclasses must implement their own preprocessing steps.")

    def register(self):
        raise NotImplementedError("Registration subclasses must implement their own registration algorithms.")

    def plot(self):
        raise NotImplementedError("Registration subclasses must implement their own visualization algorithms.")
