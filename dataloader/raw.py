import datetime as dt
import glob
import os
from collections import namedtuple
import numpy as np
import matplotlib.image as mpimg

def transform_from_rot_trans(R, t):
    '''transformation matrix from rotation matrix and translation vector
    '''
    R = R.reshape(3,3)
    t = t.reshape(3,1)
    return np.vstack((np.hstack([R,t]), [0,0,0,1]))

def read_calib_file(filepath):
    '''read in a calibration file and parse into a dictionary'''
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # the only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
        return data

def load_velo_scans(velo_files):
    '''helper method to parse velodyne files into a list of scans
    '''
    scan_list = []
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        scan_list.append(scan.reshape((-1, 4)))
    
    return scan_list

class RAW:
    '''
    load and parse raw data into a usable format
    '''
    def __init__(self, base_path, date, drive, frame_range=None):
        self.drive = date + "_drive_" + drive +"_sync"
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, self.drive)
        self.frame_range = frame_range

    def _load_calib_rigid(self, filename):
        '''read a rigid transformation calibration file as numpy.array
        '''
        filepath = os.path.join(self.calib_path,filename)
        data = utils.read_calib_file(filepath)
        return transform_from_rot_trans(data['R'], data['T'])

    def _load_calib_cam_to_cam(self, velo_to_cam_file, cam_to_cam_file):
        # we'll return the camera calibration as a dictionary
        data = {}

        # load the rigid transformation from velodyne coordinates
        # to unrectified cam0 coordinates 
        T_cam0unrect_velo = self._load_calib_rigid(velo_to_cam_file)

        cam_to_cam_filepath = os.path.join(self.calib_path, cam_to_cam_file)
        filedata = read_calib_file(cam_to_cam_filepath)

        # create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P_rect_00'], (3,4))
        P_rect_10 = np.reshape(filedata['P_rect_01'], (3,4))
        P_rect_20 = np.reshape(filedata['P_rect_02'], (3,4))
        P_rect_30 = np.reshape(filedata['P_rect_03'], (3,4))

        # create a 4x4 matrix from the rectifying rotation matrix
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3,3))

        # compute the rectified extrinsics from cam0 to camN

        # compute the velodyne to rectified camera coordinate transform

        # compute the camera intrinsics



