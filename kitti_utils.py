from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

'''
2011_09_26/
    calib_cam_to_cam.txt
    calib_imu_to_velo.txt
    calib_velo_to_cam.txt
    2011_09_26_drive_0027_sync/
        image_00/
        image_01/
        image_02/
        image_03/
        oxts/
        velodyne_points/
'''

#root_dir = os.path.dirname(os.path.abspath(__file__))
#data_dir = os.path.joint(os.path.dirname(root_dir), 'data')
#image_shape = 375, 1242

def load_velodyne_points(filename):
    '''
    filename: 2011_09_26_drive_0027_sync/velodyne_points/data/0000000026.bin
    '''
    points = np,fromfile(filename, dtype=np.float32).reshape(-1,4)
    points = points[:, :3] # exclude luminance
    return points

def homogeneous_transform(points, transform):
    '''
    parameters
    ----------
    points: (n_points, M) array-like 
        the points to transform. If 'points' is shape (n_points, M-1), a unit homogeneous coordinate will be added to make it (n_points, M)
    transform: (M,N) arrray-like
        the right-multiplying transformation to apply.
    '''


def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    # set(iterable) 
    # returns an empty set if no parameters are passed
    # a set constructed from the given iterable parameter

    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # string.split(separator, maxsplit)
            # separator: by defaul any whitespace is a separator
            # maxsplit: specifies how many splits to do. default -1.
            value = value.strip()
            # string.strip(characters)
            # removes any leading (spaces at the beginning) and trailing (spaces at the end) characters 
            # characters: optional, a set of characters to remove as leading characters
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                # map(func, iter)
                #returns a map object (which is an iterator) of the results after applying the given function to each item of a given iterable
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # data[key] already eq. value, so pass
                    pass 
                    # ValueError: raised when an operation or function receives an argument that has the right type but an inappropriate value
    return data

def sub2ind(matrixSize, rowSub, colSub):
    '''convert row, col matrix subscripts to linear indices
    '''
    m, n = matrixSize
    return rowSub * (n-1) + colSub -1 


