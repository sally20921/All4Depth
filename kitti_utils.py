from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from collections import Counter

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

    # load velodyne points and remove all behind image plane
    # each row of the velodyne data is forward, left, up, reflectance
    points = np,fromfile(filename, dtype=np.float32).reshape(-1,4)
    points = points[:, :3] # exclude luminance
    points[:, 3] = 1.0
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

def read_cam2cam(calib_path):
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    P_rect_02 = np.reshape(data['P_rect_02'], (3,4))
    P_rect_03 = np.reshape(data['P_rect_03'], (3,4))
    intrinsic_left = P_rect_02[:3, :3]
    intrinsic_right = P_rect_03[:3, :3]

    identity_l = np.eye(4)
    identity_r = np.eye(4)
    identity_l[:3, :3] = intrinsic_left
    identity_r[:3, :3] = intrinsic_right
    identity_l = identity_l.astype(np.float32)
    identity_r = identity_r.astype(np.float32)
    return identity_l, identity_r

def Point2Depth(velo2cam_path, cam2cam_path, point_path, cam=2, vel_depth=True):
    '''
    generate a depth map from velodyne data
    '''
    '''
    points_path = "./2011_09_26_drive_0017_sync/velodyne_points/data/0000000042.bin"
    
    GT depth information
    np.max ~80
    np.min ~0
    shape: [375, 1242]
    '''
    cam2cam = read_cam2cam(os.path.join(cam2cam_path))
    velo2cam = read_velo2cam(os.path.join(velo2cam_path))
    '''
    a = np.array((1,2,3))
    b = np.array((4,5,6))
    np.hstack((a,b))
    arrray([1,2,3,4,5,6])
    '''

    '''
    a = np.arange(6).reshape(2,3)
    print(a)
    [[0 1 2] [3 4 5]]
    a.shape
    (2,3)
    print(a[:, :, np.newaxis])
    [[[0] [1] [2]] [[3] [4] [5]]]
    print(a[:, :, np.newaxis].shape)
    (2,3,1)
    '''
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack(velo2cam, np.array([0,0,0,1.0]))
    
    # img_size (375, 1242)
    # list[<start>:<stop>:<step>]
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne -> image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    # cam = 2
    # numpy.dot(a,b, out=None) dot product of two arrays
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:,0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    
    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]
    # project to image

    # find the duplicate points and choose the closest depth 

def generate_depth_map(calib_dir, velo_filename, cam=1, vel_depth=False):
    '''
    generate a depth map from velodyne data
    '''

    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack(velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]) # velo2cam['T'] = (3,1)
    # numpy.newaxis is used to increase the dimension of the existing array by one more dimension
    velo2cam = np.vstack((velo2cam, np.array([0,0,0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32) # backwards


    # compute projection matrix velodyne -> image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    
    # load velodyne points and remove all behind image plane
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis] # 3x3 matrix #3x1 matrix

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]
    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts__im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]


    # find the duplicate points and choose the closest path
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == d)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    return depth
