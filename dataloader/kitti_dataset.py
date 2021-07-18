from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class KITTIDataset(MonoDataset):
    '''
    superclass for different types of KITTI dataloaders
    '''
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width
        # and the second row by 1 / image_height. 
        # MonoDepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal flip augmentation

        self.K = np.array([[0.58, 0, 0.5, 0],
                            [0, 1.92, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
       # 2011_09_25/2011_09_26_drive_0002_sync/velodyne_points/data/0000000080.bin
       line = self.filenames[0].split()
       scene_name = line[0] # 2011_02_26_drive_0002_sync
       frame_index = int(line[1]) # 0000000080.bin

       # self.filenames = 2011_09_26/2011_09_26_drive_0002_sync 0000000069 l
        velo_filename = os.path.join(self.data_path, scene_name, "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        
        return os.path.isfile(velo_filename)



