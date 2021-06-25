import __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class KITTIDataset(MonoDataset):
    '''
    superclass for different types of dataset loaders
    '''
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        '''
        make sure your intrinsics matrix is *normalized* by the original 
        image size. 
        to normalizae you need to scale the first row by 1 / image_width 
        and the second row by 1 / image_height
        monodepth2 assumes a principal point to be exactly centered.
        '''
        self.K = np.array([[0.58, 0, 0.5, 0],
                            [0, 1.92, 0.5, 0],
                            [0, 0, 1, 0],
                            [0,0,0,1]],dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
                self.data_path,
                scene_name,
                "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    '''
    KITTI dataset which loads the original velodyne depth maps for ground truth
    '''
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
                self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

