import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def disp_to_depth(disp, min_depth, max_depth):
    '''
    convert network's sigmoid output into depth prediction
    " we convert the sigmoid output to depth with D = 1/a*sigmoid+b,
    where a and b are chosen to constrain D between 0.1 and 100 units."
    '''
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp+ (max_disp - min_disp) * disp # disp = sigmoid 
    depth = 1 / scaled_disp
    return scaled_disp, depth


class BackprojectDepth(nn.Module):
    '''
    layer to transform a depth image to a point cloud
    '''
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        
        # np.meshgrid: create a rectangular grid out of an array of x values and an array of y values 
        '''
        import numpy as np
        x = np.arange(2) # 0,1
        y = np.arange(3) # 0,1,2,
        ii, jj = np.meshgrid(x,y,indexing='ij')
        ii # array([[0,0,0],
                    [1,1,1]])
        jj # array([[0,1,2],
                    [0,1,2]])

        # indexing='ij' argument tells numpy to use matrix indexing instead
        of cartesian indexing
        '''
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.bath_size, 1, self.height * self.width), requires_grad=False)
        # torch.view(-1) flattens the tensor
        
