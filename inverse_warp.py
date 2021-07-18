from __future__ division
import torch
import torch.nn.functional as F

def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    '''
    inverse warp a source image to the target image plane

    Args:
        
