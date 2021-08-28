import torch
import numpy as np

def euler2mat(angle):
    '''
    convert euler angle to ratation matrix
    
    "euler angle": any representation of 3 dimensional rotations

    parameters
    ____
    angle: [B,3]

    returns
    _____
    rot_mat: [B,3,3]
    '''
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z) # [B,1]
    sinz = torch.sin(z) # [B,1]

    zeros = z.detach() * 0 # [B,1]
    ones = zeros.detach() + 1 # [B,1]
    zmat = torch.stack([cosz, -sinz, zeros, 
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).view(B,3,3) # [B,9] => [B,3,3]

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros, 
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).view(B,3,3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack([cosy, zeros, siny, 
                        zeros, ones, zeros, 
                        -siny, zeros, cosy], dim=1).view(B,3,3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat # [B,3,3]

def pose_vec2mat(vec, mode='euler'):
    '''
    converts euler parameters to transformation matrix

    parameter
    ______
    vec: [B,6]

    returns
    _____
    mat: [B,3,4]
    '''
    if mode is None:
        return vec
    trans = vec[:, :3].unsqueeze(-1) # [B,3,1]
    rot = vec[:, 3:] # [B,3]
    
