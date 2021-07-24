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

def transformation_from_parameters(axisangle, translation, invert=False):
    '''
    convert (axisangle, translation) output into a 4x4 matrix
    '''
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1,2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R,T)
    else:
        M = torch.matmul(T,R)

    return M

def get_translation_matrix(translation_vector):
    '''
    convert a translation vector into a 4x4 transformation matrix
    B-by-3-by-1
    1. make your (X,Y,Z) vector into a homogeneous vector (X,Y,Z,1) 
    2. [x,y,z,1] * [[1 0 0 0] = [x+a, y+b, z+c, 1]
                   [0 1 0 0]
                   [0 0 1 0]
                   [a b c 1]]
    '''
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1) # B-by-3-by-1
    # 4-by-4-by-1
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:. 3, 3] = 1
    # T[:, :3, 3] => B-by-3
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    '''
    convert an axisangle rotation axang = [0 1 0 pi/2] 
    into a 4x4 transformation matrix.
    input 'vec' has to be B-by-1-by-3
    c = cos(angle)
    s = sin(angle)
    t = 1 - c
    x = normalized axis x coordinate
    y = normalized axis y coordinate
    z = normalized axis z coordinate
    R = [[t*x*x+c t*x*y-z*s t*x*z+y*s]
        [t*x*y+z*s t*y*y+c t*y*z-x*s]
        [t*x*z-y*s t*y*z+x*s t*z*z+c]]
    '''
    # torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)
    # returns the matrix norm or vector norm of a given tensor
    # If 'keepdim' is 'True', the output tensor is of the same size as 'input' except in the dimension 'dim' where it is of size 1. 
    # otherwise, 'dim' is squeezed, resulting in the output tensor having 1 fewer dimension than 'input'
    # Frobenius norm produces the same result as p=2 in all cases except dim is a list of three or more dims, in which case Frobenius norm throws an error

    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    
    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)


    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot



    '''convert a translation vector into a 4x4 transformation matrix
    ex. 
    trvec = [0.5, 6, 100]
    tform = get_translation_matrix(trvec)
    t.form.shape # 4x4
    tform # [[1.000 0 0 0.5]
            [0 1.000 0 6]
            [0 0 1.000 100]
            [0 0 0 1.000]]

    trvec- Cartesian representation of a translation vector
    n-by-3 matrix containing n translation vectors t=[x y z]
    tform- homogeneous transformation 
    4-by-2-by-n matrix of n homogeneous transformations 
    '''
    # T[:, :3, 3, None=np.newaxis] = t


class ConvBlock(nn.Module):
    '''
    layer to perform a convolution followed by ELU
    '''
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out 

class Conv3x3(nn.Module):
    '''
    layer to pad and convolve input
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

    input size (N, C_in, H, W) 
    output (N, C_out, H_out, W_out)
    out(N_i, C_out_j) = bias(C_out_j) + \sum_{k=0}^{C_in -1} weight(C_out_j, k) * input(N_i, k) 
    * 2d cross-correlation operator
    N is batch size
    C number of channels
    H height of input planes in pixels
    W widht in pixels 

    m = nn.Conv2d(16, 33, 3, stride=2) # with square kernels and equal stride
    input = torch.randn(20, 16, 50, 100)
    output = m.output # shape [20, 33, 24, 49]
    '''
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out 


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

class Project3D(nn.Module):
    '''
    layer which projects 3D points into a camera with intrinsics K and at position T
    '''
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

       
