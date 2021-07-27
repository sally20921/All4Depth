import torch
import numpy as np

def euler2mat(angle):
    '''
    convert euler angle to rotation matrix 

    we use the term "euler angle" for any representation of 3 dimensional rotations
    where we decompose the rotation into 3 separate angles

    '''
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z) #(B, 3) 
    sinz = torch.sin(z) #(B, 3)

    zeros = z.detach() * 0 #(B,3)
    ones = zeros.detach() + 1 #(B,3)
    zmat = torch.stack([cosz, -sinz, zeros, 
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).view(B, 3, 3) # (B, 9) => (B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat) 
    return rot_mat # (B, 3, 3)

def pose_vec2mat(vec, mode='euler'):
    '''
    convert euler parameters to transformation matrix
    euler parameters: (B, 4) 
    '''
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:] # (B, 3, 1) (B, 1)
    if mode == 'euler':
        rot_mat = euler2mat(rot) # (B, 3, 3)
    else:
        raise ValueError('Rotation mode not supported {}.'.format(mode))
    mat = torch.cat([rot_mat, trans], dim=2) # (B, 3, 4)
    return mat # (B, 3, 4)

def invert_pose(T):
    '''
    inverts a [B,4,4] torch.tensor pose
    '''
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1]) # (B, 4, 4)
    # Tensor.repeat(*sizes) -> Tensor
    # repeats this tensor along the specified dimensions
    # len(T) = B
    # A = [M b]
    #     [0 1]
    # inv(A) = [inv(M) -inv(M)*b]
    #          [ 0      1] 
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    # torch.transpose(input, dim0, dim1) 
    # returns a tensor that is transposed version of input given dimensions dim0 and dim1 are swappred 
    # torch.bmm(input, mat2, out=None)
    # input (bxnxm) mat2 (bxmxp) out (bxnxp)
    # (B, 3, 3) (B, 3, 1) => (B, 3, 1) => (B, 3)
    Tinv[:, :3, -1] = torch.bmm(-1*Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv

def inverse_pose_numpy(T):
    '''
    inverts a [4,4] np.array pose
    '''
    Tinv = np.copy(T) #(4,4)
    R, t = Tinv[:3, :3], Tinv[:3, 3] #(3,3) (3,1)
    Tinv[:3, :3], Tinv[:3, 3] = R.T, -np.matmul(R.T, t) # transpose
    # (3,3) (3,)
    return Tinv


