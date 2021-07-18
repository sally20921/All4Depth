from __future__ division
import torch
import torch.nn.functional as F

pixel_coords = None
def euler2mat(angle): # [B, 3, 1]
    '''
    convert euler angles to rotation matrix
    Args:
        angle: rotation angle along 3 axis -- [B, 3]
    returns: 
        rotation matrix corresponding to euler angles -- [B, 3, 3]
    '''
    B = angles.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    # tensor.detach()
    # returns a new Tensor, detached from the current graph
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)
    
    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([ones, zeros, zeros, 
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat



def pose_vec2mat(vec, rotation_mode='euler'):
    '''
    convert 6DoF parameters to transformation matrix

    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz --[B,6]
    returns:
        a transformation matrix [B, 3, 4]
    '''
    translation = vec[:, :3].unsqueeze(-1) #[B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot) # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot) # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2) # [B, 3, 4]
    # torch.cat(tensors, dim=0, out=None) -> Tensor
    # concatenates the given sequence of tensors in the given dimension
    # all tensors must either have the same shape (except concatenating dimension) or empty
    return transform_mat

def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    '''
    inverse warp a source image to the target image plane

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]

    returns:
        projected_img: source image warped to the target image plane
        valid_points: boolean array indicating point validity
    '''

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse()) # [B, 3, H, W]

    pose_mat = pose_vec2mat(pose, rotation_mode) # [B, 3, 4]

    # get projection matrix for target camera frame to source pixel frame 
    proj_cam_to_src_pixel = intrinsics @ pose_mat # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr) # [B, H, W, 2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points

