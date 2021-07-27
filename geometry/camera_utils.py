import torch
import torch.nn.functional as funct

def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    '''
    construct a [3,3] camera intrinsics from pinhole parameters
    
    the intrinsic matrix (commonly represented in equation as K) 
    fx, fy are the pixel focal length and are identical for square pixels
    cx, cy are the offsets of the principla point from the top-left-corner of the image frame
    '''
    return torch.tensor([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=1, device=device)

def scale_intrinsics(K, x_scale, y_scale):
    '''
    scale intrinsics given x_scale and y_scale factors
    '''
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5


def view_synthesis(ref_image, depth, ref_cam, cam, mode='bilinear', padding_mode='zeros'):
    '''
    synthesize an image from another plus a depth map.

    Arguments:
        ref_image: torch.Tensor [B, 3, H, W]
            reference image to be warped
        depth: torch.Tensor [B, 1, H, W]
            depth map from the original image
        ref_cam: camera
            camera class for the reference image
        cam: camera
            camera class for the original image
        mode: str
            interpolation mode
        padding_mode:
            padding mode for interpolation

    Returns:
        ref_warped: torch.Tensor [B, 3, H, W]
            warped reference image in the original frame of reference 
    '''
    assert depth.size(1) == 1

