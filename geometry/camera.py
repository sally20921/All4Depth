from functools import lru_cache
import torch
import torch.nn as nn

from geometry.pose import Pose
from geometry.camera_utils import scale_intrinsics
from utils.image import image_grid

# property() is a built-in function that creates and returns a property object 
# property(fget=None, fset=None, fdel=None, doc=None)

class Camera(nn.Module):
    '''
    differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    '''
    def __init__(self, K, Tcw=None):
        '''
        initializes the Camera class

        Arguments:
            K: torch.Tensor [3,3] # I think they assume [B,3,3]
                Camera intrinsics
            Tcw: Pose
                Camera -> World pose transformation # also [B,3,3]
        '''
        super().__init__()
        self.K = K
        self.Tcw = Pose.Identity(len(K)) if Tcw is None else Tcw

    def __len__(self):
        '''
        batch size of camera intrinsics
        '''
        return len(self.K)

    def to(self, *args, **kwargs):
        '''
        moves object to a specific device
        '''
        self.K = self.K.to(*args, **kwargs)
        self.Twc = self.Twc.to(*args, **kwargs)
        return self

    @property
    def fx(self):
        '''
        focal length in x
        '''
        return self.K[:, 0, 0]
    
        
    @property
    def fy(self):
        '''
        focal length in y
        '''
        return self.K[:, 1, 1]

    @property
    def cx(self):
        '''
        principal point in x
        '''
        return self.K[:, 0, 2]
    
    @property
    def cy(self):
        '''
        principal point in y
        '''
        return self.K[:, 1, 2]

    @property
    @lru_cache()
    def Twc(self):
        '''
        World => Camera pose transformation (inverse of Tcw)
        '''
        return self.Tcw.inverse()

    @property
    @lru_cache()
    def Kinv(self):
        '''
        inverse intrinsics 
        (for lifting)
        '''
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1 * self.cx / self.fx
        Kinv[:, 1, 2] = -1 * self.cy / self.fy
        return Kinv

    def scaled(self, x_scale, y_scale=None):
        '''
        returns a scaled version of the camera
        (chaning intrinsics)

        Argument:
            x_scale: float
                resize scale in x
            y_scale: float
                resize scale in y. If None, use the same as x_scale

        Returns:
            camera: Camera
                scaled version of the current camera
        '''
        if y_scale is None:
            y_scale = x_scale
        # if no scaling is necessary, return the same camera
        if x_scale == 1. and y_scale == 1.:
            return self
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, Tcw=self.Tcw)

    def reconstruct(self, depth, frame='w'):
        '''
        reconstructs pixel-wise 3D points from a depth map.

        Arguments:
            depth: torch.Tensor [B,1,H,W]
                depth map for the camera
            frame: 'w'
                reference frame: 'c' for camera and 'w' for world

        Returns:
            points: torch.Tensor [B,3,H,W]
                pixel-wise 3D points
        '''
        B, C, H, W = depth.shape
        assert C == 1

        # create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False) #[B,3,H,W]
        flat_grid = grid.view(B, 3, -1) # [B, 3, H*W]

        # estimate the outward rays in the camera frame
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
        # scale rays to metric depth
        Xc = xnorm * depth
        
        # if in the camera frame of reference
        if frame == 'c':
            return Xc
        # if in world frame of reference
        elif frame == 'w':
            return self.Twc @ Xc
        else:
            raise ValueError("Unknown reference frame {}".format(frame))

    def project(self, X, frame='w'):
        '''
        projects 3D points onto the image plane

        Arguments:
            X: torch.Tensor [B,3,H,W]   
                3D points to be projected 
            frame: 'w'
                reference frame: 'c' for camera and 'w' for world

        Returns:
            points: torch.Tensor [B, H, W, 2]
                2D projected points that are within image boundaries 
        '''
        B, C, H, W = X.shape
        assert C == 3

        # project 3D points onto the camera image plane
        if frame == 'c':
            Xc = self.K.bmm(X.view(B, 3, -1)) #(B,3,3) x (B, 3, H*W) => (B,3, H*W)
        elif frame == 'w':
            Xc = self.K.bmm((self.Twc @ X).view(B, 3, -1))
        else:
            raise ValueError("Unknown reference frame {}".format(frame))

        # normalize points
        X = Xc[:, 0] # (B, H*W)
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)
        # torch.clamp(input, min, max, *, out=None)
        # output is either min, input or max
        Xnorm = 2 * (X / Z) / (W-1) -1. # element-wise division
        Ynorm = 2 * (Y / Z) / (H-1) -1.
        #(B, H*W, 2) -> (B, H, W, 2)
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B,H,W,2)


        


    
