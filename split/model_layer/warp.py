# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def disparity2depth(disparity, min_depth, max_depth):
    """
    disparity, min_depth, max_depth를 받아서 depth를 계산한다.
    disp = 1 / depth 관계이므로 depth가 크면 disp는 작고, depth가 작으면 disp는 크다.    
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    scaled_disp = min_disp + (max_disp - min_disp) * disparity
    depth       = 1 / scaled_disp
    return scaled_disp, depth


def vector2translation(translation_vector): # translation_vector = [tx, ty, tz]
    """
    translation_vecotr: [N, 1, 3]
    T = torch.zeros(translation_vector[0], 4, 4) ~ [N, 4, 4] 매트릭스 생성
    t = translation_vector.contiguous().view(-1, 3, 1) ~ contiguous는 메모리 절약을 위한 것이고, view는 모양을 바꿈
    따라서 (1, 1, 3) -> (1, 3, 1)
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device = translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)

    # 대각 성분은 모두 1로 할당
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1

    # N C H W이므로 0, 1, 2 행의 성분 중 마지막 3열에 해당하는 칼럼만 추출
    T[:, :3, 3, None] = t
    return T   


def angle2rotation(anlge_axis): # anlge_axis == (Ux, Uy, Uz)
    """
    Args:
        angle_axis: [B, 1, 3]
    return:
        rotion: [B, 4, 4]
        
    anlge_axis는 unit vector와 angle로 이루어진 representation이다.
    angle_axis의 norm을 구하면 angle을 분리
    anlge_axis를 anlge로 나누면 axis만 분리
    
    이렇게 (axis, angle)을 분리하였다면, 이를 이용해서 Redriguess rotation formula에 대입하여 회전 행렬을 계산할 수 있다.
    https://en.wikipedia.org/wiki/Rotation_matrix
    에서 Rotation matrix from axis and angle 참조, 수식의 근거는 Rodriguess rotation formula에 근거
    """
    angle = torch.linalg.norm(anlge_axis, ord = 2, dim = 2, keepdim = True) # angle axis에 대한 이해가 있다면 문제되지 않음
    axis  = anlge_axis / (angle + 1e-5)
    
    """
    입력 형태가 각도가 아닌 실수 형태
    cos(0)    = 1
    cos(3.14) = -1
    sin(0)    = 0
    sin(3.14) = 0
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    C   = 1 - cos

    x = axis[..., 0].unsqueeze(1) # print(axis[..., 0], x) [[C]] -> [[[C]]]
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)
    
    # 로드리게스 회전 행렬 계산하는 부분
    xsin = x * sin
    ysin = y * sin
    zsin = z * sin

    xC = x * C # x(1-cos)
    yC = y * C # y(1-cos)
    zC = z * C # z(1-cos)

    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rotation = torch.zeros((anlge_axis.shape[0], 4, 4)).to(device = anlge_axis.device)
    rotation[:, 0, 0] = torch.squeeze(x * xC + cos) # x * x(1-cos) + cos = x^2 + (1-x^2)cos
    rotation[:, 0, 1] = torch.squeeze(xyC - zsin)
    rotation[:, 0, 2] = torch.squeeze(zxC + ysin)
    rotation[:, 1, 0] = torch.squeeze(xyC + zsin)
    rotation[:, 1, 1] = torch.squeeze(y * yC + cos)
    rotation[:, 1, 2] = torch.squeeze(yzC - xsin)
    rotation[:, 2, 0] = torch.squeeze(zxC - ysin)
    rotation[:, 2, 1] = torch.squeeze(yzC + xsin)
    rotation[:, 2, 2] = torch.squeeze(z * zC + cos)
    rotation[:, 3, 3] = 1
    return rotation


def param2matrix(axisangle, translation, invert=False):
    """
    args:
        axisangle:   [N, 1, 3]
        translation: [N, 1, 3]
        이때 3개의 성분은 각각 (rx, ry, rz), (tx, ty, tz)
    
    return
        transformation_matrix: [B, 4, 4]
    
    invert = True이면 역행렬을 계산하는 것
    타겟 프레임보다 소스 프레임이 앞에 있어서 뷰를 뒤로 옮기는 것
    invert = False이면 역행렬을 구하지 않는 것
    타겟 프레임보다 소스 프레임이 뒤에 있어서 뷰를 앞으로 옮기는 것
    """
    R = angle2rotation(axisangle)
    t = translation.clone()
    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = vector2translation(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    return M



def upsample(tensor):
    return F.interpolate(tensor, scale_factor = 2, mode = "nearest")


class ConvBlock(nn.Module):
    def __init__ (self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.elu  = nn.ELU(inplace = True)
    
    def forward(self, inputs):
        """
        out = self.pad(inputs)
        out = self.conv(out)
        out = self.elu(out)
        """
        out = self.conv(inputs)
        out = self.elu(out)
        return out


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl = True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv    = nn.Conv2d(int(in_channels), int(out_channels), 3, 2)
    
    def forward(self, inputs):
        out = self.pad(inputs)
        out = self.conv(out)
        return out


class Depth2PointCloud(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Depth2PointCloud, self).__init__()
        self.batch_size = batch_size
        self.height     = height
        self.width      = width

        # 1. width x hegith의 메쉬그리드 정의, X 성분의 인덱스와 Y 성분의 인덱스를 별도로 생성하고
        #    meshigrid의 XY 인덱싱 구성을 axis = 0에서 concat하고 np.float32 타입으로 변경
        #    print [array([x index]), array([y index])] -> [[x index], [y index]]
        #    return: X = [0, 1],
        #                [0, 1],
        #            Y = [0, 0],
        #                [1, 1],
        meshgrid        = np.meshgrid(range(self.width), range(self.height), indexing = "xy")
        self.id_coords  = np.stack(meshgrid, axis = 0).astype(np.float32)
        self.id_coords  = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad = False)
        
        # 2. 이미지 픽셀 코디네이트를 생성 [[x index], [y index]]
        #    return: [[[0, 1], 
        #              [0, 1]],
        #             [[0, 0], 
        #              [1, 1]]]
        self.pix_coords = torch.unsqueeze(torch.stack([
                                            self.id_coords[0].view(-1), 
                                            self.id_coords[1].view(-1)], axis = 0),
                                            0)

        # 3. X 픽셀 좌표와 Y 픽셀 좌표를 batch size만큼 반복하고 옆으로 쭉 늘림
        #    return: [0, 1, 0, 1],
        #            [0, 0, 1, 1]
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)

        # 4. 호모지니어스 좌표계를 만들기 위해 torch.ones ~ batch 사이즈만큼 토치 파라미터 생성
        #    return: [1, 1, 1, 1]
        self.ones       = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad = False)

        # 5. (X, Y, _) 파라미터와 (_, _, 1) 파라미터를 torch.cat하고 required_grad = False로 호모지니어스 픽셀 좌표계 생성
        #    return: [0, 1, 0, 1],
        #            [0, 0, 1, 1],
        #            [1, 1, 1, 1]
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad = False)
    
    def forward(self, depth, inverse_intrinsic_matrix):
        camera_coords = torch.matmul(inverse_intrinsic_matrix[:, :3, :3], self.pix_coords)

        # depth * camera coords ~ [X, Y, 1]
        # return: camera coords ~ [zX, zY, z]
        camera_coords = depth.view(self.batch_size, 1, -1) * camera_coords

        # return: camera coords ~ [zX, zY, z, 1]
        camera_coords = torch.cat([camera_coords, self.ones], 1)
        return camera_coords


class PointCloud2Pixel(nn.Module):
    def __init__(self, batch_size, height, width, eps = 1e-7):
        super(PointCloud2Pixel, self).__init__()
        self.batch_size = batch_size
        self.height     = height
        self.width      = width
        self.eps        = eps
    
    def forward(self, camera_coords, intrinsic_matrix, transformation_matrix):
        projection    = torch.matmul(intrinsic_matrix, transformation_matrix)[:, :3, :]
        camera_coords = torch.matmul(projection, camera_coords)

        pixel_coords  = camera_coords[:, :2, :] / (camera_coords[:, 2, :].unsqueeze(1) + self.eps)
        pixel_coords  = pixel_coords.view(self.batch_size, 2, self.height, self.width)
        pixel_coords  = pixel_coords.permute(0, 2, 3, 1)
        pixel_coords[..., 0] /= self.width - 1
        pixel_coords[..., 1] /= self.height - 1
        pixel_coords  = (pixel_coords - 0.5) * 2
        return pixel_coords
