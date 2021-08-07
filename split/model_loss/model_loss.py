from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional  as F
 


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        """
        SSIM Loss
        """
        self.intensity_x_pool = nn.AvgPool2d(3, 1)
        self.intensity_y_pool = nn.AvgPool2d(3, 1)

        self.standard_deviation_x_pool  = nn.AvgPool2d(3, 1)
        self.standard_deviation_y_pool  = nn.AvgPool2d(3, 1)
        self.standard_deviation_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def forward(self, image1, image2):
        image1  = self.refl(image1) # x
        image2  = self.refl(image2) # y

        mu_x     = self.intensity_x_pool(image1) # mu_x
        mu_y     = self.intensity_y_pool(image2) # mu_y

        sigma_x  = self.standard_deviation_x_pool(image1 ** 2) - mu_x ** 2
        sigma_y  = self.standard_deviation_y_pool(image2 ** 2) - mu_y ** 2
        sigma_xy = self.standard_deviation_xy_pool(image1 * image2) - mu_x * mu_y
        
        SSIMn = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIMd = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIMn / SSIMd) / 2, 0, 1)



class EdgeAwareSmoothLoss(nn.Module):
    def __init__(self):
        super(EdgeAwareSmoothLoss, self).__init__()
        """
        Edge-aware smooth loss
        disparity and image ~ N C H W

        example)
        test = torch.Tensor(range(1, 10))
        test = test.view(-1, 3, 3)
        print(test, test.shape)
        test = torch.unsqueeze(test, dim = 0)
        test = torch.unsqueeze(test, dim = 0)
        print(test, test.shape)
        # tensor([[[[1., 1., 1.],
        #         [2., 2., 2.],
        #         [3., 3., 3.]]]]) torch.Size([1, 1, 3, 3])
        print(test[:, :, :, :-1])
        # tensor([[[[1., 1.],
        #         [2., 2.],
        #         [3., 3.]]]])
        print(test[:, :, :, 1:])
        # tensor([[[[1., 1.],
        #         [2., 2.],
        #         [3., 3.]]]])
        print(test[:, :, 1:, :])
        # tensor([[[[2., 2., 2.],
        #         [3., 3., 3.]]]])
        print(test[:, :, :-1, :])
        # tensor([[[[1., 1., 1.],
        #         [2., 2., 2.]]]])
        """
    def forward(self, disparity, image):
        gradient_disp_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
        gradient_disp_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, ])

        gradient_imag_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim = True)
        gradient_imag_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim = True)
        # print(gradient_disp_x.shape, gradient_imag_x.shape)
        gradient_disp_x = gradient_disp_x * torch.exp(-gradient_imag_x)
        gradient_disp_y = gradient_disp_y * torch.exp(-gradient_imag_y)

        smooth_loss     = gradient_disp_x.mean() + gradient_disp_y.mean()
        return smooth_loss



class ReprojectionLoss(nn.Module):
    def __init__(self):
        super(ReprojectionLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, prediction, target):
        absolute_difference = torch.abs(target - prediction)
        L1_loss             = absolute_difference.mean(1, True)
        ssim_loss           = self.ssim(prediction, target).mean(1, True)

        reprojection_loss   = 0.85 * ssim_loss + 0.15 * L1_loss
        return reprojection_loss



class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.edge_aware_smooth = EdgeAwareSmoothLoss()

    def forward(self, disp, color):
        mean_disp           = disp.mean(2, True).mean(3, True)
        norm_disp           = disp / (mean_disp + 1e-5)

        edge_smooth_loss    = self.edge_aware_smooth(norm_disp, color)
        return edge_smooth_loss