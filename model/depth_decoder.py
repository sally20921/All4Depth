from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def upsample(tensor):
    # torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, scale_factor=None)
    # down/up samples the input to either the given size or the given scale factor 
    return F.interpolate(tensor, scale_factor=2, mode='nearest')

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, inputs):
        out = self.pad(inputs)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.elu(out)
        return out 

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        '''
        num_ch_enc: np.array([64,64,128,256,512])
        scales: range(4) = [0,1,2,3]
        num_output_channels: (since its depth map) 1
        use_skips: (at training) True, (inference) False
        '''
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16,32,64,128,256])

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        
        # decoder 
        # range(start, stop, step)
        # 4,3,2,1,0
        for index in range(4, -1, -1):
            # upconv 0 
            if index == 4:
                num_ch_in = self.num_ch_enc[-1]
            else:
                num_ch_in = self.num_ch_dec[index+1]
            num_ch_out = self.num_ch_dec[index]
            self.convs[("upconv", index, 0)] = ConvBlock(num_ch_in, num_ch_out]

           


