# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def upsample(tensor):
    return F.interpolate(tensor, scale_factor = 2, mode = "nearest")


class ConvBlock(nn.Module):
    def __init__ (self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.elu  = nn.ELU(inplace = True)
    
    def forward(self, inputs):
        '''
        out = self.pad(inputs)
        out = self.conv(out)
        out = self.elu(out)
        '''
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
        self.conv    = nn.Conv2d(int(in_channels), int(out_channels), 3)
    
    def forward(self, inputs):
        out = self.pad(inputs)
        out = self.conv(out)
        return out


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales = range(4), num_output_channels = 1, use_skips = True):
        super(DepthDecoder, self).__init__()
        '''
        num_ch_enc: np.array([64, 64, 128, 256, 512])
        scales: range(4) = [0, 1, 2, 3]
        출력 채널의 수: 뎁스 맵이므로 1
        use_skips: 훈련 시에는 True이나, 추론 시에는 False
        '''
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.num_output_channels = num_output_channels
        self.use_skips           = use_skips
        self.upsample_mode       = 'nearest'
        self.scales              = scales

        # Decoder
        self.convs = OrderedDict()
        for index in range(4, -1, -1):
            # UpConv 0
            if index == 4:
                num_ch_in = self.num_ch_enc[-1]
            else:
                num_ch_in = self.num_ch_dec[index + 1]
            num_ch_out                   = self.num_ch_dec[index]
            self.convs[("upconv", index, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # UpConv 1
            num_ch_in = self.num_ch_dec[index]
            if self.use_skips and index > 0:
                num_ch_in += self.num_ch_enc[index - 1]
            num_ch_out                       = self.num_ch_dec[index]
            self.convs[("upconv", index, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_features):
        self.outputs = {}
        feature      = input_features[-1]

        for index in range(4, -1, -1):
            feature = self.convs[("upconv", index, 0)](feature)
            feature = [upsample(feature)]

            if self.use_skips and index > 0:
                feature += [input_features[index - 1]]

            feature = torch.cat(feature, 1)
            feature = self.convs[("upconv", index, 1)](feature)

            if index in self.scales:
                self.outputs[("disp", index)] = self.sigmoid(self.convs[("dispconv", index)](feature))

        return self.outputs
