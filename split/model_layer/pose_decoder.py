# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        '''
        num_channel_encoder : 인코더 채널의 수
        num_input_features  : 입력 피처의 수
        num_frame_to_predict_for : 예측한 pose의 수
        '''
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation



class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        '''
        num_input_frames: 입력하는 이미지의 수
        nn.Conv2d(in_channels, out_channels, kerenl_size, stride, padding)
        '''
        super(PoseCNN, self).__init__()
        self.num_input_frames = num_input_frames
        
        self.convs    = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6*(num_input_frames - 1), 1)
        self.num_convs = len(self.convs)
        self.relu      = nn.ReLU(True)
        self.net       = nn.ModuleList(list(self.convs.values()))
    
    def forward(self, input_images):
        output = self.convs[0](input_images)
        for index in range(self.num_convs-1):
            output = self.convs[index+1](output)
            output = self.relu(output)
        
        output = self.pose_conv(output)
        output = output.mean(3).mean(2)
        output = 0.01 * output.view(-1, self.num_input_frames -1, 1, 6)
        
        axis_angle  = output[..., :3]
        translation = output[..., 3:]

        return axis_angle, translation
