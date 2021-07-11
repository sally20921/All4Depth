from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class ResnetMultiImageInput(models.ResNet):
    '''
    constructs a resnet model with varying number of input images.
    Args
        block: models.resnet.BasicBlock, models.resnet.Bottleneck
        layers: [2,2,2,2], [3,4,6,3]

    '''
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResnetMultiImageInput, self).__init__(block, layers)

        self.inplanes = 64
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = nn.Conv2d(num_input_images * 3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
            # isinstance(object, classinfo): checks if the object argument is an instance or subclass of classinfo class argument
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)

def resnet_multiimage_model(num_layers, pretrained=False, num_input_images=1):
    '''
    construct a resnet model.
    num_layers must be 18 or 50
    pretrained returns a model pretrained on ImageNet
    num_input_images (int): number of frames stacked as input
    '''
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    layers = {18: [2,2,2,2], 50: [3,4,6,3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResnetMultiImageInput(block_type, layers, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images

    return model

class ResnetEncoder(nn.Module):
    '''
    pytorch module for a resnet encoder
    num_layers: 18, 34, 50, 101, 152 
    pretrained: pretrained ImageNet model
    '''
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of layers.".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_model(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        #x = self.features[-1]
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        self.features.append(x)
       # x = self.features[-1]
        x = self.encoder.layer2(x)
        self.features.append(x)
        x = self.encoder.layer3(x)
        self.features.append(x)
        x = self.encoder.layer4(x)
        self.features.append(x)

        return self.features

