from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class ResNetMultiImageInput(models.ResNet):
    '''
    Construct a resnet model with varying numbers of input images.
    torchvision.models.resnet18(pretrained: bool=False, progress: bool = True, **kwargs: Any) # 18, 34, 50, 101
    return _resnet('resnet50', Bottleneck, [3,4,6,3], pretrained, progress, **kwargs)
    '''
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        # block is BasicBlock or BottleneckBlock 
        # layers is [2,2,2,2] or [3,4,6,3]
        # channel of first layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        '''
        block and layers are input parameters
        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', non_linearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    '''
    construct a resnet model.
    Args:
        num_layers (int): number of resnet layers. must be 18 or 50.
        pretrained (bool): if true, returns a model pretrained on imagenet
        num_input_images (int): number of frames stacked as input
    '''
    assert num_layers in [18, 50], "can only run with 18 or 60 layer resnet"
    blocks = {18: [2,2,2,2], 50:[3,4,6,3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1)
    

        

