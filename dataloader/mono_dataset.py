from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(data.Dataset):
    '''
    superclass for monocular dataloaders
    '''
    def __init__(self,
                 data_path,
                 filenames, 
                 height, 
                 width, 
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height 
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs
        
        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # we need to specify augmentations differently in newer versions of torchvision. 
        # we first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 0.2)
            self.contrast = (0.8, 0.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        '''
        resize color images to the required scales and augment if required

        we create the color_aug object in advance and apply the same augmentation to all images in this item. This ensures that all images input to the pose network receive the same augmentation.
        '''
        # inputs[(n, im, i)]
        for k in list(inputs):
            # k = (n, im, i)
            frame = inputs[k]
            if "color" in k:
                n, im, i = k


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        '''
        returns a single training item from the dataset as a dictionary
        '''

