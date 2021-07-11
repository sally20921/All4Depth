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


