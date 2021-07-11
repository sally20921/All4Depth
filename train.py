from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from options import MonodepthOptions

class Trainer:
    def __init__(self, options):
        self.opt = options



options = MonodepthOptions()
opts = options.parse()

if __name__=="__main__":
    trainer = Trainer(opts)
    trainer.train()
