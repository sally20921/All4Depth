from __future__ import absolute_import, division, print_function
import torch.nn as nn
from .layers import ConvBlock, Conv3x3, upsample

class Decoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=3):

