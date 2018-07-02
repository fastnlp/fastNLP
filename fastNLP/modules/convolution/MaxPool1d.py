# python: 3.6
# encoding: utf-8

import torch.nn as nn
# import torch.nn.functional as F


class MaxPool1d(nn.Module):
    """1-d max-pooling module."""

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool1d, self).__init__()
        self.maxpool = nn.MaxPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode)

    def forward(self, x):
        return self.maxpool(x)
