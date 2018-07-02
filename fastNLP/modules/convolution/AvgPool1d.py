# python: 3.6
# encoding: utf-8

import torch.nn as nn
# import torch.nn.functional as F


class AvgPool1d(nn.Module):
    """1-d average pooling module."""

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True):
        super(AvgPool1d, self).__init__()
        self.pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad)

    def forward(self, x):
        return self.pool(x)
