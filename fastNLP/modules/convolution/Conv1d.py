# python: 3.6
# encoding: utf-8

import torch.nn as nn
# import torch.nn.functional as F


class Conv1d(nn.Module):
    """
    Basic 1-d convolution module.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, x):
        return self.conv(x)
