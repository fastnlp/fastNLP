# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool(nn.Module):
    """1-d max-pooling module."""

    def __init__(self, stride=None, padding=0, dilation=1):
        super(MaxPool, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        x = torch.transpose(x, 1, 2)  # [N,L,C] -> [N,C,L]
        kernel_size = x.size(2)
        x = F.max_pool1d(  # [N,L,C] -> [N,C,1]
            input=x,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation)
        return x.squeeze(dim=-1)  # [N,C,1] -> [N,C]
