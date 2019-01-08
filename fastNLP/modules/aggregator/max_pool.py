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


class MaxPoolWithMask(nn.Module):
    def __init__(self):
        super(MaxPoolWithMask, self).__init__()
        self.inf = 10e12

    def forward(self, tensor, mask, dim=0):
        masks = mask.view(mask.size(0), mask.size(1), -1)
        masks = masks.expand(-1, -1, tensor.size(2)).float()
        return torch.max(tensor + masks.le(0.5).float() * -self.inf, dim=dim)
