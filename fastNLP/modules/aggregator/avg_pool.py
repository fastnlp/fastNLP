# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool(nn.Module):
    """1-d average pooling module."""

    def __init__(self, stride=None, padding=0):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # [N,C,L] -> [N,C]
        kernel_size = x.size(2)
        x = F.max_pool1d(
            input=x,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding)
        return x.squeeze(dim=-1)


class MeanPoolWithMask(nn.Module):
    def __init__(self):
        super(MeanPoolWithMask, self).__init__()
        self.inf = 10e12

    def forward(self, tensor, mask, dim=0):
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(tensor * masks, dim=dim) / torch.sum(masks, dim=1)

