# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn
# import torch.nn.functional as F


class KMaxPool(nn.Module):
    """K max-pooling module."""

    def __init__(self, k):
        super(KMaxPool, self).__init__()
        self.k = k

    def forward(self, x):
        # [N,C,L] -> [N,C*k]
        x, index = torch.topk(x, self.k, dim=-1, sorted=False)
        x = torch.reshape(x, (x.size(0), -1))
        return x
