# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMaxpool(nn.Module):
    """
    Convolution and max-pooling module with multiple kernel sizes.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True, activation='relu'):
        super(ConvMaxpool, self).__init__()

        # convolution
        if isinstance(kernel_sizes, (list, tuple, int)):
            if isinstance(kernel_sizes, int):
                out_channels = [out_channels]
                kernel_sizes = [kernel_sizes]
            self.convs = nn.ModuleList([nn.Conv1d(
                in_channels=in_channels,
                out_channels=oc,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias)
                for oc, ks in zip(out_channels, kernel_sizes)])
        else:
            raise Exception(
                'Incorrect kernel sizes: should be list, tuple or int')

        # activation function
        if activation == 'relu':
            self.activation = F.relu
        else:
            raise Exception(
                "Undefined activation function: choose from: relu")

    def forward(self, x):
        # [N,L,C] -> [N,C,L]
        x = torch.transpose(x, 1, 2)
        # convolution
        xs = [self.activation(conv(x)) for conv in self.convs]  # [[N,C,L]]
        # max-pooling
        xs = [F.max_pool1d(input=i, kernel_size=i.size(2)).squeeze(2)
              for i in xs]  # [[N, C]]
        return torch.cat(xs, dim=-1)  # [N,C]
