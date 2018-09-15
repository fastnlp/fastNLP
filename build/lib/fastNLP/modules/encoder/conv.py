# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
# import torch.nn.functional as F

from fastNLP.modules.utils import initial_parameter

class Conv(nn.Module):
    """
    Basic 1-d convolution module.
    initialize with xavier_uniform
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True, activation='relu',initial_method = None ):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # xavier_uniform_(self.conv.weight)

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()}
        if activation in activations:
            self.activation = activations[activation]
        else:
            raise Exception(
                'Should choose activation function from: ' +
                ', '.join([x for x in activations]))
        initial_parameter(self, initial_method)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)  # [N,L,C] -> [N,C,L]
        x = self.conv(x)  # [N,C_in,L] -> [N,C_out,L]
        x = self.activation(x)
        x = torch.transpose(x, 1, 2)  # [N,C,L] -> [N,L,C]
        return x
