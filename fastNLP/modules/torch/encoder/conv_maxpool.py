__all__ = [
    "ConvMaxpool"
]
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMaxpool(nn.Module):
    r"""
    集合了 **Convolution** 和 **Max-Pooling** 于一体的层。给定一个 ``[batch_size, max_len, input_size]`` 的输入，返回
    ``[batch_size, sum(output_channels)]`` 大小的 matrix。在内部，是先使用 ``CNN`` 给输入做卷积，然后经过 activation 
    激活层，在通过在长度(max_len)这一维进行 ``max_pooling`` 。最后得到每个 sample 的一个向量表示。

    :param in_channels: 输入 channel 的大小，一般是 embedding 的维度，或 ``encoder``的 output 维度
    :param out_channels: 输出 channel 的数量。如果为 :class:`list`，则需要与 ``kernel_sizes`` 的数量保持一致
    :param kernel_sizes: 输出 channel 的 kernel 大小。
    :param activation: **卷积** 后的结果将通过该 ``activation`` 后再经过 ``max-pooling``。支持 ``['relu', 'sigmoid', 'tanh']``。
    """

    def __init__(self, in_channels: int, out_channels: Union[int, List[int]], kernel_sizes: Union[int, List[int]], activation: str="relu"):
        super(ConvMaxpool, self).__init__()

        for kernel_size in kernel_sizes:
            assert kernel_size % 2 == 1, "kernel size has to be odd numbers."

        # convolution
        if isinstance(kernel_sizes, (list, tuple, int)):
            if isinstance(kernel_sizes, int) and isinstance(out_channels, int):
                out_channels = [out_channels]
                kernel_sizes = [kernel_sizes]
            elif isinstance(kernel_sizes, (tuple, list)) and isinstance(out_channels, (tuple, list)):
                assert len(out_channels) == len(
                    kernel_sizes), "The number of out_channels should be equal to the number" \
                                   " of kernel_sizes."
            else:
                raise ValueError("The type of out_channels and kernel_sizes should be the same.")

            self.convs = nn.ModuleList([nn.Conv1d(
                in_channels=in_channels,
                out_channels=oc,
                kernel_size=ks,
                stride=1,
                padding=ks // 2,
                dilation=1,
                groups=1,
                bias=False)
                for oc, ks in zip(out_channels, kernel_sizes)])

        else:
            raise Exception(
                'Incorrect kernel sizes: should be list, tuple or int')

        # activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh
        else:
            raise Exception(
                "Undefined activation function: choose from: relu, tanh, sigmoid")

    def forward(self, x: torch.FloatTensor, mask=None):
        r"""

        :param x: ``[batch_size, max_len, input_size]``，一般是经过 ``embedding`` 后的值
        :param mask: ``[batch_size, max_len]``，**0** 的位置表示 padding，不影响卷积运算，``max-pooling`` 一定不会 pool 到 padding 为 0 的位置
        :return:
        """
        # [N,L,C] -> [N,C,L]
        x = torch.transpose(x, 1, 2)
        # convolution
        xs = [self.activation(conv(x)) for conv in self.convs]  # [[N,C,L], ...]
        if mask is not None:
            mask = mask.unsqueeze(1)  # B x 1 x L
            xs = [x.masked_fill(mask.eq(False), float('-inf')) for x in xs]
        # max-pooling
        xs = [F.max_pool1d(input=i, kernel_size=i.size(2)).squeeze(2)
              for i in xs]  # [[N, C], ...]
        return torch.cat(xs, dim=-1)  # [N, C]
