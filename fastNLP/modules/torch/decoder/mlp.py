__all__ = [
    "MLP"
]

from typing import List, Callable, Union
import torch
import torch.nn as nn


class MLP(nn.Module):
    r"""
    多层感知器。
        
    Examples::

        >>> net1 = MLP([5, 10, 5])
        >>> net2 = MLP([5, 10, 5], 'tanh')
        >>> net3 = MLP([5, 6, 7, 8, 5], 'tanh')
        >>> net4 = MLP([5, 6, 7, 8, 5], 'relu', output_activation='tanh')
        >>> net5 = MLP([5, 6, 7, 8, 5], ['tanh', 'relu', 'tanh'], 'tanh')
        >>> for net in [net1, net2, net3, net4, net5]:
        >>>     x = torch.randn(5, 5)
        >>>     y = net(x)
        >>>     print(x)
        >>>     print(y)

    :param size_layer: 一个 int 的列表，用来定义 :class:`MLP` 的层数，列表中的数字为每一层是 hidden 数目。 :class:`MLP` 的层数为 ``len(size_layer) - 1``
    :param activation: 隐藏层的激活函数，可以支持多种类型：
    
            - 一个 :class:`str` 或函数 -- 所有隐藏层的激活函数都为 ``activation`` 代表的函数；
            - :class:`str` 或函数的列表 -- 每一个隐藏层的激活函数都为列表中对应的函数，其中列表长度为隐藏层数；

        对于字符串类型的输入，支持 ``['relu', 'tanh', 'sigmoid']`` 三种激活函数。
    :param output_activation: 输出层的激活函数。默认值为 ``None``，表示输出层没有激活函数
    :param dropout: dropout 概率
    """

    def __init__(self, size_layer: List[int], activation: Union[str, Callable, List[str]]='relu',
                output_activation: Union[str, Callable]=None, dropout: float=0.0):
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None
        self.output_activation = output_activation
        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i - 1], size_layer[i])
            else:
                self.hiddens.append(nn.Linear(size_layer[i - 1], size_layer[i]))

        self.dropout = nn.Dropout(p=dropout)

        actives = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        if not isinstance(activation, list):
            activation = [activation] * (len(size_layer) - 2)
        elif len(activation) == len(size_layer) - 2:
            pass
        else:
            raise ValueError(
                f"the length of activation function list except {len(size_layer) - 2} but got {len(activation)}!")
        self.hidden_active = []
        for func in activation:
            if callable(func):
                self.hidden_active.append(func)
            elif func.lower() in actives:
                self.hidden_active.append(actives[func])
            else:
                raise ValueError("should set activation correctly: {}".format(activation))
        if self.output_activation is not None:
            if callable(self.output_activation):
                pass
            elif self.output_activation.lower() in actives:
                self.output_activation = actives[self.output_activation]
            else:
                raise ValueError("should set activation correctly: {}".format(activation))

    def forward(self, x):
        r"""
        :param x:
        :return:
        """
        for layer, func in zip(self.hiddens, self.hidden_active):
            x = self.dropout(func(layer(x)))
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = self.dropout(x)
        return x
