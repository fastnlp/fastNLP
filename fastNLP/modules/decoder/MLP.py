import torch
import torch.nn as nn

from fastNLP.modules.utils import initial_parameter


class MLP(nn.Module):
    """Multilayer Perceptrons as a decoder

    :param list size_layer: list of int, define the size of MLP layers.
    :param str activation: str or function, the activation function for hidden layers.
    :param str initial_method: the name of initialization method.
    :param float dropout: the probability of dropout.

    .. note::
        There is no activation function applying on output layer.

    """

    def __init__(self, size_layer, activation='relu', initial_method=None, dropout=0.0):
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None
        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i-1], size_layer[i])
            else:
                self.hiddens.append(nn.Linear(size_layer[i-1], size_layer[i]))

        self.dropout = nn.Dropout(p=dropout)

        actives = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        if activation in actives:
            self.hidden_active = actives[activation]
        elif callable(activation):
            self.hidden_active = activation
        else:
            raise ValueError("should set activation correctly: {}".format(activation))
        initial_parameter(self, initial_method)

    def forward(self, x):
        for layer in self.hiddens:
            x = self.dropout(self.hidden_active(layer(x)))
        x = self.dropout(self.output(x))
        return x


if __name__ == '__main__':
    net1 = MLP([5, 10, 5])
    net2 = MLP([5, 10, 5], 'tanh')
    for net in [net1, net2]:
        x = torch.randn(5, 5)
        y = net(x)
        print(x)
        print(y)
