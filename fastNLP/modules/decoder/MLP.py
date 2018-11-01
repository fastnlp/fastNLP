import torch
import torch.nn as nn
from fastNLP.modules.utils import initial_parameter


class MLP(nn.Module):
    def __init__(self, size_layer, activation='relu', initial_method=None):
        """Multilayer Perceptrons as a decoder

        :param size_layer: list of int, define the size of MLP layers.
        :param activation: str or function, the activation function for hidden layers.
        :param initial_method: str, the name of init method.

        .. note::
            There is no activation function applying on output layer.

        """
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None
        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i-1], size_layer[i])
            else:
                self.hiddens.append(nn.Linear(size_layer[i-1], size_layer[i]))

        actives = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        if activation in actives:
            self.hidden_active = actives[activation]
        elif isinstance(activation, callable):
            self.hidden_active = activation
        else:
            raise ValueError("should set activation correctly: {}".format(activation))
        initial_parameter(self, initial_method)

    def forward(self, x):
        for layer in self.hiddens:
            x = self.hidden_active(layer(x))
        x = self.output(x)
        return x


if __name__ == '__main__':
    net1 = MLP([5, 10, 5])
    net2 = MLP([5, 10, 5], 'tanh')
    for net in [net1, net2]:
        x = torch.randn(5, 5)
        y = net(x)
        print(x)
        print(y)
