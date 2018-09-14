import torch
import torch.nn as nn
from fastNLP.modules.utils import initial_parameter
class MLP(nn.Module):
    def __init__(self, size_layer, num_class=2, activation='relu' , initial_method = None):
        """Multilayer Perceptrons as a decoder

        Args:
            size_layer: list of int, define the size of MLP layers
            num_class: int, num of class in output, should be 2 or the last layer's size
            activation: str or function, the activation function for hidden layers
        """
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None
        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i-1], size_layer[i])
            else:
                self.hiddens.append(nn.Linear(size_layer[i-1], size_layer[i]))

        if num_class == 2:
            self.out_active = nn.LogSigmoid()
        elif num_class == size_layer[-1]:
            self.out_active = nn.LogSoftmax(dim=1)
        else:
            raise ValueError("should set output num_class correctly: {}".format(num_class))
        
        actives = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }
        if activation in actives:
            self.hidden_active = actives[activation]
        elif isinstance(activation, callable):
            self.hidden_active = activation
        else:
            raise ValueError("should set activation correctly: {}".format(activation))
        initial_parameter(self, initial_method  )
    def forward(self, x):
        for layer in self.hiddens:
            x = self.hidden_active(layer(x))
        x = self.out_active(self.output(x))
        return x



if __name__ == '__main__':
    net1 = MLP([5,10,5])
    net2 = MLP([5,10,5], 5)
    for net in [net1, net2]:
        x = torch.randn(5, 5)
        y = net(x)
        print(x)
        print(y)
    