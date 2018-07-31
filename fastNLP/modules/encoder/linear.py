import torch.nn as nn


class Linear(nn.Module):
    """
    Linear module
    Args:
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers
    dropout : dropout rate
    bidirectional : If True, becomes a bidirectional RNN
    """

    def __init__(self, input_size, output_size, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias)

    def forward(self, x):
        x = self.linear(x)
        return x
