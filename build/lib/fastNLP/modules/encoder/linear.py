import torch.nn as nn

from fastNLP.modules.utils import initial_parameter
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

    def __init__(self, input_size, output_size, bias=True,initial_method = None        ):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias)
        initial_parameter(self, initial_method)
    def forward(self, x):
        x = self.linear(x)
        return x
