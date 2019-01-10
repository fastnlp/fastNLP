import torch.nn as nn

from fastNLP.modules.utils import initial_parameter


class Linear(nn.Module):
    """

    :param int input_size: input size
    :param int output_size: output size
    :param bool bias:
    :param str initial_method:
    """
    def __init__(self, input_size, output_size, bias=True, initial_method=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias)
        initial_parameter(self, initial_method)

    def forward(self, x):
        x = self.linear(x)
        return x
