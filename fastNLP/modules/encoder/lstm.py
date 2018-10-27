import torch.nn as nn

from fastNLP.modules.utils import initial_parameter


class LSTM(nn.Module):
    """Long Short Term Memory

    Args:
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.
    """

    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, batch_first=True,
                 bidirectional=False, bias=True, initial_method=None, get_hidden=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.get_hidden = get_hidden
        initial_parameter(self, initial_method)

    def forward(self, x, h0=None, c0=None):
        if h0 is not None and c0 is not None:
            x, (ht, ct) = self.lstm(x, (h0, c0))
        else:
            x, (ht, ct) = self.lstm(x)
        if self.get_hidden:
            return x, (ht, ct)
        else:
            return x


if __name__ == "__main__":
    lstm = LSTM(10)
