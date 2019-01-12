import torch.nn as nn

from fastNLP.modules.utils import initial_parameter


class LSTM(nn.Module):
    """Long Short Term Memory

    :param int input_size:
    :param int hidden_size:
    :param int num_layers:
    :param float dropout:
    :param bool batch_first:
    :param bool bidirectional:
    :param bool bias:
    :param str initial_method:
    :param bool get_hidden:
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
