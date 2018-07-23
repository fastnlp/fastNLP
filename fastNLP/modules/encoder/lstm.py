import torch.nn as nn


class Lstm(nn.Module):
    """
    LSTM module
    Args:
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers
    dropout : dropout rate
    bidirectional : If True, becomes a bidirectional RNN
    """

    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, bidirectional=False):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
