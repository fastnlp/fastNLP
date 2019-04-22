import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

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
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.get_hidden = get_hidden
        initial_parameter(self, initial_method)

    def forward(self, x, seq_lens=None, h0=None, c0=None):
        if h0 is not None and c0 is not None:
            hx = (h0, c0)
        else:
            hx = None
        if seq_lens is not None and not isinstance(x, rnn.PackedSequence):
            print('padding')
            sort_lens, sort_idx = torch.sort(seq_lens, dim=0, descending=True)
            if self.batch_first:
                x = x[sort_idx]
            else:
                x = x[:, sort_idx]
            x = rnn.pack_padded_sequence(x, sort_lens, batch_first=self.batch_first)
            output, hx = self.lstm(x, hx)  # -> [N,L,C]
            output, _ = rnn.pad_packed_sequence(output, batch_first=self.batch_first)
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            if self.batch_first:
                output = output[unsort_idx]
            else:
                output = output[:, unsort_idx]
        else:
            output, hx = self.lstm(x, hx)
        if self.get_hidden:
            return output, hx
        return output


if __name__ == "__main__":
    lstm = LSTM(input_size=2, hidden_size=2, get_hidden=False)
    x = torch.randn((3, 5, 2))
    seq_lens = torch.tensor([5,1,2])
    y = lstm(x, seq_lens)
    print(x)
    print(y)
    print(x.size(), y.size(), )
