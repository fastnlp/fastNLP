import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from fastNLP.modules.utils import initial_parameter

try:
    from torch import flip
except ImportError:
   def flip(x, dims):
        indices = [slice(None)] * x.dim()
        for dim in dims:
            indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

class VarRnnCellWrapper(nn.Module):
    """Wrapper for normal RNN Cells, make it support variational dropout
    """
    def __init__(self, cell, hidden_size, input_p, hidden_p):
        super(VarRnnCellWrapper, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.input_p = input_p
        self.hidden_p = hidden_p

    def forward(self, input, hidden, mask_x=None, mask_h=None):
        """
        :param input: [seq_len, batch_size, input_size]
        :param hidden: for LSTM, tuple of (h_0, c_0), [batch_size, hidden_size]
                       for other RNN, h_0, [batch_size, hidden_size]
        :param mask_x: [batch_size, input_size] dropout mask for input
        :param mask_h: [batch_size, hidden_size] dropout mask for hidden
        :return output: [seq_len, bacth_size, hidden_size]
                hidden: for LSTM, tuple of (h_n, c_n), [batch_size, hidden_size]
                        for other RNN, h_n, [batch_size, hidden_size]
        """
        is_lstm = isinstance(hidden, tuple)
        input = input * mask_x.unsqueeze(0) if mask_x is not None else input
        output_list = []
        for x in input:
            if is_lstm:
                hx, cx = hidden
                hidden = (hx * mask_h, cx) if mask_h is not None else (hx, cx)
            else:
                hidden *= mask_h if mask_h is not None else hidden
            hidden = self.cell(x, hidden)
            output_list.append(hidden[0] if is_lstm else hidden)
        output = torch.stack(output_list, dim=0)
        return output, hidden


class VarRNNBase(nn.Module):
    """Implementation of Variational Dropout RNN network.
    refer to `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (Yarin Gal and Zoubin Ghahramani, 2016)
    https://arxiv.org/abs/1512.05287`.
    """
    def __init__(self, mode, Cell, input_size, hidden_size, num_layers=1,
                 bias=True, batch_first=False,
                 input_dropout=0, hidden_dropout=0, bidirectional=False):
        super(VarRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self._all_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
                cell = Cell(input_size, self.hidden_size, bias)
                self._all_cells.append(VarRnnCellWrapper(cell, self.hidden_size, input_dropout, hidden_dropout))
        initial_parameter(self)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        is_lstm = (self.mode == "LSTM")
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            hx = input.new_zeros(self.num_layers * self.num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)
            if is_lstm:
                hx = (hx, hx)

        if self.batch_first:
            input = input.transpose(0, 1)
            batch_size = input.shape[1]

        mask_x = input.new_ones((batch_size, self.input_size))
        mask_out = input.new_ones((batch_size, self.hidden_size * self.num_directions))
        mask_h_ones = input.new_ones((batch_size, self.hidden_size))
        nn.functional.dropout(mask_x, p=self.input_dropout, training=self.training, inplace=True)
        nn.functional.dropout(mask_out, p=self.hidden_dropout, training=self.training, inplace=True)

        hidden_list = []
        for layer in range(self.num_layers):
            output_list = []
            mask_h = nn.functional.dropout(mask_h_ones, p=self.hidden_dropout, training=self.training, inplace=False)
            for direction in range(self.num_directions):
                input_x = input if direction == 0 else flip(input, [0])
                idx = self.num_directions * layer + direction
                cell = self._all_cells[idx]
                hi = (hx[0][idx], hx[1][idx]) if is_lstm else hx[idx]
                mask_xi = mask_x if layer == 0 else mask_out
                output_x, hidden_x = cell(input_x, hi, mask_xi, mask_h)
                output_list.append(output_x if direction == 0 else flip(output_x, [0]))
                hidden_list.append(hidden_x)
            input = torch.cat(output_list, dim=-1)

        output = input.transpose(0, 1) if self.batch_first else input
        if is_lstm:
            h_list, c_list = zip(*hidden_list)
            hn = torch.stack(h_list, dim=0)
            cn = torch.stack(c_list, dim=0)
            hidden = (hn, cn)
        else:
            hidden = torch.stack(hidden_list, dim=0)

        if is_packed:
            output = PackedSequence(output, batch_sizes)

        return output, hidden


class VarLSTM(VarRNNBase):
    """Variational Dropout LSTM.
    """
    def __init__(self, *args, **kwargs):
        super(VarLSTM, self).__init__(mode="LSTM", Cell=nn.LSTMCell, *args, **kwargs)

class VarRNN(VarRNNBase):
    """Variational Dropout RNN.
    """
    def __init__(self, *args, **kwargs):
        super(VarRNN, self).__init__(mode="RNN", Cell=nn.RNNCell, *args, **kwargs)

class VarGRU(VarRNNBase):
    """Variational Dropout GRU.
    """
    def __init__(self, *args, **kwargs):
        super(VarGRU, self).__init__(mode="GRU", Cell=nn.GRUCell, *args, **kwargs)
