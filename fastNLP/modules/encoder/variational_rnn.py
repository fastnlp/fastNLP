import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
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

    def forward(self, input_x, hidden, mask_x, mask_h, is_reversed=False):
        """
        :param PackedSequence input_x: [seq_len, batch_size, input_size]
        :param hidden: for LSTM, tuple of (h_0, c_0), [batch_size, hidden_size]
                       for other RNN, h_0, [batch_size, hidden_size]
        :param mask_x: [batch_size, input_size] dropout mask for input
        :param mask_h: [batch_size, hidden_size] dropout mask for hidden
        :return PackedSequence output: [seq_len, bacth_size, hidden_size]
                hidden: for LSTM, tuple of (h_n, c_n), [batch_size, hidden_size]
                        for other RNN, h_n, [batch_size, hidden_size]
        """
        def get_hi(hi, h0, size):
            h0_size = size - hi.size(0)
            if h0_size > 0:
                return torch.cat([hi, h0[:h0_size]], dim=0)
            return hi[:size]
        is_lstm = isinstance(hidden, tuple)
        input, batch_sizes = input_x
        output = []
        cell = self.cell
        if is_reversed:
            batch_iter = flip(batch_sizes, [0])
            idx = input.size(0)
        else:
            batch_iter = batch_sizes
            idx = 0

        if is_lstm:
            hn = (hidden[0].clone(), hidden[1].clone())
        else:
            hn = hidden.clone()
        hi = hidden
        for size in batch_iter:
            if is_reversed:
                input_i = input[idx-size: idx] * mask_x[:size]
                idx -= size
            else:
                input_i = input[idx: idx+size] * mask_x[:size]
                idx += size
            mask_hi = mask_h[:size]
            if is_lstm:
                hx, cx = hi
                hi = (get_hi(hx, hidden[0], size) * mask_hi, get_hi(cx, hidden[1], size))
                hi = cell(input_i, hi)
                hn[0][:size] = hi[0]
                hn[1][:size] = hi[1]
                output.append(hi[0])
            else:
                hi = get_hi(hi, hidden, size) * mask_hi
                hi = cell(input_i, hi)
                hn[:size] = hi
                output.append(hi)

        if is_reversed:
            output = list(reversed(output))
        output = torch.cat(output, dim=0)
        return PackedSequence(output, batch_sizes), hn


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
        self.is_lstm = (self.mode == "LSTM")

    def _forward_one(self, n_layer, n_direction, input, hx, mask_x, mask_h):
        is_lstm = self.is_lstm
        idx = self.num_directions * n_layer + n_direction
        cell = self._all_cells[idx]
        hi = (hx[0][idx], hx[1][idx]) if is_lstm else hx[idx]
        output_x, hidden_x = cell(input, hi, mask_x, mask_h, is_reversed=(n_direction == 1))
        return output_x, hidden_x

    def forward(self, input, hx=None):
        is_lstm = self.is_lstm
        is_packed = isinstance(input, PackedSequence)
        if not is_packed:
            seq_len = input.size(1) if self.batch_first else input.size(0)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            seq_lens = torch.LongTensor([seq_len for _ in range(max_batch_size)])
            input, batch_sizes = pack_padded_sequence(input, seq_lens, batch_first=self.batch_first)
        else:
            max_batch_size = int(input.batch_sizes[0])
            input, batch_sizes = input

        if hx is None:
            hx = input.new_zeros(self.num_layers * self.num_directions,
                                 max_batch_size, self.hidden_size, requires_grad=True)
            if is_lstm:
                hx = (hx, hx.new_zeros(hx.size(), requires_grad=True))

        mask_x = input.new_ones((max_batch_size, self.input_size))
        mask_out = input.new_ones((max_batch_size, self.hidden_size * self.num_directions))
        mask_h_ones = input.new_ones((max_batch_size, self.hidden_size))
        nn.functional.dropout(mask_x, p=self.input_dropout, training=self.training, inplace=True)
        nn.functional.dropout(mask_out, p=self.hidden_dropout, training=self.training, inplace=True)

        hidden = input.new_zeros((self.num_layers*self.num_directions, max_batch_size, self.hidden_size))
        if is_lstm:
            cellstate = input.new_zeros((self.num_layers*self.num_directions, max_batch_size, self.hidden_size))
        for layer in range(self.num_layers):
            output_list = []
            input_seq = PackedSequence(input, batch_sizes)
            mask_h = nn.functional.dropout(mask_h_ones, p=self.hidden_dropout, training=self.training, inplace=False)
            for direction in range(self.num_directions):
                output_x, hidden_x = self._forward_one(layer, direction, input_seq, hx,
                                                       mask_x if layer == 0 else mask_out, mask_h)
                output_list.append(output_x.data)
                idx = self.num_directions * layer + direction
                if is_lstm:
                    hidden[idx] = hidden_x[0]
                    cellstate[idx] = hidden_x[1]
                else:
                    hidden[idx] = hidden_x
            input = torch.cat(output_list, dim=-1)

        if is_lstm:
            hidden = (hidden, cellstate)

        if is_packed:
            output = PackedSequence(input, batch_sizes)
        else:
            input = PackedSequence(input, batch_sizes)
            output, _ = pad_packed_sequence(input, batch_first=self.batch_first)

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

# if __name__ == '__main__':
#     x = torch.Tensor([[1,2,3], [4,5,0], [6,0,0]])[:,:,None] * 0.1
#     mask = (x != 0).float().view(3, -1)
#     seq_lens = torch.LongTensor([3,2,1])
#     y = torch.Tensor([[0,1,1], [1,1,0], [0,0,0]])
#     # rev = _reverse_packed_sequence(pack)
#     # # print(rev)
#     lstm = VarLSTM(input_size=1, num_layers=2, hidden_size=2,
#                    batch_first=True, bidirectional=True,
#                    input_dropout=0.0, hidden_dropout=0.0,)
#     # lstm = nn.LSTM(input_size=1, num_layers=2, hidden_size=2,
#     #                batch_first=True, bidirectional=True,)
#     loss = nn.BCELoss()
#     m = nn.Sigmoid()
#     optim = torch.optim.SGD(lstm.parameters(), lr=1e-3)
#     for i in range(2000):
#         optim.zero_grad()
#         pack = pack_padded_sequence(x, seq_lens, batch_first=True)
#         out, hidden = lstm(pack)
#         out, lens = pad_packed_sequence(out, batch_first=True)
#         # print(lens)
#         # print(out)
#         # print(hidden[0])
#         # print(hidden[0].size())
#         # print(hidden[1])
#         out = out.sum(-1)
#         out = m(out) * mask
#         l = loss(out, y)
#         l.backward()
#         optim.step()
#         if i % 50 == 0:
#             print(out)
