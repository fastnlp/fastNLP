r"""undocumented
Variational RNN 及相关模型的 fastNLP实现，相关论文参考：
`A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (Yarin Gal and Zoubin Ghahramani, 2016) <https://arxiv.org/abs/1512.05287>`_
"""

__all__ = [
    "VarRNN",
    "VarLSTM",
    "VarGRU"
]

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

try:
    from torch import flip
except ImportError:
    def flip(x, dims):
        indices = [slice(None)] * x.dim()
        for dim in dims:
            indices[dim] = torch.arange(
                x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

from ..utils import initial_parameter


class VarRnnCellWrapper(nn.Module):
    r"""
    Wrapper for normal RNN Cells, make it support variational dropout
    """

    def __init__(self, cell, hidden_size, input_p, hidden_p):
        super(VarRnnCellWrapper, self).__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.input_p = input_p
        self.hidden_p = hidden_p

    def forward(self, input_x, hidden, mask_x, mask_h, is_reversed=False):
        r"""
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
        input, batch_sizes = input_x.data, input_x.batch_sizes
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
                input_i = input[idx - size: idx] * mask_x[:size]
                idx -= size
            else:
                input_i = input[idx: idx + size] * mask_x[:size]
                idx += size
            mask_hi = mask_h[:size]
            if is_lstm:
                hx, cx = hi
                hi = (get_hi(hx, hidden[0], size) *
                      mask_hi, get_hi(cx, hidden[1], size))
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
    r"""
    Variational Dropout RNN 实现.

    论文参考: `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (Yarin Gal and Zoubin Ghahramani, 2016)
    https://arxiv.org/abs/1512.05287`.

    """

    def __init__(self, mode, Cell, input_size, hidden_size, num_layers=1,
                 bias=True, batch_first=False,
                 input_dropout=0, hidden_dropout=0, bidirectional=False):
        r"""
        
        :param mode: rnn 模式, (lstm or not)
        :param Cell: rnn cell 类型, (lstm, gru, etc)
        :param input_size:  输入 `x` 的特征维度
        :param hidden_size: 隐状态 `h` 的特征维度
        :param num_layers: rnn的层数. Default: 1
        :param bias: 如果为 ``False``, 模型将不会使用bias. Default: ``True``
        :param batch_first: 若为 ``True``, 输入和输出 ``Tensor`` 形状为
            (batch, seq, feature). Default: ``False``
        :param input_dropout: 对输入的dropout概率. Default: 0
        :param hidden_dropout: 对每个隐状态的dropout概率. Default: 0
        :param bidirectional: 若为 ``True``, 使用双向的RNN. Default: ``False``
        """
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
                self._all_cells.append(VarRnnCellWrapper(
                    cell, self.hidden_size, input_dropout, hidden_dropout))
        initial_parameter(self)
        self.is_lstm = (self.mode == "LSTM")

    def _forward_one(self, n_layer, n_direction, input, hx, mask_x, mask_h):
        is_lstm = self.is_lstm
        idx = self.num_directions * n_layer + n_direction
        cell = self._all_cells[idx]
        hi = (hx[0][idx], hx[1][idx]) if is_lstm else hx[idx]
        output_x, hidden_x = cell(
            input, hi, mask_x, mask_h, is_reversed=(n_direction == 1))
        return output_x, hidden_x

    def forward(self, x, hx=None):
        r"""

        :param x: [batch, seq_len, input_size] 输入序列
        :param hx: [batch, hidden_size] 初始隐状态, 若为 ``None`` , 设为全1向量. Default: ``None``
        :return (output, ht): [batch, seq_len, hidden_size*num_direction] 输出序列
            和 [batch, hidden_size*num_direction] 最后时刻隐状态
        """
        is_lstm = self.is_lstm
        is_packed = isinstance(x, PackedSequence)
        if not is_packed:
            seq_len = x.size(1) if self.batch_first else x.size(0)
            max_batch_size = x.size(0) if self.batch_first else x.size(1)
            seq_lens = torch.LongTensor(
                [seq_len for _ in range(max_batch_size)])
            x = pack_padded_sequence(x, seq_lens, batch_first=self.batch_first)
        else:
            max_batch_size = int(x.batch_sizes[0])
        x, batch_sizes = x.data, x.batch_sizes

        if hx is None:
            hx = x.new_zeros(self.num_layers * self.num_directions,
                             max_batch_size, self.hidden_size, requires_grad=True)
            if is_lstm:
                hx = (hx, hx.new_zeros(hx.size(), requires_grad=True))

        mask_x = x.new_ones((max_batch_size, self.input_size))
        mask_out = x.new_ones(
            (max_batch_size, self.hidden_size * self.num_directions))
        mask_h_ones = x.new_ones((max_batch_size, self.hidden_size))
        nn.functional.dropout(mask_x, p=self.input_dropout,
                              training=self.training, inplace=True)
        nn.functional.dropout(mask_out, p=self.hidden_dropout,
                              training=self.training, inplace=True)

        hidden = x.new_zeros(
            (self.num_layers * self.num_directions, max_batch_size, self.hidden_size))
        if is_lstm:
            cellstate = x.new_zeros(
                (self.num_layers * self.num_directions, max_batch_size, self.hidden_size))
        for layer in range(self.num_layers):
            output_list = []
            input_seq = PackedSequence(x, batch_sizes)
            mask_h = nn.functional.dropout(
                mask_h_ones, p=self.hidden_dropout, training=self.training, inplace=False)
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
            x = torch.cat(output_list, dim=-1)

        if is_lstm:
            hidden = (hidden, cellstate)

        if is_packed:
            output = PackedSequence(x, batch_sizes)
        else:
            x = PackedSequence(x, batch_sizes)
            output, _ = pad_packed_sequence(x, batch_first=self.batch_first)

        return output, hidden


class VarLSTM(VarRNNBase):
    r"""
    Variational Dropout LSTM.
    相关论文参考：`A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (Yarin Gal and Zoubin Ghahramani, 2016) <https://arxiv.org/abs/1512.05287>`_

    """

    def __init__(self, *args, **kwargs):
        r"""
        
        :param input_size:  输入 `x` 的特征维度
        :param hidden_size: 隐状态  `h`  的特征维度
        :param num_layers: rnn的层数. Default: 1
        :param bias: 如果为 ``False``, 模型将不会使用bias. Default: ``True``
        :param batch_first: 若为 ``True``, 输入和输出 ``Tensor`` 形状为
            (batch, seq, feature). Default: ``False``
        :param input_dropout: 对输入的dropout概率. Default: 0
        :param hidden_dropout: 对每个隐状态的dropout概率. Default: 0
        :param bidirectional: 若为 ``True``, 使用双向的LSTM. Default: ``False``
        """
        super(VarLSTM, self).__init__(
            mode="LSTM", Cell=nn.LSTMCell, *args, **kwargs)

    def forward(self, x, hx=None):
        return super(VarLSTM, self).forward(x, hx)


class VarRNN(VarRNNBase):
    r"""
    Variational Dropout RNN.
    相关论文参考：`A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (Yarin Gal and Zoubin Ghahramani, 2016) <https://arxiv.org/abs/1512.05287>`_
    
    """

    def __init__(self, *args, **kwargs):
        r"""
        
        :param input_size:  输入 `x` 的特征维度
        :param hidden_size: 隐状态 `h` 的特征维度
        :param num_layers: rnn的层数. Default: 1
        :param bias: 如果为 ``False``, 模型将不会使用bias. Default: ``True``
        :param batch_first: 若为 ``True``, 输入和输出 ``Tensor`` 形状为
            (batch, seq, feature). Default: ``False``
        :param input_dropout: 对输入的dropout概率. Default: 0
        :param hidden_dropout: 对每个隐状态的dropout概率. Default: 0
        :param bidirectional: 若为 ``True``, 使用双向的RNN. Default: ``False``
        """
        super(VarRNN, self).__init__(
            mode="RNN", Cell=nn.RNNCell, *args, **kwargs)

    def forward(self, x, hx=None):
        return super(VarRNN, self).forward(x, hx)


class VarGRU(VarRNNBase):
    r"""
    Variational Dropout GRU.
    相关论文参考：`A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (Yarin Gal and Zoubin Ghahramani, 2016) <https://arxiv.org/abs/1512.05287>`_
    
    """

    def __init__(self, *args, **kwargs):
        r"""
        
        :param input_size:  输入 `x` 的特征维度
        :param hidden_size: 隐状态 `h` 的特征维度
        :param num_layers: rnn的层数. Default: 1
        :param bias: 如果为 ``False``, 模型将不会使用bias. Default: ``True``
        :param batch_first: 若为 ``True``, 输入和输出 ``Tensor`` 形状为
            (batch, seq, feature). Default: ``False``
        :param input_dropout: 对输入的dropout概率. Default: 0
        :param hidden_dropout: 对每个隐状态的dropout概率. Default: 0
        :param bidirectional: 若为 ``True``, 使用双向的GRU. Default: ``False``
        """
        super(VarGRU, self).__init__(
            mode="GRU", Cell=nn.GRUCell, *args, **kwargs)

    def forward(self, x, hx=None):
        return super(VarGRU, self).forward(x, hx)
