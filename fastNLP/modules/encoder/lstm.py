"""
轻量封装的 Pytorch LSTM 模块.
可在 forward 时传入序列的长度, 自动对padding做合适的处理.
"""
__all__ = [
    "LSTM"
]

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from ..utils import initial_parameter


class LSTM(nn.Module):
    """
    别名：:class:`fastNLP.modules.LSTM`  :class:`fastNLP.modules.encoder.lstm.LSTM`

    LSTM 模块, 轻量封装的Pytorch LSTM

    :param input_size:  输入 `x` 的特征维度
    :param hidden_size: 隐状态 `h` 的特征维度
    :param num_layers: rnn的层数. Default: 1
    :param dropout: 层间dropout概率. Default: 0
    :param bidirectional: 若为 ``True``, 使用双向的RNN. Default: ``False``
    :param batch_first: 若为 ``True``, 输入和输出 ``Tensor`` 形状为
        :(batch, seq, feature). Default: ``False``
    :param bias: 如果为 ``False``, 模型将不会使用bias. Default: ``True``
    """
    
    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, batch_first=True,
                 bidirectional=False, bias=True, initial_method=None):
        super(LSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        initial_parameter(self, initial_method)
    
    def forward(self, x, seq_len=None, h0=None, c0=None):
        """

        :param x: [batch, seq_len, input_size] 输入序列
        :param seq_len: [batch, ] 序列长度, 若为 ``None``, 所有输入看做一样长. Default: ``None``
        :param h0: [batch, hidden_size] 初始隐状态, 若为 ``None`` , 设为全1向量. Default: ``None``
        :param c0: [batch, hidden_size] 初始Cell状态, 若为 ``None`` , 设为全1向量. Default: ``None``
        :return (output, ht) 或 output: 若 ``get_hidden=True`` [batch, seq_len, hidden_size*num_direction] 输出序列
            和 [batch, hidden_size*num_direction] 最后时刻隐状态.
        """
        if h0 is not None and c0 is not None:
            hx = (h0, c0)
        else:
            hx = None
        if seq_len is not None and not isinstance(x, rnn.PackedSequence):
            sort_lens, sort_idx = torch.sort(seq_len, dim=0, descending=True)
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
        return output, hx
