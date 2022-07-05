r"""
轻量封装的 **Pytorch LSTM** 模块.
可在 :meth:`forward` 时传入序列的长度, 自动对 padding 做合适的处理.
"""

__all__ = [
    "LSTM"
]

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.utils.rnn as rnn
    from torch.nn import Module
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Module


class LSTM(Module):
    r"""
    **LSTM** 模块，轻量封装的 **Pytorch LSTM** 。在提供 ``seq_len`` 的情况下，将自动使用 ``pack_padded_sequence``；同时默认将 ``forget gate``
    的 bias 初始化为 **1**，且可以应对 :class:`DataParallel` 中 LSTM 的使用问题。
    
    :param input_size:  输入 `x` 的特征维度
    :param hidden_size: 隐状态 `h` 的特征维度. 如果 ``bidirectional`` 为 ``True``，则输出的维度会是 ``hidde_size*2``
    :param num_layers: rnn 的层数
    :param dropout: 层间 dropout 概率
    :param bidirectional: 若为 ``True``, 使用双向的 RNN
    :param batch_first: 若为 ``True``, 输入和输出 :class:`torch.Tensor` 形状为 ``[batch_size, seq_len, feature]``，否则为
        ``[seq_len, batch_size, features]``
    :param bias: 如果为 ``False``, 模型将不会使用 bias
    """

    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, batch_first=True,
                 bidirectional=False, bias=True):
        super(LSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.init_param()

    def init_param(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                # based on https://github.com/pytorch/pytorch/issues/750#issuecomment-280671871
                param.data.fill_(0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, seq_len=None, h0=None, c0=None):
        r"""
        :param x: 输入序列，形状为 ``[batch_size, seq_len, input_size]``
        :param seq_len: 序列长度，形状为 ``[batch_size, ]``，若为 ``None``，表示所有输入看做一样长
        :param h0: 初始隐状态，形状为 ``[batch_size, hidden_size]``，若为 ``None`` ，设为全 **0** 向量
        :param c0: 初始 ``Cell`` 状态，形状为 ``[batch_size, hidden_size]``，若为 ``None`` ，设为全 **0** 向量
        :return: 返回 ``(output, (ht, ct))`` 格式的结果。``output`` 形状为 ``[batch_size, seq_len, hidden_size*num_direction]``，表示输出序列；
            ``ht`` 和 ``ct`` 形状为 ``[num_layers*num_direction, batch_size, hidden_size]``，表示最后时刻隐状态。
        """
        batch_size, max_len, _ = x.size()
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
            x = rnn.pack_padded_sequence(x, sort_lens.cpu(), batch_first=self.batch_first)
            output, hx = self.lstm(x, hx)  # -> [N,L,C]
            output, _ = rnn.pad_packed_sequence(output, batch_first=self.batch_first, total_length=max_len)
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            if self.batch_first:
                output = output[unsort_idx]
            else:
                output = output[:, unsort_idx]
            hx = hx[0][:, unsort_idx], hx[1][:, unsort_idx]
        else:
            output, hx = self.lstm(x, hx)
        return output, hx