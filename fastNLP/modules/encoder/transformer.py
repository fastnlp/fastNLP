"""undocumented"""

__all__ = [
    "TransformerEncoder"
]
from torch import nn

from .attention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    """
    transformer的encoder模块，不包含embedding层

    """

    class SubLayer(nn.Module):
        def __init__(self, model_size, inner_size, key_size, value_size, num_head, dropout=0.1):
            super(TransformerEncoder.SubLayer, self).__init__()
            self.atte = MultiHeadAttention(model_size, key_size, value_size, num_head, dropout)
            self.norm1 = nn.LayerNorm(model_size, eps=1e-6)
            self.ffn = nn.Sequential(nn.Linear(model_size, inner_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(inner_size, model_size))
            self.norm2 = nn.LayerNorm(model_size, eps=1e-6)
            self.dropout = nn.Dropout(dropout)

        def forward(self, input, seq_mask=None, atte_mask_out=None):
            """

            :param input: [batch, seq_len, model_size]
            :param seq_mask: [batch, seq_len]
            :return: [batch, seq_len, model_size]
            """
            if seq_mask is None:  # 防止后续乘法时出错
                seq_mask = 1
            input = self.norm1(input)
            attention = self.atte(input, input, input, atte_mask_out)
            input = input + self.dropout(attention)
            attention *= seq_mask
            input = self.norm2(input)
            output = self.ffn(input)
            input = input + self.dropout(output)
            input *= seq_mask
            return input

    def __init__(self, num_layers, **kargs):
        """
        
        :param int num_layers: transformer的层数
        :param int model_size: 输入维度的大小。同时也是输出维度的大小。
        :param int inner_size: FFN层的hidden大小
        :param int key_size: 每个head的维度大小。
        :param int value_size: 每个head中value的维度。
        :param int num_head: head的数量。
        :param float dropout: dropout概率. Default: 0.1
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([self.SubLayer(**kargs) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(kargs['model_size'], eps=1e-6)

    def forward(self, x, seq_mask=None):
        """
        :param x: [batch, seq_len, model_size] 输入序列
        :param seq_mask: [batch, seq_len] 输入序列的padding mask, 若为 ``None`` , 生成全1向量.
            Default: ``None``
        :return: [batch, seq_len, model_size] 输出序列
        """
        output = x
        if seq_mask is None:
            atte_mask_out = None
        else:
            atte_mask_out = (seq_mask == 0)[:, None, :]
            seq_mask = seq_mask[:, :, None]
        for layer in self.layers:
            output = layer(output, seq_mask, atte_mask_out)
        return self.norm(output)
