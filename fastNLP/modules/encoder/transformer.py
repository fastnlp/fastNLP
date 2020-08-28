r"""undocumented"""

__all__ = [
    "TransformerEncoder"
]
from torch import nn

from .seq2seq_encoder import TransformerSeq2SeqEncoderLayer


class TransformerEncoder(nn.Module):
    r"""
    transformer的encoder模块，不包含embedding层

    """
    def __init__(self, num_layers, d_model=512, n_head=8, dim_ff=2048, dropout=0.1):
        """

        :param int num_layers: 多少层Transformer
        :param int d_model: input和output的大小
        :param int n_head: 多少个head
        :param int dim_ff: FFN中间hidden大小
        :param float dropout: 多大概率drop attention和ffn中间的表示
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerSeq2SeqEncoderLayer(d_model = d_model, n_head = n_head, dim_ff = dim_ff,
                 dropout = dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, seq_mask=None):
        r"""
        :param x: [batch, seq_len, model_size] 输入序列
        :param seq_mask: [batch, seq_len] 输入序列的padding mask, 若为 ``None`` , 生成全1向量. 为1的地方需要attend
            Default: ``None``
        :return: [batch, seq_len, model_size] 输出序列
        """
        output = x
        if seq_mask is None:
            seq_mask = x.new_ones(x.size(0), x.size(1)).bool()
        for layer in self.layers:
            output = layer(output, seq_mask)
        return self.norm(output)
