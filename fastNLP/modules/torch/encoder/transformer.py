__all__ = [
    "TransformerEncoder"
]

from torch import nn

from .seq2seq_encoder import TransformerSeq2SeqEncoderLayer


class TransformerEncoder(nn.Module):
    r"""
    **Transformer** 的 encoder 模块，不包含 embedding 层。

    :param num_layers: **TransformerEncoder** 的层数。
    :param d_model: 输入维度的大小，同时也是输出维度的大小。
    :param n_head: **多头注意力** head 的数目，需要能被 ``d_model`` 整除
    :param dim_ff: FFN 中间映射的维度
    :param dropout: :class:`~fastNLP.modules.torch.decoder.SelfAttention` 和 FFN 中的 dropout 的大小
    """
    def __init__(self, num_layers: int, d_model: int=512, n_head: int=8, dim_ff: int=2048, dropout: float=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerSeq2SeqEncoderLayer(d_model = d_model, n_head = n_head, dim_ff = dim_ff,
                 dropout = dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, seq_mask=None):
        r"""
        :param x: 输入序列，形状为 ``[batch_size, seq_len, d_model]``
        :param seq_mask: 输入序列的 padding mask ，形状为 ``[batch, seq_len]``，若为 ``None``，则生成全 **1** 向量；为 **1**
            的地方表示需要 attend 。
        :return: 输出序列，形状为 ``[batch, seq_len, d_model]``
        """
        output = x
        if seq_mask is None:
            seq_mask = x.new_ones(x.size(0), x.size(1)).bool()
        for layer in self.layers:
            output = layer(output, seq_mask)
        return self.norm(output)
