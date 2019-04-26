import torch.nn as nn


class Embedding(nn.Module):
    """Embedding组件."""

    def __init__(self, vocab_size, embed_dim, padding_idx=0, sparse=False, init_emb=None, dropout=0.0):
        """
        :param int vocab_size: 词表大小.
        :param int embed_dim: embedding维度.
        :param int padding_idx: 如果碰到padding_idx则自动补0.
        :param bool sparse: 如果为`True`则权重矩阵是一个sparse的矩阵.
        :param torch.Tensor init_emb: 初始的embedding矩阵.
        :param float dropout: dropout概率.
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx, sparse=sparse, _weight=init_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param torch.LongTensor x: [batch, seq_len]
        :return: torch.Tensor : [batch, seq_len, embed_dim]
        """
        x = self.embed(x)
        return self.dropout(x)
