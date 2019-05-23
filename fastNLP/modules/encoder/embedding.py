__all__ = [
    "Embedding"
]
import torch.nn as nn
from ..utils import get_embeddings


class Embedding(nn.Embedding):
    """
    别名：:class:`fastNLP.modules.Embedding`   :class:`fastNLP.modules.encoder.embedding.Embedding`

    Embedding组件. 可以通过self.num_embeddings获取词表大小; self.embedding_dim获取embedding的维度"""
    
    def __init__(self, init_embed, padding_idx=None, dropout=0.0, sparse=False, max_norm=None, norm_type=2,
                 scale_grad_by_freq=False):
        """

        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
        :param None,int padding_idx: 该index的Embedding将一直为0.
        :param float dropout: 对Embedding的输出的dropout。
        :param bool sparse: 如果为True，则对Embedding的梯度将是sparse的，参考Pytorch Embedding获取更多信息。
        :param None,float max_norm: 每个vector最大的norm能为多大
        :param int norm_type: norm的类型
        :param bool scale_grad_by_freq: 如果为True，将会把梯度除以这个词出现的次数.
        """
        embed = get_embeddings(init_embed)
        num_embeddings, embedding_dim = embed.weight.size()
        
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                         max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse, _weight=embed.weight.data)
        del embed
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        :param torch.LongTensor x: [batch, seq_len]
        :return: torch.Tensor : [batch, seq_len, embed_dim]
        """
        x = super().forward(x)
        return self.dropout(x)

    def size(self):
        """
        Embedding的大小
        :return: torch.Size()
        """
        return self.weight.size()
