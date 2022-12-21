import torch.nn as nn
import torch
from torch.nn import LayerNorm
import torch.nn.functional as F
from typing import Union, Tuple
from ....core.utils import seq_len_to_mask
import math
from .lstm import LSTM
from ..attention import MultiHeadAttention
from ....embeddings.torch import StaticEmbedding
from ....embeddings.torch.utils import get_embeddings


__all__ = ['Seq2SeqEncoder', 'TransformerSeq2SeqEncoder', 'LSTMSeq2SeqEncoder']


class Seq2SeqEncoder(nn.Module):
    """
    所有 **Sequence2Sequence Encoder** 的基类。需要实现 :meth:`forward` 函数

    """
    def __init__(self):
        super().__init__()

    def forward(self, tokens: torch.LongTensor, seq_len: torch.LongTensor):
        """

        :param tokens: ``[batch_size, max_len]]``，encoder 的输入
        :param seq_len: ``[batch_size,]``
        :return:
        """
        raise NotImplementedError


class TransformerSeq2SeqEncoderLayer(nn.Module):
    """
    **Self-Attention** 的 Layer，

    :param int d_model: input和output的输出维度
    :param int n_head: 多少个head，每个head的维度为d_model/n_head
    :param int dim_ff: FFN的维度大小
    :param float dropout: Self-attention和FFN的dropout大小，0表示不drop
    """
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048,
                 dropout: float = 0.1):
        super(TransformerSeq2SeqEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.attn_layer_norm = LayerNorm(d_model)
        self.ffn_layer_norm = LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """

        :param x: batch_size, src_seq, d_model
        :param mask: batch_size, src_seq，为0的地方为padding
        :return:
        """
        # attention
        residual = x
        x = self.attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              key_mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerSeq2SeqEncoder(Seq2SeqEncoder):
    """
    基于 **Transformer** 的 :class:`Encoder`

    :param embed: ``decoder`` 输入的 embedding，支持以下几种输入类型：

            - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
            - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
            - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
            - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
    :param pos_embed: 位置 embedding
    :param d_model: 输入、输出的维度
    :param num_layers: :class:`TransformerSeq2SeqDecoderLayer` 的层数
    :param n_head: **多头注意力** head 的数目，需要能被 ``d_model`` 整除
    :param dim_ff: FFN 中间映射的维度
    :param dropout: :class:`~fastNLP.modules.torch.decoder.SelfAttention` 和 FFN 中的 dropout 的大小
    """
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], pos_embed: nn.Module = None,
                 d_model: int = 512, num_layers: int = 6, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1):
        super(TransformerSeq2SeqEncoder, self).__init__()
        self.embed = get_embeddings(embed)
        self.embed_scale = math.sqrt(d_model)
        self.pos_embed = pos_embed
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_fc = nn.Linear(self.embed.embedding_dim, d_model)
        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqEncoderLayer(d_model, n_head, dim_ff, dropout)
                                           for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, tokens, seq_len):
        """

        :param tokens: 输入序列，形状为 ``[batch_size, max_len]``
        :param seq_len: 序列长度，形状为 ``[batch_size, ]``，若为 ``None``，表示所有输入看做一样长
        :return: 一个元组，第一个元素形状为 ``[batch_size, max_len, d_model]`` 表示前向传播的结果，第二个元素形状为
            ``[batch_size, max_len]``， 表示产生的掩码 ``encoder_mask``，为 **0** 的地方为 padding
        """
        x = self.embed(tokens) * self.embed_scale  # batch, seq, dim
        batch_size, max_src_len, _ = x.size()
        device = x.device
        if self.pos_embed is not None:
            position = torch.arange(1, max_src_len + 1).unsqueeze(0).long().to(device)
            x += self.pos_embed(position)

        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        encoder_mask = seq_len_to_mask(seq_len, max_len=max_src_len)
        encoder_mask = encoder_mask.to(device)

        for layer in self.layer_stacks:
            x = layer(x, encoder_mask)

        x = self.layer_norm(x)

        return x, encoder_mask


class LSTMSeq2SeqEncoder(Seq2SeqEncoder):
    """
    **LSTM** 的 Encoder

    :param embed: ``decoder`` 输入的 embedding，支持以下几种输入类型：

            - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
            - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
            - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
            - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
    :param num_layers: LSTM 的层数
    :param hidden_size: 隐藏层大小, 该值也被认为是 ``encoder`` 的输出维度大小
    :param dropout: Dropout 的大小
    :param bidirectional: 是否使用双向
    """
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], num_layers: int = 3,
                 hidden_size: int = 400, dropout: float = 0.3, bidirectional: bool=True):
        super().__init__()
        self.embed = get_embeddings(embed)
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        hidden_size = hidden_size//2 if bidirectional else hidden_size
        self.lstm = LSTM(input_size=embed.embedding_dim, hidden_size=hidden_size, bidirectional=bidirectional,
                         batch_first=True, dropout=dropout if num_layers>1 else 0, num_layers=num_layers)

    def forward(self, tokens: torch.LongTensor, seq_len: torch.LongTensor):
        """

        :param tokens: 输入序列，形状为 ``[batch_size, max_len]``
        :param seq_len: 序列长度，形状为 ``[batch_size, ]``，若为 ``None``，表示所有输入看做一样长
        :return: 返回 ``((output, (ht, ct)), encoder_mask)`` 格式的结果。
        
                - ``output`` 形状为 ``[batch_size, seq_len, hidden_size*num_direction]``，表示输出序列；
                - ``ht`` 和 ``ct`` 形状为 ``[num_layers*num_direction, batch_size, hidden_size]``，表示最后时刻隐状态；
                - ``encoder_mask`` 形状为 ``[batch_size, max_len]``， 表示产生的掩码 ``encoder_mask``，为 **0** 的地方为 padding

        """
        x = self.embed(tokens)
        device = x.device
        x, (final_hidden, final_cell) = self.lstm(x, seq_len)
        encoder_mask = seq_len_to_mask(seq_len).to(device)

        # x: batch,seq_len,dim; h/c: num_layers*2,batch,dim

        if self.bidirectional:
            final_hidden = self.concat_bidir(final_hidden)  # 将双向的hidden state拼接起来，用于接下来的decoder的input
            final_cell = self.concat_bidir(final_cell)

        return (x, (final_hidden[-1], final_cell[-1])), encoder_mask  # 为了配合Seq2SeqBaseModel的forward，这边需要分为两个return

    def concat_bidir(self, input):
        output = input.view(self.num_layers, 2, input.size(1), -1).transpose(1, 2)
        return output.reshape(self.num_layers, input.size(1), -1)
