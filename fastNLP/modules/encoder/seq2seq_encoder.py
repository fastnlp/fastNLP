r"""undocumented"""
import torch.nn as nn
import torch
from torch.nn import LayerNorm
import torch.nn.functional as F
from typing import Union, Tuple
from ...core.utils import seq_len_to_mask
import math
from ...modules.encoder.lstm import LSTM
from fastNLP.modules.attention import MultiHeadAttention
from ...embeddings import StaticEmbedding
from ...embeddings.utils import get_embeddings


class Seq2SeqEncoder(nn.Module):
    """
    所有Sequence2Sequence Encoder的基类。需要实现forward函数

    """
    def __init__(self):
        super().__init__()

    def forward(self, tokens, seq_len):
        """

        :param torch.LongTensor tokens: bsz x max_len, encoder的输入
        :param torch.LongTensor seq_len: bsz
        :return:
        """
        raise NotImplementedError


class TransformerSeq2SeqEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048,
                 dropout: float = 0.1):
        """
        Self-Attention的Layer，

        :param int d_model: input和output的输出维度
        :param int n_head: 多少个head，每个head的维度为d_model/n_head
        :param int dim_ff: FFN的维度大小
        :param float dropout: Self-attention和FFN的dropout大小，0表示不drop
        """
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

        :param x: batch x src_seq x d_model
        :param mask: batch x src_seq，为0的地方为padding
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
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], pos_embed = None,
                 num_layers = 6, d_model = 512, n_head = 8, dim_ff = 2048, dropout = 0.1):
        """
        基于Transformer的Encoder

        :param embed: encoder输入token的embedding
        :param nn.Module pos_embed: position embedding
        :param int num_layers: 多少层的encoder
        :param int d_model: 输入输出的维度
        :param int n_head: 多少个head
        :param int dim_ff: FFN中间的维度大小
        :param float dropout: Attention和FFN的dropout大小
        """
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

        :param tokens: batch x max_len
        :param seq_len: [batch]
        :return: bsz x max_len x d_model, bsz x max_len(为0的地方为padding)
        """
        x = self.embed(tokens) * self.embed_scale  # batch, seq, dim
        batch_size, max_src_len, _ = x.size()
        device = x.device
        if self.pos_embed is not None:
            position = torch.arange(1, max_src_len + 1).unsqueeze(0).long().to(device)
            x += self.pos_embed(position)

        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        encoder_mask = seq_len_to_mask(seq_len)
        encoder_mask = encoder_mask.to(device)

        for layer in self.layer_stacks:
            x = layer(x, encoder_mask)

        x = self.layer_norm(x)

        return x, encoder_mask


class LSTMSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], num_layers = 3,
                 hidden_size = 400, dropout = 0.3, bidirectional=True):
        """
        LSTM的Encoder

        :param embed: encoder的token embed
        :param int num_layers: 多少层
        :param int hidden_size: LSTM隐藏层、输出的大小
        :param float dropout: LSTM层之间的Dropout是多少
        :param bool bidirectional: 是否使用双向
        """
        super().__init__()
        self.embed = get_embeddings(embed)
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        hidden_size = hidden_size//2 if bidirectional else hidden_size
        self.lstm = LSTM(input_size=embed.embedding_dim, hidden_size=hidden_size, bidirectional=bidirectional,
                         batch_first=True, dropout=dropout if num_layers>1 else 0, num_layers=num_layers)

    def forward(self, tokens, seq_len):
        """

        :param torch.LongTensor tokens: bsz x max_len
        :param torch.LongTensor seq_len: bsz
        :return: (output, (hidden, cell)), encoder_mask
            output: bsz x max_len x hidden_size,
            hidden,cell: batch_size x hidden_size, 最后一层的隐藏状态或cell状态
            encoder_mask: bsz x max_len, 为0的地方是padding
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
