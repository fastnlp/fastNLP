import torch.nn as nn
import torch
from torch.nn import LayerNorm
import torch.nn.functional as F
from typing import Union, Tuple
from ...core.utils import seq_len_to_mask
import math
from ...core import Vocabulary
from ...modules import LSTM


class MultiheadAttention(nn.Module):  # todo 这个要放哪里？
    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.0, layer_idx: int = None):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_model // n_head
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.reset_parameters()

    def forward(self, query, key, value, key_mask=None, attn_mask=None, past=None):
        """

        :param query: batch x seq x dim
        :param key:
        :param value:
        :param key_mask: batch x seq 用于指示哪些key不要attend到；注意到mask为1的地方是要attend到的
        :param attn_mask: seq x seq, 用于mask掉attention map。 主要是用在训练时decoder端的self attention，下三角为1
        :param past: 过去的信息，在inference的时候会用到，比如encoder output、decoder的prev kv。这样可以减少计算。
        :return:
        """
        assert key.size() == value.size()
        if past is not None:
            assert self.layer_idx is not None
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()

        q = self.q_proj(query)  # batch x seq x dim
        q *= self.scaling
        k = v = None
        prev_k = prev_v = None

        # 从past中取kv
        if past is not None:  # 说明此时在inference阶段
            if qkv_same:  # 此时在decoder self attention
                prev_k = past.decoder_prev_key[self.layer_idx]
                prev_v = past.decoder_prev_value[self.layer_idx]
            else:  # 此时在decoder-encoder attention，直接将保存下来的key装载起来即可
                k = past.encoder_key[self.layer_idx]
                v = past.encoder_value[self.layer_idx]

        if k is None:
            k = self.k_proj(key)
            v = self.v_proj(value)

        if prev_k is not None:
            k = torch.cat((prev_k, k), dim=1)
            v = torch.cat((prev_v, v), dim=1)

        # 更新past
        if past is not None:
            if qkv_same:
                past.decoder_prev_key[self.layer_idx] = k
                past.decoder_prev_value[self.layer_idx] = v
            else:
                past.encoder_key[self.layer_idx] = k
                past.encoder_value[self.layer_idx] = v

        # 开始计算attention
        batch_size, q_len, d_model = query.size()
        k_len, v_len = k.size(1), v.size(1)
        q = q.contiguous().view(batch_size, q_len, self.n_head, self.head_dim)
        k = k.contiguous().view(batch_size, k_len, self.n_head, self.head_dim)
        v = v.contiguous().view(batch_size, v_len, self.n_head, self.head_dim)

        attn_weights = torch.einsum('bqnh,bknh->bqkn', q, k)  # bs,q_len,k_len,n_head
        if key_mask is not None:
            _key_mask = ~key_mask[:, None, :, None].bool()  # batch,1,k_len,n_head
            attn_weights = attn_weights.masked_fill(_key_mask, -float('inf'))

        if attn_mask is not None:
            _attn_mask = ~attn_mask[None, :, :, None].bool()  # 1,q_len,k_len,n_head
            attn_weights = attn_weights.masked_fill(_attn_mask, -float('inf'))

        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.einsum('bqkn,bknh->bqnh', attn_weights, v)  # batch,q_len,n_head,head_dim
        output = output.reshape(batch_size, q_len, -1)
        output = self.out_proj(output)  # batch,q_len,dim

        return output, attn_weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def set_layer_idx(self, layer_idx):
        self.layer_idx = layer_idx


class TransformerSeq2SeqEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048,
                 dropout: float = 0.1):
        super(TransformerSeq2SeqEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.self_attn = MultiheadAttention(d_model, n_head, dropout)
        self.attn_layer_norm = LayerNorm(d_model)
        self.ffn_layer_norm = LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, encoder_mask):
        """

        :param x: batch,src_seq,dim
        :param encoder_mask: batch,src_seq
        :return:
        """
        # attention
        residual = x
        x = self.attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              key_mask=encoder_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, src_words, src_seq_len):
        raise NotImplementedError


class TransformerSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self, vocab: Vocabulary, embed: nn.Module, pos_embed: nn.Module = None, num_layers: int = 6,
                 d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1):
        super(TransformerSeq2SeqEncoder, self).__init__(vocab)
        self.embed = embed
        self.embed_scale = math.sqrt(d_model)
        self.pos_embed = pos_embed
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqEncoderLayer(d_model, n_head, dim_ff, dropout)
                                           for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, src_words, src_seq_len):
        """

        :param src_words: batch, src_seq_len
        :param src_seq_len: [batch]
        :return:
        """
        batch_size, max_src_len = src_words.size()
        device = src_words.device
        x = self.embed(src_words) * self.embed_scale  # batch, seq, dim
        if self.pos_embed is not None:
            position = torch.arange(1, max_src_len + 1).unsqueeze(0).long().to(device)
            x += self.pos_embed(position)
        x = F.dropout(x, p=self.dropout, training=self.training)

        encoder_mask = seq_len_to_mask(src_seq_len)
        encoder_mask = encoder_mask.to(device)

        for layer in self.layer_stacks:
            x = layer(x, encoder_mask)

        x = self.layer_norm(x)

        return x, encoder_mask


class LSTMSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self, vocab: Vocabulary, embed: nn.Module, num_layers: int = 3, hidden_size: int = 400,
                 dropout: float = 0.3, bidirectional=True):
        super().__init__(vocab)
        self.embed = embed
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = LSTM(input_size=embed.embedding_dim, hidden_size=hidden_size // 2, bidirectional=bidirectional,
                         batch_first=True, dropout=dropout, num_layers=num_layers)

    def forward(self, src_words, src_seq_len):
        batch_size = src_words.size(0)
        device = src_words.device
        x = self.embed(src_words)
        x, (final_hidden, final_cell) = self.lstm(x, src_seq_len)
        encoder_mask = seq_len_to_mask(src_seq_len).to(device)

        # x: batch,seq_len,dim; h/c: num_layers*2,batch,dim

        def concat_bidir(input):
            output = input.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
            return output.view(self.num_layers, batch_size, -1)

        if self.bidirectional:
            final_hidden = concat_bidir(final_hidden)  # 将双向的hidden state拼接起来，用于接下来的decoder的input
            final_cell = concat_bidir(final_cell)

        return (x, (final_hidden, final_cell)), encoder_mask  # 为了配合Seq2SeqBaseModel的forward，这边需要分为两个return
