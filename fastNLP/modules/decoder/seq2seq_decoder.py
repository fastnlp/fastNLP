import torch.nn as nn
import torch
from torch.nn import LayerNorm
from ..encoder.seq2seq_encoder import MultiheadAttention
import torch.nn.functional as F
import math
from ...embeddings import StaticEmbedding
from ...core import Vocabulary
import abc
import torch
from typing import Union


class AttentionLayer(nn.Module):
    def __init__(self, input_size, encode_hidden_size, decode_hidden_size, bias=False):
        super().__init__()

        self.input_proj = nn.Linear(input_size, encode_hidden_size, bias=bias)
        self.output_proj = nn.Linear(input_size + encode_hidden_size, decode_hidden_size, bias=bias)

    def forward(self, input, encode_outputs, encode_mask):
        """

        :param input: batch_size x input_size
        :param encode_outputs: batch_size x max_len x encode_hidden_size
        :param encode_mask: batch_size x max_len
        :return: batch_size x decode_hidden_size, batch_size x max_len
        """

        # x: bsz x encode_hidden_size
        x = self.input_proj(input)

        # compute attention
        attn_scores = torch.matmul(encode_outputs, x.unsqueeze(-1)).squeeze(-1)  # b x max_len

        # don't attend over padding
        if encode_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encode_mask.eq(0),
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=-1)  # srclen x bsz

        # sum weighted sources
        x = torch.matmul(attn_scores.unsqueeze(1), encode_outputs).squeeze(1)  # b x encode_hidden_size

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


# ----- class past ----- #

class Past:
    def __init__(self):
        pass

    @abc.abstractmethod
    def num_samples(self):
        raise NotImplementedError

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0):
        if type(state) == torch.Tensor:
            state = state.index_select(index=indices, dim=dim)
        elif type(state) == list:
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif type(state) == tuple:
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))

        return state


class TransformerPast(Past):
    def __init__(self, num_decoder_layer: int = 6):
        super().__init__()
        self.encoder_output = None  # batch,src_seq,dim
        self.encoder_mask = None
        self.encoder_key = [None] * num_decoder_layer
        self.encoder_value = [None] * num_decoder_layer
        self.decoder_prev_key = [None] * num_decoder_layer
        self.decoder_prev_value = [None] * num_decoder_layer

    def num_samples(self):
        if self.encoder_key[0] is not None:
            return self.encoder_key[0].size(0)
        return None

    def reorder_past(self, indices: torch.LongTensor):
        self.encoder_output = self._reorder_state(self.encoder_output, indices)
        self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        self.encoder_key = self._reorder_state(self.encoder_key, indices)
        self.encoder_value = self._reorder_state(self.encoder_value, indices)
        self.decoder_prev_key = self._reorder_state(self.decoder_prev_key, indices)
        self.decoder_prev_value = self._reorder_state(self.decoder_prev_value, indices)


class LSTMPast(Past):
    def __init__(self):
        self.encoder_output = None  # batch,src_seq,dim
        self.encoder_mask = None
        self.prev_hidden = None  # n_layer,batch,dim
        self.pre_cell = None  # n_layer,batch,dim
        self.input_feed = None  # batch,dim

    def num_samples(self):
        if self.prev_hidden is not None:
            return self.prev_hidden.size(0)
        return None

    def reorder_past(self, indices: torch.LongTensor):
        self.encoder_output = self._reorder_state(self.encoder_output, indices)
        self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        self.prev_hidden = self._reorder_state(self.prev_hidden, indices, dim=1)
        self.pre_cell = self._reorder_state(self.pre_cell, indices, dim=1)
        self.input_feed = self._reorder_state(self.input_feed, indices)


# ------   #

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self._past = None

    def forward(self, tgt_prev_words, encoder_output, encoder_mask, past=None, return_attention=False):
        raise NotImplementedError

    def init_past(self, *args, **kwargs):
        raise NotImplementedError

    def reset_past(self):
        self._past = None

    def train(self, mode=True):
        self.reset_past()
        super().train()

    def reorder_past(self, indices: torch.LongTensor, past: Past = None):
        """
        根据indices中的index，将past的中状态置为正确的顺序

        :param torch.LongTensor indices:
        :param Past past:
        :return:
        """
        raise NotImplemented

    # def decode(self, *args, **kwargs) -> torch.Tensor:
    #     """
    #     当模型进行解码时，使用这个函数。只返回一个batch_size x vocab_size的结果。需要考虑一种特殊情况，即tokens长度不是1，即给定了
    #         解码句子开头的情况，这种情况需要查看Past中是否正确计算了decode的状态
    #
    #     :return:
    #     """
    #     raise NotImplemented

    @torch.no_grad()
    def decode(self, tgt_prev_words, encoder_output, encoder_mask, past=None) -> torch.Tensor:
        """
        :param tgt_prev_words: 传入的是完整的prev tokens
        :param encoder_output:
        :param encoder_mask:
        :param past
        :return:
        """
        if past is None:
            past = self._past
        assert past is not None
        output = self.forward(tgt_prev_words, encoder_output, encoder_mask, past)  # batch,1,vocab_size
        return output.squeeze(1)


class TransformerSeq2SeqDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 layer_idx: int = None):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_idx = layer_idx  # 记录layer的层索引，以方便获取past的信息

        self.self_attn = MultiheadAttention(d_model, n_head, dropout, layer_idx)
        self.self_attn_layer_norm = LayerNorm(d_model)

        self.encoder_attn = MultiheadAttention(d_model, n_head, dropout, layer_idx)
        self.encoder_attn_layer_norm = LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

        self.final_layer_norm = LayerNorm(self.d_model)

    def forward(self, x, encoder_output, encoder_mask=None, self_attn_mask=None, past=None):
        """

        :param x: (batch, seq_len, dim), decoder端的输入
        :param encoder_output: (batch,src_seq_len,dim)
        :param encoder_mask: batch,src_seq_len
        :param self_attn_mask: seq_len, seq_len，下三角的mask矩阵，只在训练时传入
        :param past: 只在inference阶段传入
        :return:
        """

        # self attention part
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              attn_mask=self_attn_mask,
                              past=past)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # encoder attention part
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn_weight = self.encoder_attn(query=x,
                                           key=encoder_output,
                                           value=encoder_output,
                                           key_mask=encoder_mask,
                                           past=past)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weight


class TransformerSeq2SeqDecoder(Seq2SeqDecoder):
    def __init__(self, vocab: Vocabulary, embed: nn.Module, pos_embed: nn.Module = None, num_layers: int = 6,
                 d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 output_embed: nn.Parameter = None):
        """

        :param embed: decoder端输入的embedding
        :param num_layers: Transformer Decoder层数
        :param d_model: Transformer参数
        :param n_head: Transformer参数
        :param dim_ff: Transformer参数
        :param dropout:
        :param output_embed: 输出embedding
        """
        super().__init__(vocab)

        self.embed = embed
        self.pos_embed = pos_embed
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqDecoderLayer(d_model, n_head, dim_ff, dropout, layer_idx)
                                           for layer_idx in range(num_layers)])

        self.embed_scale = math.sqrt(d_model)
        self.layer_norm = LayerNorm(d_model)
        self.output_embed = output_embed  # len(vocab), d_model

    def forward(self, tgt_prev_words, encoder_output, encoder_mask, past=None, return_attention=False):
        """

        :param tgt_prev_words: batch, tgt_len
        :param encoder_output: batch, src_len, dim
        :param encoder_mask: batch, src_seq
        :param past:
        :param return_attention:
        :return:
        """
        batch_size, max_tgt_len = tgt_prev_words.size()
        device = tgt_prev_words.device

        position = torch.arange(1, max_tgt_len + 1).unsqueeze(0).long().to(device)
        if past is not None:  # 此时在inference阶段
            position = position[:, -1]
            tgt_prev_words = tgt_prev_words[:-1]

        x = self.embed_scale * self.embed(tgt_prev_words)
        if self.pos_embed is not None:
            x += self.pos_embed(position)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if past is None:
            triangle_mask = self._get_triangle_mask(max_tgt_len)
            triangle_mask = triangle_mask.to(device)
        else:
            triangle_mask = None

        for layer in self.layer_stacks:
            x, attn_weight = layer(x=x,
                                   encoder_output=encoder_output,
                                   encoder_mask=encoder_mask,
                                   self_attn_mask=triangle_mask,
                                   past=past
                                   )

        x = self.layer_norm(x)  # batch, tgt_len, dim
        output = F.linear(x, self.output_embed)

        if return_attention:
            return output, attn_weight
        return output

    def reorder_past(self, indices: torch.LongTensor, past: TransformerPast = None) -> TransformerPast:
        if past is None:
            past = self._past
        past.reorder_past(indices)
        return past

    @property
    def past(self):
        return self._past

    def init_past(self, encoder_output=None, encoder_mask=None):
        self._past = TransformerPast(self.num_layers)
        self._past.encoder_output = encoder_output
        self._past.encoder_mask = encoder_mask

    @past.setter
    def past(self, past):
        assert isinstance(past, TransformerPast)
        self._past = past

    @staticmethod
    def _get_triangle_mask(max_seq_len):
        tensor = torch.ones(max_seq_len, max_seq_len)
        return torch.tril(tensor).byte()


class LSTMSeq2SeqDecoder(Seq2SeqDecoder):
    def __init__(self, vocab: Vocabulary, embed: nn.Module, num_layers: int = 3, hidden_size: int = 300,
                 dropout: float = 0.3, output_embed: nn.Parameter = None, attention=True):
        super().__init__(vocab)

        self.embed = embed
        self.output_embed = output_embed
        self.embed_dim = embed.embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.embed_dim + hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=False, dropout=dropout)
        self.attention_layer = AttentionLayer(hidden_size, self.embed_dim, hidden_size) if attention else None
        assert self.attention_layer is not None, "Attention Layer is required for now"  # todo 支持不做attention
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, tgt_prev_words, encoder_output, encoder_mask, past=None, return_attention=False):
        """

        :param tgt_prev_words: batch, tgt_len
        :param encoder_output:
            output: batch, src_len, dim
            (hidden,cell): num_layers, batch, dim
        :param encoder_mask: batch, src_seq
        :param past:
        :param return_attention:
        :return:
        """
        # input feed就是上一个时间步的最后一层layer的hidden state和out的融合

        batch_size, max_tgt_len = tgt_prev_words.size()
        device = tgt_prev_words.device
        src_output, (src_final_hidden, src_final_cell) = encoder_output
        if past is not None:
            tgt_prev_words = tgt_prev_words[:-1]  # 只取最后一个

        x = self.embed(tgt_prev_words)
        x = self.dropout_layer(x)

        attn_weights = [] if self.attention_layer is not None else None  # 保存attention weight, batch,tgt_seq,src_seq
        input_feed = None
        cur_hidden = None
        cur_cell = None

        if past is not None:  # 若past存在，则从中获取历史input feed
            input_feed = past.input_feed

        if input_feed is None:
            input_feed = src_final_hidden[-1]  # 以encoder的hidden作为初值, batch, dim
        decoder_out = []

        if past is not None:
            cur_hidden = past.prev_hidden
            cur_cell = past.prev_cell

        if cur_hidden is None:
            cur_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            cur_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # 开始计算
        for i in range(max_tgt_len):
            input = torch.cat(
                (x[:, i:i + 1, :],
                 input_feed[:, None, :]
                 ),
                dim=2
            )  # batch,1,2*dim
            _, (cur_hidden, cur_cell) = self.lstm(input, hx=(cur_hidden, cur_cell))  # hidden/cell保持原来的size
            if self.attention_layer is not None:
                input_feed, attn_weight = self.attention_layer(cur_hidden[-1], src_output, encoder_mask)
                attn_weights.append(attn_weight)
            else:
                input_feed = cur_hidden[-1]

            if past is not None:  # 保存状态
                past.input_feed = input_feed  # batch, hidden
                past.prev_hidden = cur_hidden
                past.prev_cell = cur_cell
            decoder_out.append(input_feed)

        decoder_out = torch.cat(decoder_out, dim=1)  # batch,seq_len,hidden
        decoder_out = self.dropout_layer(decoder_out)
        if attn_weights is not None:
            attn_weights = torch.cat(attn_weights, dim=1)  # batch, tgt_len, src_len

        output = F.linear(decoder_out, self.output_embed)
        if return_attention:
            return output, attn_weights
        return output

    def reorder_past(self, indices: torch.LongTensor, past: LSTMPast) -> LSTMPast:
        if past is None:
            past = self._past
        past.reorder_past(indices)

        return past

    def init_past(self, encoder_output=None, encoder_mask=None):
        self._past = LSTMPast()
        self._past.encoder_output = encoder_output
        self._past.encoder_mask = encoder_mask

    @property
    def past(self):
        return self._past

    @past.setter
    def past(self, past):
        assert isinstance(past, LSTMPast)
        self._past = past
