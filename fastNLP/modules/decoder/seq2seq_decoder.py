# coding=utf-8
import torch
from torch import nn
import abc
import torch.nn.functional as F
from fastNLP.embeddings import StaticEmbedding
import numpy as np
from typing import Union, Tuple
from fastNLP.embeddings import get_embeddings
from fastNLP.modules import LSTM
from torch.nn import LayerNorm
import math
from reproduction.Summarization.Baseline.tools.PositionEmbedding import \
    get_sinusoid_encoding_table  # todo: 应该将position embedding移到core


class Past:
    def __init__(self):
        pass

    @abc.abstractmethod
    def num_samples(self):
        pass


class TransformerPast(Past):
    def __init__(self, encoder_outputs: torch.Tensor = None, encoder_mask: torch.Tensor = None,
                 num_decoder_layer: int = 6):
        """

        :param encoder_outputs: (batch,src_seq_len,dim)
        :param encoder_mask: (batch,src_seq_len)
        :param encoder_key: list of (batch, src_seq_len, dim)
        :param encoder_value:
        :param decoder_prev_key:
        :param decoder_prev_value:
        """
        self.encoder_outputs = encoder_outputs
        self.encoder_mask = encoder_mask
        self.encoder_key = [None] * num_decoder_layer
        self.encoder_value = [None] * num_decoder_layer
        self.decoder_prev_key = [None] * num_decoder_layer
        self.decoder_prev_value = [None] * num_decoder_layer

    def num_samples(self):
        if self.encoder_outputs is not None:
            return self.encoder_outputs.size(0)
        return None

    def _reorder_state(self, state, indices):
        if type(state) == torch.Tensor:
            state = state.index_select(index=indices, dim=0)
        elif type(state) == list:
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = state[i].index_select(index=indices, dim=0)
        else:
            raise ValueError('State does not support other format')

        return state

    def reorder_past(self, indices: torch.LongTensor):
        self.encoder_outputs = self._reorder_state(self.encoder_outputs, indices)
        self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        self.encoder_key = self._reorder_state(self.encoder_key, indices)
        self.encoder_value = self._reorder_state(self.encoder_value, indices)
        self.decoder_prev_key = self._reorder_state(self.decoder_prev_key, indices)
        self.decoder_prev_value = self._reorder_state(self.decoder_prev_value, indices)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def reorder_past(self, indices: torch.LongTensor, past: Past) -> Past:
        """
        根据indices中的index，将past的中状态置为正确的顺序

        :param torch.LongTensor indices:
        :param Past past:
        :return:
        """
        raise NotImplemented

    def decode_one(self, *args, **kwargs) -> Tuple[torch.Tensor, Past]:
        """
        当模型进行解码时，使用这个函数。只返回一个batch_size x vocab_size的结果。需要考虑一种特殊情况，即tokens长度不是1，即给定了
            解码句子开头的情况，这种情况需要查看Past中是否正确计算了decode的状态

        :return:
        """
        raise NotImplemented


class DecoderMultiheadAttention(nn.Module):
    """
    Transformer Decoder端的multihead layer
    相比原版的Multihead功能一致，但能够在inference时加速
    参考fairseq
    """

    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.0, layer_idx: int = None):
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

    def forward(self, query, key, value, self_attn_mask=None, encoder_attn_mask=None, past=None, inference=False):
        """

        :param query: (batch, seq_len, dim)
        :param key: (batch, seq_len, dim)
        :param value: (batch, seq_len, dim)
        :param self_attn_mask: None or ByteTensor (1, seq_len, seq_len)
        :param encoder_attn_mask: (batch, src_len) ByteTensor
        :param past: required for now
        :param inference:
        :return: x和attention weight
        """
        if encoder_attn_mask is not None:
            assert self_attn_mask is None
        assert past is not None, "Past is required for now"
        is_encoder_attn = True if encoder_attn_mask is not None else False

        q = self.q_proj(query)  # (batch,q_len,dim)
        q *= self.scaling
        k = v = None
        prev_k = prev_v = None

        if inference and is_encoder_attn and past.encoder_key[self.layer_idx] is not None:
            k = past.encoder_key[self.layer_idx]  # (batch,k_len,dim)
            v = past.encoder_value[self.layer_idx]  # (batch,v_len,dim)
        else:
            if inference and not is_encoder_attn and past.decoder_prev_key[self.layer_idx] is not None:
                prev_k = past.decoder_prev_key[self.layer_idx]  # (batch, seq_len, dim)
                prev_v = past.decoder_prev_value[self.layer_idx]

        if k is None:
            k = self.k_proj(key)
            v = self.v_proj(value)
        if prev_k is not None:
            k = torch.cat((prev_k, k), dim=1)
            v = torch.cat((prev_v, v), dim=1)

        # 更新past
        if inference and is_encoder_attn and past.encoder_key[self.layer_idx] is None:
            past.encoder_key[self.layer_idx] = k
            past.encoder_value[self.layer_idx] = v
        if inference and not is_encoder_attn:
            past.decoder_prev_key[self.layer_idx] = prev_k
            past.decoder_prev_value[self.layer_idx] = prev_v

        batch_size, q_len, d_model = query.size()
        k_len, v_len = key.size(1), value.size(1)
        q = q.contiguous().view(batch_size, q_len, self.n_head, self.head_dim)
        k = k.contiguous().view(batch_size, k_len, self.n_head, self.head_dim)
        v = v.contiguous().view(batch_size, v_len, self.n_head, self.head_dim)

        attn_weights = torch.einsum('bqnh,bknh->bqkn', q, k)  # bs,q_len,k_len,n_head
        mask = encoder_attn_mask if is_encoder_attn else self_attn_mask
        if mask is not None:
            if len(mask.size()) == 2:  # 是encoder mask, batch,src_len/k_len
                mask = mask[:, None, :, None]
            else:  # (1, seq_len, seq_len)
                mask = mask[...:None]
            _mask = mask

            attn_weights = attn_weights.masked_fill(_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.einsum('bqkn,bknh->bqnh', attn_weights, v)  # batch,q_len,n_head,head_dim
        output = output.view(batch_size, q_len, -1)
        output = self.out_proj(output)  # batch,q_len,dim

        return output, attn_weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj)
        nn.init.xavier_uniform_(self.k_proj)
        nn.init.xavier_uniform_(self.v_proj)
        nn.init.xavier_uniform_(self.out_proj)


class TransformerSeq2SeqDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 layer_idx: int = None):
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_idx = layer_idx  # 记录layer的层索引，以方便获取past的信息

        self.self_attn = DecoderMultiheadAttention(d_model, n_head, dropout, layer_idx)
        self.self_attn_layer_norm = LayerNorm(d_model)

        self.encoder_attn = DecoderMultiheadAttention(d_model, n_head, dropout, layer_idx)
        self.encoder_attn_layer_norm = LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

        self.final_layer_norm = LayerNorm(self.d_model)

    def forward(self, x, encoder_outputs, self_attn_mask=None, encoder_attn_mask=None, past=None, inference=False):
        """

        :param x: (batch, seq_len, dim)
        :param encoder_outputs: (batch,src_seq_len,dim)
        :param self_attn_mask:
        :param encoder_attn_mask:
        :param past:
        :param inference:
        :return:
        """
        if inference:
            assert past is not None, "Past is required when inference"

        # self attention part
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              self_attn_mask=self_attn_mask,
                              past=past,
                              inference=inference)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # encoder attention part
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn_weight = self.encoder_attn(query=x,
                                           key=past.encoder_outputs,
                                           value=past.encoder_outputs,
                                           encoder_attn_mask=past.encoder_mask,
                                           past=past,
                                           inference=inference)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weight


class TransformerSeq2SeqDecoder(Decoder):
    def __init__(self, embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray], num_layers: int = 6,
                 d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 output_embed: Union[Tuple[int, int], int, nn.Module, torch.Tensor, np.ndarray] = None,
                 bind_input_output_embed=False):
        """

        :param embed: decoder端输入的embedding
        :param num_layers: Transformer Decoder层数
        :param d_model: Transformer参数
        :param n_head: Transformer参数
        :param dim_ff: Transformer参数
        :param dropout:
        :param output_embed: 输出embedding
        :param bind_input_output_embed: 是否共享输入输出的embedding权重
        """
        super(TransformerSeq2SeqDecoder, self).__init__()
        self.token_embed = get_embeddings(embed)
        self.dropout = dropout

        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqDecoderLayer(d_model, n_head, dim_ff, dropout, layer_idx)
                                           for layer_idx in range(num_layers)])

        if isinstance(output_embed, int):
            output_embed = (output_embed, d_model)
            output_embed = get_embeddings(output_embed)
        elif output_embed is not None:
            assert not bind_input_output_embed, "When `output_embed` is not None, " \
                                                "`bind_input_output_embed` must be False."
            if isinstance(output_embed, StaticEmbedding):
                for i in self.token_embed.words_to_words:
                    assert i == self.token_embed.words_to_words[i], "The index does not match."
                output_embed = self.token_embed.embedding.weight
            else:
                output_embed = get_embeddings(output_embed)
        else:
            if not bind_input_output_embed:
                raise RuntimeError("You have to specify output embedding.")

        # todo: 由于每个模型都有embedding的绑定或其他操作，建议挪到外部函数以减少冗余，可参考fairseq
        self.pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position=1024, d_hid=d_model, padding_idx=0),
            freeze=True
        )

        if bind_input_output_embed:
            assert output_embed is None, "When `bind_input_output_embed=True`, `output_embed` must be None"
            if isinstance(self.token_embed, StaticEmbedding):
                for i in self.token_embed.words_to_words:
                    assert i == self.token_embed.words_to_words[i], "The index does not match."
            self.output_embed = nn.Parameter(self.token_embed.weight.transpose(0, 1))
        else:
            if isinstance(output_embed, nn.Embedding):
                self.output_embed = nn.Parameter(output_embed.weight.transpose(0, 1))
            else:
                self.output_embed = output_embed.transpose(0, 1)
            self.output_hidden_size = self.output_embed.size(0)

        self.embed_scale = math.sqrt(d_model)

    def forward(self, tokens, past, return_attention=False, inference=False):
        """

        :param tokens: torch.LongTensor, tokens: batch_size x decode_len
        :param past: TransformerPast: 包含encoder输出及mask，在inference阶段保存了上一时刻的key和value以减少矩阵运算
        :param return_attention:
        :param inference: 是否在inference阶段
        :return:
        """
        assert past is not None
        batch_size, decode_len = tokens.size()
        device = tokens.device
        if not inference:
            self_attn_mask = self._get_triangle_mask(decode_len)
            self_attn_mask = self_attn_mask.to(device)[None, :, :]  # 1,seq,seq
        else:
            self_attn_mask = None
        tokens = self.token_embed(tokens) * self.embed_scale  # bs,decode_len,embed_dim
        pos = self.pos_embed(tokens)  # bs,decode_len,embed_dim
        tokens = pos + tokens
        if inference:
            tokens = tokens[:, -1:, :]

        x = F.dropout(tokens, p=self.dropout, training=self.training)
        for layer in self.layer_stacks:
            x, attn_weight = layer(x, past.encoder_outputs, self_attn_mask=self_attn_mask,
                                   encoder_attn_mask=past.encoder_mask, past=past, inference=inference)

        output = torch.matmul(x, self.output_embed)

        if return_attention:
            return output, attn_weight
        return output

    @torch.no_grad()
    def decode_one(self, tokens, past) -> Tuple[torch.Tensor, Past]:
        """
        # todo: 对于transformer而言，因为position的原因，需要输入整个prefix序列，因此lstm的decode one和beam search需要改一下，以统一接口
        # todo: 是否不需要return past？ 因为past已经被改变了，不需要显式return？
        :param tokens: torch.LongTensor (batch_size,1)
        :param past: TransformerPast
        :return:
        """
        output = self.forward(tokens, past, inference=True)  # batch,1,vocab_size
        return output.squeeze(1), past

    def reorder_past(self, indices: torch.LongTensor, past: TransformerPast) -> TransformerPast:
        past.reorder_past(indices)
        return past  # todo : 其实可以不要这个的

    def _get_triangle_mask(self, max_seq_len):
        tensor = torch.ones(max_seq_len, max_seq_len)
        return torch.tril(tensor).byte()


class BiLSTMEncoder(nn.Module):
    def __init__(self, embed, num_layers=3, hidden_size=400, dropout=0.3):
        super().__init__()
        self.embed = embed
        self.lstm = LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_size // 2, bidirectional=True,
                         batch_first=True, dropout=dropout, num_layers=num_layers)

    def forward(self, words, seq_len):
        words = self.embed(words)
        words, hx = self.lstm(words, seq_len)

        return words, hx


class LSTMPast(Past):
    def __init__(self, encode_outputs=None, encode_mask=None, decode_states=None, hx=None):
        """

        :param torch.Tensor encode_outputs: batch_size x max_len x input_size
        :param torch.Tensor encode_mask: batch_size x max_len, 与encode_outputs一样大，用以辅助decode的时候attention到正确的
            词。为1的地方有词
        :param torch.Tensor decode_states: batch_size x decode_len x hidden_size, Decoder中LSTM的输出结果
        :param tuple hx: 包含LSTM所需要的h与c，h: num_layer x batch_size x hidden_size, c: num_layer x batch_size x hidden_size
        """
        super().__init__()
        self._encode_outputs = encode_outputs
        if encode_mask is None:
            if encode_outputs is not None:
                self._encode_mask = encode_outputs.new_ones(encode_outputs.size(0), encode_outputs.size(1)).eq(1)
            else:
                self._encode_mask = None
        else:
            self._encode_mask = encode_mask
        self._decode_states = decode_states
        self._hx = hx  # 包含了hidden和cell
        self._attn_states = None  # 当LSTM使用了Attention时会用到

    def num_samples(self):
        for tensor in (self.encode_outputs, self.decode_states, self.hx):
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    return tensor.size(0)
                else:
                    return tensor[0].size(0)
        return None

    @property
    def hx(self):
        return self._hx

    @hx.setter
    def hx(self, hx):
        self._hx = hx

    @property
    def encode_outputs(self):
        return self._encode_outputs

    @encode_outputs.setter
    def encode_outputs(self, value):
        self._encode_outputs = value

    @property
    def encode_mask(self):
        return self._encode_mask

    @encode_mask.setter
    def encode_mask(self, value):
        self._encode_mask = value

    @property
    def decode_states(self):
        return self._decode_states

    @decode_states.setter
    def decode_states(self, value):
        self._decode_states = value

    @property
    def attn_states(self):
        """
        表示LSTMDecoder中attention模块的结果，正常情况下不需要手动设置
        :return:
        """
        return self._attn_states

    @attn_states.setter
    def attn_states(self, value):
        self._attn_states = value


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


class LSTMDecoder(Decoder):
    def __init__(self, embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray], num_layers, input_size,
                 hidden_size=None, dropout=0,
                 output_embed: Union[Tuple[int, int], int, nn.Module, torch.Tensor, np.ndarray] = None,
                 bind_input_output_embed=False,
                 attention=True):
        """
        # embed假设是TokenEmbedding, 则没有对应关系（因为可能一个token会对应多个word）？vocab出来的结果是不对的

        :param embed: 输入的embedding
        :param int num_layers: 使用多少层LSTM
        :param int input_size: 输入被encode后的维度
        :param int hidden_size: LSTM中的隐藏层维度
        :param float dropout: 多层LSTM的dropout
        :param int output_embed: 输出的词表如何初始化，如果bind_input_output_embed为True，则改值无效
        :param bool bind_input_output_embed: 是否将输入输出的embedding权重使用同一个
        :param bool attention: 是否使用attention对encode之后的内容进行计算
        """

        super().__init__()
        self.token_embed = get_embeddings(embed)
        if hidden_size is None:
            hidden_size = input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        if num_layers == 1:
            self.lstm = nn.LSTM(self.token_embed.embedding_dim + hidden_size, hidden_size, num_layers=num_layers,
                                bidirectional=False, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.token_embed.embedding_dim + hidden_size, hidden_size, num_layers=num_layers,
                                bidirectional=False, batch_first=True, dropout=dropout)
        if input_size != hidden_size:
            self.encode_hidden_proj = nn.Linear(input_size, hidden_size)
            self.encode_cell_proj = nn.Linear(input_size, hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout)

        if isinstance(output_embed, int):
            output_embed = (output_embed, hidden_size)
            output_embed = get_embeddings(output_embed)
        elif output_embed is not None:
            assert not bind_input_output_embed, "When `output_embed` is not None, `bind_input_output_embed` must " \
                                                "be False."
            if isinstance(output_embed, StaticEmbedding):
                for i in self.token_embed.words_to_words:
                    assert i == self.token_embed.words_to_words[i], "The index does not match."
                output_embed = self.token_embed.embedding.weight
            else:
                output_embed = get_embeddings(output_embed)
        else:
            if not bind_input_output_embed:
                raise RuntimeError("You have to specify output embedding.")

        if bind_input_output_embed:
            assert output_embed is None, "When `bind_input_output_embed=True`, `output_embed` must be None"
            if isinstance(self.token_embed, StaticEmbedding):
                for i in self.token_embed.words_to_words:
                    assert i == self.token_embed.words_to_words[i], "The index does not match."
            self.output_embed = nn.Parameter(self.token_embed.weight.transpose(0, 1))
            self.output_hidden_size = self.token_embed.embedding_dim
        else:
            if isinstance(output_embed, nn.Embedding):
                self.output_embed = nn.Parameter(output_embed.weight.transpose(0, 1))
            else:
                self.output_embed = output_embed.transpose(0, 1)
            self.output_hidden_size = self.output_embed.size(0)

        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, self.output_hidden_size))
        self.num_layers = num_layers

        if attention:
            self.attention_layer = AttentionLayer(hidden_size, input_size, hidden_size, bias=False)
        else:
            self.attention_layer = None

    def _init_hx(self, past, tokens):
        batch_size = tokens.size(0)
        if past.hx is None:
            zeros = tokens.new_zeros((self.num_layers, batch_size, self.hidden_size)).float()
            past.hx = (zeros, zeros)
        else:
            assert past.hx[0].size(-1) == self.input_size
            if self.attention_layer is not None:
                if past.attn_states is None:
                    past.attn_states = past.hx[0].new_zeros(batch_size, self.hidden_size)
                else:
                    assert past.attn_states.size(-1) == self.hidden_size, "The attention states dimension mismatch."
            if self.hidden_size != past.hx[0].size(-1):
                hidden, cell = past.hx
                hidden = self.encode_hidden_proj(hidden)
                cell = self.encode_cell_proj(cell)
                past.hx = (hidden, cell)
        return past

    def forward(self, tokens, past=None, return_attention=False):
        """

        :param torch.LongTensor, tokens: batch_size x decode_len, 应该输入整个句子
        :param LSTMPast past: 应该包含了encode的输出
        :param bool return_attention: 是否返回各处attention的值
        :return:
        """
        batch_size, decode_len = tokens.size()
        tokens = self.token_embed(tokens)  # b x decode_len x embed_size

        past = self._init_hx(past, tokens)

        tokens = self.dropout_layer(tokens)

        decode_states = tokens.new_zeros((batch_size, decode_len, self.hidden_size))
        if self.attention_layer is not None:
            attn_scores = tokens.new_zeros((tokens.size(0), tokens.size(1), past.encode_outputs.size(1)))
        if self.attention_layer is not None:
            input_feed = past.attn_states
        else:
            input_feed = past.hx[0][-1]
        for i in range(tokens.size(1)):
            input = torch.cat([tokens[:, i:i + 1], input_feed.unsqueeze(1)], dim=2)  # batch_size x 1 x h'
            # bsz x 1 x hidden_size, (n_layer x bsz x hidden_size, n_layer x bsz x hidden_size)
            _, (hidden, cell) = self.lstm(input, hx=past.hx)
            past.hx = (hidden, cell)
            if self.attention_layer is not None:
                input_feed, attn_score = self.attention_layer(hidden[-1], past.encode_outputs, past.encode_mask)
                attn_scores[:, i] = attn_score
                past.attn_states = input_feed
            else:
                input_feed = hidden[-1]
            decode_states[:, i] = input_feed

        decode_states = self.dropout_layer(decode_states)

        outputs = self.ffn(decode_states)  # batch_size x decode_len x output_hidden_size

        feats = torch.matmul(outputs, self.output_embed)  # bsz x decode_len x vocab_size
        if return_attention:
            return feats, attn_scores
        else:
            return feats

    @torch.no_grad()
    def decode_one(self, tokens, past) -> Tuple[torch.Tensor, Past]:
        """
        给定上一个位置的输出，决定当前位置的输出。
        :param torch.LongTensor tokens: batch_size x seq_len
        :param LSTMPast past:
        :return:
        """
        # past = self._init_hx(past, tokens)
        tokens = tokens[:, -1:]
        feats = self.forward(tokens, past, return_attention=False)
        return feats.squeeze(1), past

    def reorder_past(self, indices: torch.LongTensor, past: LSTMPast) -> LSTMPast:
        """
        将LSTMPast中的状态重置一下

        :param torch.LongTensor indices: 在batch维度的index
        :param LSTMPast past: 保存的过去的状态
        :return:
        """
        encode_outputs = past.encode_outputs.index_select(index=indices, dim=0)
        encoder_mask = past.encode_mask.index_select(index=indices, dim=0)
        hx = (past.hx[0].index_select(index=indices, dim=1),
              past.hx[1].index_select(index=indices, dim=1))
        if past.attn_states is not None:
            past.attn_states = past.attn_states.index_select(index=indices, dim=0)
        past.encode_mask = encoder_mask
        past.encode_outputs = encode_outputs
        past.hx = hx
        return past
