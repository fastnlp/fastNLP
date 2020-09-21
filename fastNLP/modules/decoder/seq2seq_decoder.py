r"""undocumented"""
from typing import Union, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
from ..attention import AttentionLayer, MultiHeadAttention
from ...embeddings import StaticEmbedding
from ...embeddings.utils import get_embeddings
from .seq2seq_state import State, LSTMState, TransformerState


class Seq2SeqDecoder(nn.Module):
    """
    Sequence-to-Sequence Decoder的基类。一定需要实现forward函数，剩下的函数根据需要实现。每个Seq2SeqDecoder都应该有相应的State对象
        用来承载该Decoder所需要的Encoder输出、Decoder需要记录的历史信息(例如LSTM的hidden信息)。

    """
    def __init__(self):
        super().__init__()

    def forward(self, tokens, state, **kwargs):
        """

        :param torch.LongTensor tokens: bsz x max_len
        :param State state: state包含了encoder的输出以及decode之前的内容
        :return: 返回值可以为bsz x max_len x vocab_size的Tensor，也可以是一个list，但是第一个元素必须是词的预测分布
        """
        raise NotImplemented

    def reorder_states(self, indices, states):
        """
        根据indices重新排列states中的状态，在beam search进行生成时，会用到该函数。

        :param torch.LongTensor indices:
        :param State states:
        :return:
        """
        assert isinstance(states, State), f"`states` should be of type State instead of {type(states)}"
        states.reorder_state(indices)

    def init_state(self, encoder_output, encoder_mask):
        """
        初始化一个state对象，用来记录了encoder的输出以及decode已经完成的部分。

        :param Union[torch.Tensor, list, tuple] encoder_output: 如果不为None，内部元素需要为torch.Tensor, 默认其中第一维是batch
            维度
        :param Union[torch.Tensor, list, tuple] encoder_mask: 如果部位None，内部元素需要torch.Tensor, 默认其中第一维是batch
            维度
        :param kwargs:
        :return: State, 返回一个State对象，记录了encoder的输出
        """
        state = State(encoder_output, encoder_mask)
        return state

    def decode(self, tokens, state):
        """
        根据states中的内容，以及tokens中的内容进行之后的生成。

        :param torch.LongTensor tokens: bsz x max_len, 上一个时刻的token输出。
        :param State state: 记录了encoder输出与decoder过去状态
        :return: torch.FloatTensor: bsz x vocab_size, 输出的是下一个时刻的分布
        """
        outputs = self(state=state, tokens=tokens)
        if isinstance(outputs, torch.Tensor):
            return outputs[:, -1]
        else:
            raise RuntimeError("Unrecognized output from the `forward()` function. Please override the `decode()` function.")


class TiedEmbedding(nn.Module):
    """
    用于将weight和原始weight绑定

    """
    def __init__(self, weight):
        super().__init__()
        self.weight = weight  # vocab_size x embed_size

    def forward(self, x):
        """

        :param torch.FloatTensor x: bsz x * x embed_size
        :return: torch.FloatTensor bsz x * x vocab_size
        """
        return torch.matmul(x, self.weight.t())


def get_binded_decoder_output_embed(embed):
    """
    给定一个embedding，输出对应的绑定的embedding，输出对象为TiedEmbedding

    :param embed:
    :return:
    """
    if isinstance(embed, StaticEmbedding):
        for idx, map2idx in enumerate(embed.words_to_words):
            assert idx == map2idx, "Invalid StaticEmbedding for Decoder, please check:(1) whether the vocabulary " \
                                   "include `no_create_entry=True` word; (2) StaticEmbedding should  not initialize with " \
                                   "`lower=True` or `min_freq!=1`."
    elif not isinstance(embed, nn.Embedding):
        raise TypeError("Only nn.Embedding or StaticEmbedding is allowed for binding.")

    return TiedEmbedding(embed.weight)


class LSTMSeq2SeqDecoder(Seq2SeqDecoder):
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], num_layers = 3, hidden_size = 300,
                 dropout = 0.3, bind_decoder_input_output_embed = True, attention=True):
        """
        LSTM的Decoder

        :param nn.Module,tuple embed: decoder输入的embedding.
        :param int num_layers: 多少层LSTM
        :param int hidden_size: 隐藏层大小, 该值也被认为是encoder的输出维度大小
        :param dropout: Dropout的大小
        :param bool bind_decoder_input_output_embed: 是否将输出层和输入层的词向量绑定在一起（即为同一个），若embed为StaticEmbedding，
            则StaticEmbedding的vocab不能包含no_create_entry的token，同时StaticEmbedding初始化时lower为False, min_freq=1.
        :param bool attention: 是否使用attention
        """
        super().__init__()
        self.embed = get_embeddings(init_embed=embed)
        self.embed_dim = embed.embedding_dim

        if bind_decoder_input_output_embed:
            self.output_layer = get_binded_decoder_output_embed(self.embed)
        else:  # 不需要bind
            self.output_embed = get_embeddings((self.embed.num_embeddings, self.embed.embedding_dim))
            self.output_layer = TiedEmbedding(self.output_embed.weight)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.embed_dim + hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=False, dropout=dropout if num_layers>1 else 0)

        self.attention_layer = AttentionLayer(hidden_size, hidden_size, hidden_size) if attention else None
        self.output_proj = nn.Linear(hidden_size, self.embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, tokens, state, return_attention=False):
        """

        :param torch.LongTensor tokens: batch x max_len
        :param LSTMState state: 保存encoder输出和decode状态的State对象
        :param bool return_attention: 是否返回attention的的score
        :return: bsz x max_len x vocab_size; 如果return_attention=True, 还会返回bsz x max_len x encode_length
        """
        src_output = state.encoder_output
        encoder_mask = state.encoder_mask

        assert tokens.size(1)>state.decode_length, "The state does not match the tokens."
        tokens = tokens[:, state.decode_length:]
        x = self.embed(tokens)

        attn_weights = [] if self.attention_layer is not None else None  # 保存attention weight, batch,tgt_seq,src_seq
        input_feed = state.input_feed
        decoder_out = []

        cur_hidden = state.hidden
        cur_cell = state.cell

        # 开始计算
        for i in range(tokens.size(1)):
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

            state.input_feed = input_feed  # batch, hidden
            state.hidden = cur_hidden
            state.cell = cur_cell
            state.decode_length += 1
            decoder_out.append(input_feed)

        decoder_out = torch.stack(decoder_out, dim=1)  # batch,seq_len,hidden
        decoder_out = self.dropout_layer(decoder_out)
        if attn_weights is not None:
            attn_weights = torch.cat(attn_weights, dim=1)  # batch, tgt_len, src_len

        decoder_out = self.output_proj(decoder_out)
        feats = self.output_layer(decoder_out)

        if return_attention:
            return feats, attn_weights
        return feats

    def init_state(self, encoder_output, encoder_mask) -> LSTMState:
        """

        :param encoder_output: 输入可以有两种情况(1) 输入为一个tuple，包含三个内容(encoder_output, (hidden, cell))，其中encoder_output:
            bsz x max_len x hidden_size, hidden: bsz x hidden_size, cell:bsz x hidden_size,一般使用LSTMEncoder的最后一层的
            hidden state和cell state来赋值这两个值
            (2) 只有encoder_output: bsz x max_len x hidden_size, 这种情况下hidden和cell使用0初始化
        :param torch.ByteTensor encoder_mask: bsz x max_len, 为0的位置是padding, 用来指示source中哪些不需要attend
        :return:
        """
        if not isinstance(encoder_output, torch.Tensor):
            encoder_output, (hidden, cell) = encoder_output
        else:
            hidden = cell = None
        assert encoder_output.ndim==3
        assert encoder_mask.size()==encoder_output.size()[:2]
        assert encoder_output.size(-1)==self.hidden_size, "The dimension of encoder outputs should be the same with " \
                                                          "the hidden_size."

        t = [hidden, cell]
        for idx in range(2):
            v = t[idx]
            if v is None:
                v = encoder_output.new_zeros(self.num_layers, encoder_output.size(0), self.hidden_size)
            else:
                assert v.dim()==2
                assert v.size(-1)==self.hidden_size
                v = v[None].repeat(self.num_layers, 1, 1)  # num_layers x bsz x hidden_size
            t[idx] = v

        state = LSTMState(encoder_output, encoder_mask, t[0], t[1])

        return state


class TransformerSeq2SeqDecoderLayer(nn.Module):
    def __init__(self, d_model = 512, n_head = 8, dim_ff = 2048, dropout = 0.1, layer_idx = None):
        """

        :param int d_model: 输入、输出的维度
        :param int n_head: 多少个head，需要能被d_model整除
        :param int dim_ff:
        :param float dropout:
        :param int layer_idx: layer的编号
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_idx = layer_idx  # 记录layer的层索引，以方便获取state的信息

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout, layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.encoder_attn = MultiHeadAttention(d_model, n_head, dropout, layer_idx)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x, encoder_output, encoder_mask=None, self_attn_mask=None, state=None):
        """

        :param x: (batch, seq_len, dim), decoder端的输入
        :param encoder_output: (batch,src_seq_len,dim), encoder的输出
        :param encoder_mask: batch,src_seq_len, 为1的地方需要attend
        :param self_attn_mask: seq_len, seq_len，下三角的mask矩阵，只在训练时传入
        :param TransformerState state: 只在inference阶段传入
        :return:
        """

        # self attention part
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              attn_mask=self_attn_mask,
                              state=state)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # encoder attention part
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn_weight = self.encoder_attn(query=x,
                                           key=encoder_output,
                                           value=encoder_output,
                                           key_mask=encoder_mask,
                                           state=state)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weight


class TransformerSeq2SeqDecoder(Seq2SeqDecoder):
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], pos_embed: nn.Module = None,
                 d_model = 512, num_layers=6, n_head = 8, dim_ff = 2048, dropout = 0.1,
                 bind_decoder_input_output_embed = True):
        """

        :param embed: 输入token的embedding
        :param nn.Module pos_embed: 位置embedding
        :param int d_model: 输出、输出的大小
        :param int num_layers: 多少层
        :param int n_head: 多少个head
        :param int dim_ff: FFN 的中间大小
        :param float dropout: Self-Attention和FFN中的dropout的大小
        :param bool bind_decoder_input_output_embed: 是否将输出层和输入层的词向量绑定在一起（即为同一个），若embed为StaticEmbedding，
            则StaticEmbedding的vocab不能包含no_create_entry的token，同时StaticEmbedding初始化时lower为False, min_freq=1.
        """
        super().__init__()

        self.embed = get_embeddings(embed)
        self.pos_embed = pos_embed

        if bind_decoder_input_output_embed:
            self.output_layer = get_binded_decoder_output_embed(self.embed)
        else:  # 不需要bind
            self.output_embed = get_embeddings((self.embed.num_embeddings, self.embed.embedding_dim))
            self.output_layer = TiedEmbedding(self.output_embed.weight)

        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_fc = nn.Linear(self.embed.embedding_dim, d_model)
        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqDecoderLayer(d_model, n_head, dim_ff, dropout, layer_idx)
                                           for layer_idx in range(num_layers)])

        self.embed_scale = math.sqrt(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_fc = nn.Linear(self.d_model, self.embed.embedding_dim)

    def forward(self, tokens, state, return_attention=False):
        """

        :param torch.LongTensor tokens: batch x tgt_len，decode的词
        :param TransformerState state: 用于记录encoder的输出以及decode状态的对象，可以通过init_state()获取
        :param bool return_attention: 是否返回对encoder结果的attention score
        :return: bsz x max_len x vocab_size; 如果return_attention=True, 还会返回bsz x max_len x encode_length
        """

        encoder_output = state.encoder_output
        encoder_mask = state.encoder_mask

        assert state.decode_length<tokens.size(1), "The decoded tokens in State should be less than tokens."
        tokens = tokens[:, state.decode_length:]
        device = tokens.device

        x = self.embed_scale * self.embed(tokens)
        if self.pos_embed is not None:
            position = torch.arange(state.decode_length, state.decode_length+tokens.size(1)).long().to(device)[None]
            x += self.pos_embed(position)
        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        batch_size, max_tgt_len = tokens.size()

        if max_tgt_len>1:
            triangle_mask = self._get_triangle_mask(tokens)
        else:
            triangle_mask = None

        for layer in self.layer_stacks:
            x, attn_weight = layer(x=x,
                                   encoder_output=encoder_output,
                                   encoder_mask=encoder_mask,
                                   self_attn_mask=triangle_mask,
                                   state=state
                                   )

        x = self.layer_norm(x)  # batch, tgt_len, dim
        x = self.output_fc(x)
        feats = self.output_layer(x)

        if return_attention:
            return feats, attn_weight
        return feats

    def init_state(self, encoder_output, encoder_mask):
        """
        初始化一个TransformerState用于forward

        :param torch.FloatTensor encoder_output: bsz x max_len x d_model, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x max_len, 为1的位置需要attend。
        :return: TransformerState
        """
        if isinstance(encoder_output, torch.Tensor):
            encoder_output = encoder_output
        elif isinstance(encoder_output, (list, tuple)):
            encoder_output = encoder_output[0]  # 防止是LSTMEncoder的输出结果
        else:
            raise TypeError("Unsupported `encoder_output` for TransformerSeq2SeqDecoder")
        state = TransformerState(encoder_output, encoder_mask, num_decoder_layer=self.num_layers)
        return state

    @staticmethod
    def _get_triangle_mask(tokens):
        tensor = tokens.new_ones(tokens.size(1), tokens.size(1))
        return torch.tril(tensor).byte()


