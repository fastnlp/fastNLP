from typing import Union, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
from ..attention import AttentionLayer, MultiHeadAttention
from ....embeddings.torch.utils import get_embeddings
from ....embeddings.torch.static_embedding import StaticEmbedding
from .seq2seq_state import State, LSTMState, TransformerState


__all__ = ['Seq2SeqDecoder', 'TransformerSeq2SeqDecoder', 'LSTMSeq2SeqDecoder']


class Seq2SeqDecoder(nn.Module):
    """
    **Sequence-to-Sequence Decoder** 的基类。一定需要实现 :meth:`forward` 和 :meth:`decode` 函数，剩下的函数根据需要实现。每个 ``Seq2SeqDecoder`` 都应该有相应的
    :class:`~fastNLP.modules.torch.decoder.State` 对象用来承载该 ``Decoder`` 所需要的 ``Encoder`` 输出、``Decoder`` 需要记录的历史信（例如 :class:`~fastNLP.modules.torch.encoder.LSTM`
    的 hidden 信息）。
    """
    def __init__(self):
        super().__init__()

    def forward(self, tokens: "torch.LongTensor", state: State, **kwargs):
        """

        :param tokens: ``[batch_size, max_len]``
        :param state: ``state`` 包含了 ``encoder`` 的输出以及 ``decode`` 之前的内容
        :return: 返回值可以为 ``[batch_size, max_len, vocab_size]`` 的张量，也可以是一个 :class:`list`，但是第一个元素必须是词的预测分布
        """
        raise NotImplemented

    def reorder_states(self, indices: torch.LongTensor, states):
        """
        根据 ``indices`` 重新排列 ``states`` 中的状态，在 ``beam search`` 进行生成时，会用到该函数。

        :param indices:
        :param states:
        """
        assert isinstance(states, State), f"`states` should be of type State instead of {type(states)}"
        states.reorder_state(indices)

    def init_state(self, encoder_output: Union[torch.Tensor, list, tuple], encoder_mask: Union[torch.Tensor, list, tuple]):
        """
        初始化一个 :class:`~fastNLP.modules.torch.decoder.State` 对象，用来记录 ``encoder`` 的输出以及 ``decode`` 已经完成的部分。

        :param encoder_output: 如果不为 ``None`` ，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 batch_size
            维度
        :param encoder_mask: 如果不为 ``None``，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 batch_size
            维度
        :return: 一个 :class:`~fastNLP.modules.torch.decoder.State` 对象，记录了 ``encoder`` 的输出
        """
        state = State(encoder_output, encoder_mask)
        return state

    def decode(self, tokens: torch.LongTensor, state) -> torch.FloatTensor:
        """
        根据 ``states`` 中的内容，以及 ``tokens`` 中的内容进行之后的生成。

        :param tokens: ``[batch_size, max_len]``，截止到上一个时刻所有的 token 输出。
        :param state: 记录了 ``encoder`` 输出与 ``decoder`` 过去状态
        :return: `下一个时刻的分布，形状为 ``[batch_size, vocab_size]``
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

        :param torch.FloatTensor x: batch_size x * x embed_size
        :return: torch.FloatTensor batch_size x * x vocab_size
        """
        return torch.matmul(x, self.weight.t())


def get_bind_decoder_output_embed(embed):
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
    """
    **LSTM** 的 Decoder

    :param embed: ``decoder`` 输入的 embedding，支持以下几种输入类型：

            - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
            - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
            - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
            - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
    :param num_layers: LSTM 的层数
    :param hidden_size: 隐藏层大小, 该值也被认为是 ``encoder`` 的输出维度大小
    :param dropout: Dropout 的大小
    :param bind_decoder_input_output_embed: ``decoder`` 的输出 embedding 是否与其输入 embedding 是一样的权重（即为同一个），若 ``embed`` 为 
        :class:`~fastNLP.embeddings.StaticEmbedding`，则 ``StaticEmbedding`` 的 ``vocab`` 不能包含 ``no_create_entry`` 的 token ，同时
        ``StaticEmbedding`` 初始化时 ``lower`` 为 ``False``，``min_freq=1``。
    :param attention: 是否使用attention
    """
    def __init__(self, embed: Union[nn.Module, Tuple[int, int]], num_layers: int = 3, hidden_size: int = 300,
                 dropout: float = 0.3, bind_decoder_input_output_embed: bool = True, attention: bool = True):
        super().__init__()
        self.embed = get_embeddings(init_embed=embed)
        self.embed_dim = embed.embedding_dim

        if bind_decoder_input_output_embed:
            self.output_layer = get_bind_decoder_output_embed(self.embed)
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

    def forward(self, tokens: torch.LongTensor, state: LSTMState, return_attention: bool=False):
        """

        :param tokens: ``[batch_size, max_len]``
        :param state: 保存 ``encoder`` 输出和 ``decode`` 状态的 :class:`~fastNLP.modules.torch.decoder.LSTMState` 对象
        :param return_attention: 是否返回 attention 的 score
        :return: 形状为 ``[batch_size, max_len, vocab_size]`` 的结果。如果 ``return_attention=True`` 则返回一个元组，一个元素为结果，第二个结果为
            注意力权重，形状为 ``[batch_size, max_len, encode_length]``
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

    def init_state(self, encoder_output, encoder_mask: torch.ByteTensor) -> LSTMState:
        """

        :param encoder_output: ``encoder`` 的输出，可以有两种情况：
                
                - 输入一个 :class:`tuple`，包含三个内容 ``(encoder_output, (hidden, cell))``，其中 ``encoder_output`` 形状为
                  ``[batch_size, max_len, hidden_size]``， ``hidden`` 形状为 ``[batch_size, hidden_size]``， ``cell`` 形状为
                  ``[batch_size, hidden_size]`` ，一般使用 :class:`~fastNLP.modules.torch.encoder.LSTMSeq2SeqEncoder` 最后一层的
                  ``hidden state`` 和 ``cell state`` 来赋值这两个值。
                - 只有形状为 ``[batch_size, max_len, hidden_size]`` 的 ``encoder_output``, 这种情况下 ``hidden`` 和 ``cell``
                  使用 **0** 初始化。
        :param encoder_mask: ``[batch_size, max_len]]``，为 **0** 的位置是 padding, 用来指示输入中哪些不需要 attend 。
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
    """
    **Transformer** 的 Decoder 层

    :param d_model: 输入、输出的维度
    :param n_head: **多头注意力** head 的数目，需要能被 ``d_model`` 整除
    :param dim_ff:  FFN 中间映射的维度
    :param dropout: Dropout 的大小
    :param layer_idx: layer的编号
    """
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1, layer_idx: int = None):
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

    def forward(self, x, encoder_output, encoder_mask=None, self_attn_mask=None, state: TransformerState=None):
        """

        :param x: ``decoder`` 端的输入，形状为 ``[batch_size, seq_len, dim]`` 
        :param encoder_output: ``encoder`` 的输出，形状为 ``[batch_size, src_seq_len, dim]``
        :param encoder_mask: 掩码，形状为 ``[batch_size, src_seq_len]``，为 **1** 的地方表示需要 attend
        :param self_attn_mask: 下三角的mask矩阵，只在训练时传入。形状为 ``[seq_len, seq_len]``
        :param state: 只在 inference 阶段传入，记录了 ``encoder`` 和 ``decoder`` 的状态
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
    """
    **Transformer** 的 Decoder

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
    :param bind_decoder_input_output_embed: ``decoder`` 的输出 embedding 是否与其输入 embedding 是一样的权重（即为同一个），若 ``embed`` 为 
        :class:`~fastNLP.embeddings.StaticEmbedding`，则 ``StaticEmbedding`` 的 ``vocab`` 不能包含 ``no_create_entry`` 的 token ，同时
        ``StaticEmbedding`` 初始化时 ``lower`` 为 ``False``，``min_freq=1``。
    """
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], pos_embed: nn.Module = None,
                 d_model = 512, num_layers=6, n_head = 8, dim_ff = 2048, dropout = 0.1,
                 bind_decoder_input_output_embed = True):
        super().__init__()

        self.embed = get_embeddings(embed)
        self.pos_embed = pos_embed

        if bind_decoder_input_output_embed:
            self.output_layer = get_bind_decoder_output_embed(self.embed)
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

    def forward(self, tokens: torch.LongTensor, state: TransformerState, return_attention: bool=False):
        """

        :param tokens: 用于解码的词，形状为 ``[batch_size, tgt_len]``
        :param state: 用于记录 ``encoder`` 的输出以及 ``decode`` 状态的对象，可以通过 :meth:`init_state` 获取
        :param return_attention: 是否返回对 ``encoder`` 结果的 attention score
        :return: 形状为 ``[batch_size, max_len, vocab_size]`` 的结果。如果 ``return_attention=True`` 则返回一个元组，一个元素为结果，第二个结果为
            注意力权重，形状为 ``[batch_size, max_len, encode_length]``
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

    def init_state(self, encoder_output: torch.FloatTensor, encoder_mask: torch.ByteTensor) -> TransformerState:
        """
        初始化一个 :class:`~fastNLP.modules.torch.decoder.TransformerState`` 用于 :meth:`forward`

        :param encoder_output: ``encoder`` 的输出，形状为 ``[batch_size, max_len, d_model]``
        :param encoder_mask: ``[batch_size, max_len]]``，为 **0** 的位置是 padding, 用来指示输入中哪些不需要 attend 。
        :return:
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


