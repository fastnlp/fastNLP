r"""
主要包含组成Sequence-to-Sequence的model

"""

import torch
from torch import nn
import torch.nn.functional as F

from fastNLP import seq_len_to_mask
from ...embeddings.torch.utils import get_embeddings
from ...embeddings.torch.utils import get_sinusoid_encoding_table
from ...modules.torch.decoder.seq2seq_decoder import Seq2SeqDecoder, TransformerSeq2SeqDecoder, LSTMSeq2SeqDecoder
from ...modules.torch.encoder.seq2seq_encoder import Seq2SeqEncoder, TransformerSeq2SeqEncoder, LSTMSeq2SeqEncoder


__all__ = ['Seq2SeqModel', 'TransformerSeq2SeqModel', 'LSTMSeq2SeqModel']


class Seq2SeqModel(nn.Module):
    """
    可以用于在 :class:`~fastNLP.core.controllers.Trainer` 中训练的 **Seq2Seq模型** 。正常情况下，继承了该函数之后，只需要
    实现 classmethod ``build_model`` 即可。如果需要使用该模型进行生成，需要把该模型输入到 :class:`~fastNLP.models.torch.SequenceGeneratorModel`
    中。在本模型中， :meth:`forward` 会把 encoder 后的结果传入到 decoder 中，并将 decoder 的输出 output 出来。

    :param encoder: :class:`~fastNLP.modules.torch.encoder.Seq2SeqEncoder` 对象，需要实现对应的 :meth:`forward` 函数，接受两个参数，第一个为
        ``[batch_size, max_len]`` 的 source tokens, 第二个为 ``[batch_size,]`` 的 source 的长度；需要返回两个 tensor： 
        
            - ``encoder_outputs`` : ``[batch_size, max_len, hidden_size]``
            - ``encoder_mask`` :  ``[batch_size, max_len]``，为 **0** 的地方为 pad。
        如果encoder的输出或者输入有变化，可以重载本模型的 :meth:`prepare_state` 函数或者 :meth:`forward` 函数。
    :param decoder: :class:`~fastNLP.modules.torch.decoder.Seq2SeqEncoder` 对象，需要实现 :meth:`init_state` 函数，需要接受两个参数，分别为
        上述的 ``encoder_outputs`` 和 ``encoder_mask``。若decoder需要更多输入，请重载当前模型的 :meth:`prepare_state` 或 :meth:`forward` 函数。
    """
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens: "torch.LongTensor", tgt_tokens: "torch.LongTensor",
                src_seq_len: "torch.LongTensor"=None, tgt_seq_len: "torch.LongTensor"=None):
        """
        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param tgt_tokens: target 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :param tgt_seq_len: target的长度，形状为 ``[batch_size,]``
        :return: 字典 ``{'pred': torch.Tensor}``, 其中 ``pred`` 的形状为 ``[batch_size, max_len, vocab_size]``
        """
        state = self.prepare_state(src_tokens, src_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

    def train_step(self, src_tokens: "torch.LongTensor", tgt_tokens: "torch.LongTensor",
                    src_seq_len: "torch.LongTensor"=None, tgt_seq_len: "torch.LongTensor"=None):
        """
        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param tgt_tokens: target 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :param tgt_seq_len: target的长度，形状为 ``[batch_size,]``
        :return: 字典 ``{'loss': torch.Tensor}``
        """
        res = self(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)
        pred = res['pred']
        if tgt_seq_len is not None:
            mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1))
            tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)
        loss = F.cross_entropy(pred[:, :-1].transpose(1, 2), tgt_tokens[:, 1:])
        return {'loss': loss}

    def prepare_state(self, src_tokens: "torch.LongTensor", src_seq_len: "torch.LongTensor"=None):
        """
        调用 encoder 获取 state，会把 encoder 的 ``encoder_output``, ``encoder_mask`` 直接传入到 :meth:`decoder.init_state` 中初始化一个 state

        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :return: decode 初始化的 state
        """
        encoder_output, encoder_mask = self.encoder(src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        return state

    @classmethod
    def build_model(cls, *args, **kwargs):
        """
        需要实现本方法来进行 :class:`Seq2SeqModel` 的初始化

        :return:
        """
        raise NotImplementedError("A `Seq2SeqModel` must implement its own classmethod `build_model()`.")


class TransformerSeq2SeqModel(Seq2SeqModel):
    """
    Encoder 为 :class:`~fastNLP.modules.torch.encoder.TransformerSeq2SeqEncoder` ，decoder 为 
    :class:`~fastNLP.modules.torch.decoder.TransformerSeq2SeqDecoder` 的 :class:`Seq2SeqModel` ，
    通过 :meth:`build_model` 方法初始化。
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_embed, tgt_embed=None,
                    pos_embed: str='sin', max_position: int=1024, num_layers: int=6, d_model: int=512,
                    n_head: int=8, dim_ff: int=2048, dropout: float=0.1,
                    bind_encoder_decoder_embed: bool=False,
                    bind_decoder_input_output_embed: bool=True):
        """
        初始化一个 :class:`TransformerSeq2SeqModel` 。

        :param src_embed: source 的 embedding，支持以下几种输入类型：

                - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
                - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
                - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
                - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
        :param tgt_embed: target 的 embedding，如果 ``bind_encoder_decoder_embed``
            为 ``True`` ，则不要输入该值。支持以下几种输入类型：

                - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
                - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
                - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
                - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
        :param pos_embed: 支持 ``['sin', 'learned']`` 两种
        :param max_position: 最大支持长度
        :param num_layers: ``encoder`` 和 ``decoder`` 的层数
        :param d_model: ``encoder`` 和 ``decoder`` 输入输出的大小
        :param n_head: ``encoder`` 和 ``decoder`` 的 head 的数量
        :param dim_ff: ``encoder`` 和 ``decoder`` 中 FFN 中间映射的维度
        :param dropout: Attention 和 FFN dropout的大小
        :param bind_encoder_decoder_embed: 是否对 ``encoder`` 和 ``decoder`` 使用相同的 embedding
        :param bind_decoder_input_output_embed: ``decoder`` 的输出 embedding 是否与其输入 embedding 是一样的权重
        :return: :class:`TransformerSeq2SeqModel` 模型
        """
        if bind_encoder_decoder_embed and tgt_embed is not None:
            raise RuntimeError("If you set `bind_encoder_decoder_embed=True`, please do not provide `tgt_embed`.")

        src_embed = get_embeddings(src_embed)

        if bind_encoder_decoder_embed:
            tgt_embed = src_embed
        else:
            assert tgt_embed is not None, "You need to pass `tgt_embed` when `bind_encoder_decoder_embed=False`"
            tgt_embed = get_embeddings(tgt_embed)

        if pos_embed == 'sin':
            encoder_pos_embed = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(max_position + 1, src_embed.embedding_dim, padding_idx=0),
                freeze=True)  # 这里规定0是padding
            deocder_pos_embed = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(max_position + 1, tgt_embed.embedding_dim, padding_idx=0),
                freeze=True)  # 这里规定0是padding
        elif pos_embed == 'learned':
            encoder_pos_embed = get_embeddings((max_position + 1, src_embed.embedding_dim), padding_idx=0)
            deocder_pos_embed = get_embeddings((max_position + 1, src_embed.embedding_dim), padding_idx=1)
        else:
            raise ValueError("pos_embed only supports sin or learned.")

        encoder = TransformerSeq2SeqEncoder(embed=src_embed, pos_embed=encoder_pos_embed,
                                            num_layers=num_layers, d_model=d_model, n_head=n_head, dim_ff=dim_ff,
                                            dropout=dropout)
        decoder = TransformerSeq2SeqDecoder(embed=tgt_embed, pos_embed=deocder_pos_embed,
                                            d_model=d_model, num_layers=num_layers, n_head=n_head, dim_ff=dim_ff,
                                            dropout=dropout,
                                            bind_decoder_input_output_embed=bind_decoder_input_output_embed)

        return cls(encoder, decoder)


class LSTMSeq2SeqModel(Seq2SeqModel):
    """
    使用 :class:`~fastNLP.modules.torch.encoder.LSTMSeq2SeqEncoder` 和 :class:`~fastNLP.modules.torch.decoder.LSTMSeq2SeqDecoder` 的
    :class:`Seq2SeqModel`，通过 :meth:`build_model` 方法初始化。

    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_embed, tgt_embed=None,
                    num_layers: int = 3, hidden_size: int = 400, dropout: float = 0.3, bidirectional: bool=True,
                    attention: bool=True, bind_encoder_decoder_embed: bool=False,
                    bind_decoder_input_output_embed: bool=True):
        """

        :param src_embed: source 的 embedding，支持以下几种输入类型：

                - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
                - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
                - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
                - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
        :param tgt_embed: target 的 embedding，如果 ``bind_encoder_decoder_embed``
            为 ``True`` ，则不要输入该值，支持以下几种输入类型：

                - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
                - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
                - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
                - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
        :param num_layers: ``encoder`` 和 ``decoder`` 的层数
        :param hidden_size: ``encoder`` 和 ``decoder`` 的隐藏层大小
        :param dropout: 每层之间的 Dropout 的大小
        :param bidirectional: ``encoder`` 是否使用 **双向LSTM**
        :param attention: 是否在 ``decoder`` 中使用 attention 来添加 ``encoder`` 在所有时刻的状态
        :param bind_encoder_decoder_embed: 是否对 ``encoder`` 和 ``decoder`` 使用相同的 embedding
        :param bind_decoder_input_output_embed: ``decoder`` 的输出 embedding 是否与其输入 embedding 是一样的权重
        :return: :class:`LSTMSeq2SeqModel` 模型
        """
        if bind_encoder_decoder_embed and tgt_embed is not None:
            raise RuntimeError("If you set `bind_encoder_decoder_embed=True`, please do not provide `tgt_embed`.")

        src_embed = get_embeddings(src_embed)

        if bind_encoder_decoder_embed:
            tgt_embed = src_embed
        else:
            assert tgt_embed is not None, "You need to pass `tgt_embed` when `bind_encoder_decoder_embed=False`"
            tgt_embed = get_embeddings(tgt_embed)

        encoder = LSTMSeq2SeqEncoder(embed=src_embed, num_layers = num_layers,
                 hidden_size = hidden_size, dropout = dropout, bidirectional=bidirectional)
        decoder = LSTMSeq2SeqDecoder(embed=tgt_embed, num_layers = num_layers, hidden_size = hidden_size,
                 dropout = dropout, bind_decoder_input_output_embed = bind_decoder_input_output_embed,
                                     attention=attention)
        return cls(encoder, decoder)
