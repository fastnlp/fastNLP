r"""
主要包含组成Sequence-to-Sequence的model

"""

import torch
from torch import nn

from ..embeddings import get_embeddings
from ..embeddings.utils import get_sinusoid_encoding_table
from ..modules.decoder.seq2seq_decoder import Seq2SeqDecoder, TransformerSeq2SeqDecoder, LSTMSeq2SeqDecoder
from ..modules.encoder.seq2seq_encoder import Seq2SeqEncoder, TransformerSeq2SeqEncoder, LSTMSeq2SeqEncoder


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        """
        可以用于在Trainer中训练的Seq2Seq模型。正常情况下，继承了该函数之后，只需要实现classmethod build_model即可。

        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

    def prepare_state(self, src_tokens, src_seq_len=None):
        """
        调用encoder获取state，会把encoder的encoder_output, encoder_mask直接传入到decoder.init_state中初始化一个state

        :param src_tokens:
        :param src_seq_len:
        :return:
        """
        encoder_output, encoder_mask = self.encoder(src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        return state

    @classmethod
    def build_model(cls, *args, **kwargs):
        """
        需要实现本方法来进行Seq2SeqModel的初始化

        :return:
        """
        raise NotImplemented


class TransformerSeq2SeqModel(Seq2SeqModel):
    """
    Encoder为TransformerSeq2SeqEncoder, decoder为TransformerSeq2SeqDecoder，通过build_model方法初始化

    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_embed, tgt_embed=None,
                    pos_embed='sin', max_position=1024, num_layers=6, d_model=512, n_head=8, dim_ff=2048, dropout=0.1,
                    bind_encoder_decoder_embed=False,
                    bind_decoder_input_output_embed=True):
        """
        初始化一个TransformerSeq2SeqModel

        :param nn.Module, StaticEmbedding, Tuple[int, int] src_embed: source的embedding
        :param nn.Module, StaticEmbedding, Tuple[int, int] tgt_embed: target的embedding，如果bind_encoder_decoder_embed为
            True，则不要输入该值
        :param str pos_embed: 支持sin, learned两种
        :param int max_position: 最大支持长度
        :param int num_layers: encoder和decoder的层数
        :param int d_model: encoder和decoder输入输出的大小
        :param int n_head: encoder和decoder的head的数量
        :param int dim_ff: encoder和decoder中FFN中间映射的维度
        :param float dropout: Attention和FFN dropout的大小
        :param bool bind_encoder_decoder_embed: 是否对encoder和decoder使用相同的embedding
        :param bool bind_decoder_input_output_embed: decoder的输出embedding是否与其输入embedding是一样的权重
        :return: TransformerSeq2SeqModel
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
    使用LSTMSeq2SeqEncoder和LSTMSeq2SeqDecoder的model

    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_embed, tgt_embed=None,
                    num_layers = 3, hidden_size = 400, dropout = 0.3, bidirectional=True,
                    attention=True, bind_encoder_decoder_embed=False,
                    bind_decoder_input_output_embed=True):
        """

        :param nn.Module, StaticEmbedding, Tuple[int, int] src_embed: source的embedding
        :param nn.Module, StaticEmbedding, Tuple[int, int] tgt_embed: target的embedding，如果bind_encoder_decoder_embed为
            True，则不要输入该值
        :param int num_layers: Encoder和Decoder的层数
        :param int hidden_size: encoder和decoder的隐藏层大小
        :param float dropout: 每层之间的Dropout的大小
        :param bool bidirectional: encoder是否使用双向LSTM
        :param bool attention: decoder是否使用attention attend encoder在所有时刻的状态
        :param bool bind_encoder_decoder_embed: 是否对encoder和decoder使用相同的embedding
        :param bool bind_decoder_input_output_embed: decoder的输出embedding是否与其输入embedding是一样的权重
        :return: LSTMSeq2SeqModel
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
