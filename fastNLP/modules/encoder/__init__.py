r"""
.. todo::
    doc
"""

__all__ = [
    "ConvolutionCharEncoder",
    "LSTMCharEncoder",

    "ConvMaxpool",

    "LSTM",

    "StarTransformer",

    "TransformerEncoder",

    "VarRNN",
    "VarLSTM",
    "VarGRU",

    "MaxPool",
    "MaxPoolWithMask",
    "KMaxPool",
    "AvgPool",
    "AvgPoolWithMask",

    "MultiHeadAttention",
    "BiAttention",
    "SelfAttention",

    "BertModel",

    "RobertaModel",

    "GPT2Model",

    "LSTMSeq2SeqEncoder",
    "TransformerSeq2SeqEncoder",
    "Seq2SeqEncoder"
]

from fastNLP.modules.attention import MultiHeadAttention, BiAttention, SelfAttention
from .bert import BertModel
from .roberta import RobertaModel
from .gpt2 import GPT2Model
from .char_encoder import ConvolutionCharEncoder, LSTMCharEncoder
from .conv_maxpool import ConvMaxpool
from .lstm import LSTM
from .pooling import MaxPool, MaxPoolWithMask, AvgPool, AvgPoolWithMask, KMaxPool
from .star_transformer import StarTransformer
from .transformer import TransformerEncoder
from .variational_rnn import VarRNN, VarLSTM, VarGRU
from .seq2seq_encoder import LSTMSeq2SeqEncoder, TransformerSeq2SeqEncoder, Seq2SeqEncoder
