r"""

.. image:: figures/text_classification.png

大部分用于的 NLP 任务神经网络都可以看做由 :mod:`embedding<fastNLP.embeddings>` 、 :mod:`~fastNLP.modules.encoder` 、
:mod:`~fastNLP.modules.decoder` 三种模块组成。 本模块中实现了 fastNLP 提供的诸多模块组件，
可以帮助用户快速搭建自己所需的网络。几种模块的功能和常见组件如下:

.. csv-table::
   :header: "类型", "功能", "常见组件"

   "embedding", 参见 :mod:`/fastNLP.embeddings` ,  "Elmo, Bert"
   "encoder", "将输入编码为具有表示能力的向量", "CNN, LSTM, Transformer"
   "decoder", "将具有某种表示意义的向量解码为需要的输出形式 ", "MLP, CRF"
   "其它", "配合其它组件使用的组件", "Dropout"


"""
__all__ = [
    # "BertModel",

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

    "MLP",
    "ConditionalRandomField",
    "viterbi_decode",
    "allowed_transitions",

    "TimestepDropout",

    'summary',

    "BertTokenizer",
    "BertModel",

    "RobertaTokenizer",
    "RobertaModel",

    "GPT2Model",
    "GPT2Tokenizer",

    "TransformerSeq2SeqEncoder",
    "LSTMSeq2SeqEncoder",
    "Seq2SeqEncoder",

    "TransformerSeq2SeqDecoder",
    "LSTMSeq2SeqDecoder",
    "Seq2SeqDecoder",

    "TransformerState",
    "LSTMState",
    "State",

    "SequenceGenerator"
]

import sys

from . import decoder
from . import encoder
from .decoder import *
from .dropout import TimestepDropout
from .encoder import *
from .generator import *
from .utils import summary
from ..doc_utils import doc_process
from .tokenizer import *

doc_process(sys.modules[__name__])
