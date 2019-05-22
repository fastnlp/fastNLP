"""
大部分用于的 NLP 任务神经网络都可以看做由编码 :mod:`~fastNLP.modules.encoder` 、
聚合 :mod:`~fastNLP.modules.aggregator` 、解码 :mod:`~fastNLP.modules.decoder` 三种模块组成。

.. image:: figures/text_classification.png

:mod:`~fastNLP.modules` 中实现了 fastNLP 提供的诸多模块组件，可以帮助用户快速搭建自己所需的网络。
三种模块的功能和常见组件如下:

+-----------------------+-----------------------+-----------------------+
| module type           | functionality         | example               |
+=======================+=======================+=======================+
| encoder               | 将输入编码为具有具    | embedding, RNN, CNN,  |
|                       | 有表示能力的向量      | transformer           |
+-----------------------+-----------------------+-----------------------+
| aggregator            | 从多个向量中聚合信息  | self-attention,       |
|                       |                       | max-pooling           |
+-----------------------+-----------------------+-----------------------+
| decoder               | 将具有某种表示意义的  | MLP, CRF              |
|                       | 向量解码为需要的输出  |                       |
|                       | 形式                  |                       |
+-----------------------+-----------------------+-----------------------+

"""
__all__ = [
    # "BertModel",
    "ConvolutionCharEncoder",
    "LSTMCharEncoder",
    "ConvMaxpool",
    "Embedding",
    "LSTM",
    "StarTransformer",
    "TransformerEncoder",
    "VarRNN",
    "VarLSTM",
    "VarGRU",
    
    "MaxPool",
    "MaxPoolWithMask",
    "AvgPool",
    "MultiHeadAttention",
    
    "MLP",
    "ConditionalRandomField",
    "viterbi_decode",
    "allowed_transitions",
]

from . import aggregator
from . import decoder
from . import encoder
from .aggregator import *
from .decoder import *
from .dropout import TimestepDropout
from .encoder import *
from .utils import get_embeddings
