"""
embeddings 模块主要用于从各种预训练的模型中获取词语的分布式表示，目前支持的预训练模型包括word2vec, glove, ELMO, BERT等。这里所有
embedding的forward输入都是形状为 ``(batch_size, max_len)`` 的torch.LongTensor，输出都是 ``(batch_size, max_len, embedding_dim)`` 的
torch.FloatTensor。所有的embedding都可以使用 `self.num_embedding` 获取最大的输入index范围, 用 `self.embeddig_dim` 或 `self.embed_size` 获取embedding的
输出维度。
"""

__all__ = [
    "Embedding",
    "TokenEmbedding",
    "StaticEmbedding",
    "ElmoEmbedding",
    "BertEmbedding",
    "BertWordPieceEncoder",
    "StackEmbedding",
    "LSTMCharEmbedding",
    "CNNCharEmbedding",
    "get_embeddings",
]

from .embedding import Embedding, TokenEmbedding
from .static_embedding import StaticEmbedding
from .elmo_embedding import ElmoEmbedding
from .bert_embedding import BertEmbedding, BertWordPieceEncoder
from .char_embedding import CNNCharEmbedding, LSTMCharEmbedding
from .stack_embedding import StackEmbedding
from .utils import get_embeddings

import sys
from ..doc_utils import doc_process
doc_process(sys.modules[__name__])