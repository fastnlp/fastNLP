r"""
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

    "RobertaEmbedding",
    "RobertaWordPieceEncoder",

    "GPT2Embedding",
    "GPT2WordPieceEncoder",

    "StackEmbedding",
    "LSTMCharEmbedding",
    "CNNCharEmbedding",

    "get_embeddings",
    "get_sinusoid_encoding_table"
]

from .embedding import Embedding, TokenEmbedding
from .static_embedding import StaticEmbedding
from .elmo_embedding import ElmoEmbedding
from .bert_embedding import BertEmbedding, BertWordPieceEncoder
from .roberta_embedding import RobertaEmbedding, RobertaWordPieceEncoder
from .gpt2_embedding import GPT2WordPieceEncoder, GPT2Embedding
from .char_embedding import CNNCharEmbedding, LSTMCharEmbedding
from .stack_embedding import StackEmbedding
from .utils import get_embeddings, get_sinusoid_encoding_table

import sys
from ..doc_utils import doc_process
doc_process(sys.modules[__name__])