"""
embeddings 模块主要用于从各种预训练的模型中获取词语的分布式表示，目前支持的预训练模型包括word2vec, glove, ELMO, BERT等。这里所有
embedding的forward输入都是形状为(batch_size, max_len)的torch.LongTensor，输出都是(batch_size, max_len, embedding_dim)的
torch.FloatTensor。所有的embedding都可以使用num_embedding获取最大的输入index范围, 用embedding_dim或embed_size获取embedding的
输出维度。
"""

__all__ = [
    "Embedding",
    "StaticEmbedding",
    "ElmoEmbedding",
    "BertEmbedding",
    "StackEmbedding",
    "LSTMCharEmbedding",
    "CNNCharEmbedding",
    "get_embeddings"
]


from .embedding import Embedding
from .static_embedding import StaticEmbedding
from .elmo_embedding import ElmoEmbedding
from .bert_embedding import BertEmbedding
from .char_embedding import CNNCharEmbedding, LSTMCharEmbedding
from .stack_embedding import StackEmbedding
from .utils import get_embeddings