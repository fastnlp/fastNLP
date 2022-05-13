"""
torch 可使用的几种 Embedding 。
"""
__all__ = [
    "CNNCharEmbedding",
    "LSTMCharEmbedding",
    "Embedding",
    "StackEmbedding",
    "StaticEmbedding"
]

from .char_embedding import *
from .embedding import *
from .stack_embedding import *
from .static_embedding import StaticEmbedding