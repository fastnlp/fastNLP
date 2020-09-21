r"""
该模块中的Embedding主要用于随机初始化的embedding(更推荐使用 :class:`fastNLP.embeddings.StaticEmbedding` )，或按照预训练权重初始化Embedding。

"""

__all__ = [
    "Embedding",
    "TokenEmbedding"
]

from abc import abstractmethod

import torch
import torch.nn as nn

from .utils import get_embeddings


class Embedding(nn.Module):
    r"""
    词向量嵌入，支持输入多种方式初始化. 可以通过self.num_embeddings获取词表大小; self.embedding_dim获取embedding的维度.

    Example::

        >>> import numpy as np
        >>> from fastNLP.embeddings import Embedding
        >>> init_embed = (2000, 100)
        >>> embed = Embedding(init_embed)  # 随机初始化一个具有2000个词，每个词表示为100维的词向量
        >>> init_embed = np.zeros((2000, 100))
        >>> embed = Embedding(init_embed)  # 使用numpy.ndarray的值作为初始化值初始化一个Embedding

    """
    
    def __init__(self, init_embed, word_dropout=0, dropout=0.0, unk_index=None):
        r"""
        
        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: 支持传入Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 或传入Tensor, Embedding, numpy.ndarray等则直接使用该值初始化Embedding;
        :param float word_dropout: 按照一定概率随机将word设置为unk_index，这样可以使得unk这个token得到足够的训练, 且会对网络有
            一定的regularize的作用。设置该值时，必须同时设置unk_index
        :param float dropout: 对Embedding的输出的dropout。
        :param int unk_index: drop word时替换为的index。fastNLP的Vocabulary的unk_index默认为1。
        """
        super(Embedding, self).__init__()
        
        self.embed = get_embeddings(init_embed)
        
        self.dropout = nn.Dropout(dropout)
        if not isinstance(self.embed, TokenEmbedding):
            if hasattr(self.embed, 'embed_size'):
                self._embed_size = self.embed.embed_size
            elif hasattr(self.embed, 'embedding_dim'):
                self._embed_size = self.embed.embedding_dim
            else:
                self._embed_size = self.embed.weight.size(1)
            if word_dropout > 0 and not isinstance(unk_index, int):
                raise ValueError("When drop word is set, you need to pass in the unk_index.")
        else:
            self._embed_size = self.embed.embed_size
            unk_index = self.embed.get_word_vocab().unknown_idx
        self.unk_index = unk_index
        self.word_dropout = word_dropout
    
    def forward(self, words):
        r"""
        :param torch.LongTensor words: [batch, seq_len]
        :return: torch.Tensor : [batch, seq_len, embed_dim]
        """
        if self.word_dropout > 0 and self.training:
            mask = torch.ones_like(words).float() * self.word_dropout
            mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            words = words.masked_fill(mask, self.unk_index)
        words = self.embed(words)
        return self.dropout(words)
    
    @property
    def num_embedding(self) -> int:
        if isinstance(self.embed, nn.Embedding):
            return self.embed.weight.size(0)
        else:
            return self.embed.num_embeddings
    
    def __len__(self):
        return len(self.embed)
    
    @property
    def embed_size(self) -> int:
        return self._embed_size
    
    @property
    def embedding_dim(self) -> int:
        return self._embed_size
    
    @property
    def requires_grad(self):
        r"""
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        :return:
        """
        if not isinstance(self.embed, TokenEmbedding):
            return self.embed.weight.requires_grad
        else:
            return self.embed.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(self.embed, TokenEmbedding):
            self.embed.weight.requires_grad = value
        else:
            self.embed.requires_grad = value
    
    @property
    def size(self):
        if isinstance(self.embed, TokenEmbedding):
            return self.embed.size
        else:
            return self.embed.weight.size()


class TokenEmbedding(nn.Module):
    r"""
    fastNLP中各种Embedding的基类

    """
    def __init__(self, vocab, word_dropout=0.0, dropout=0.0):
        super(TokenEmbedding, self).__init__()
        if vocab.rebuild:
            vocab.build_vocab()
        assert vocab.padding is not None, "Vocabulary must have a padding entry."
        self._word_vocab = vocab
        self._word_pad_index = vocab.padding_idx
        if word_dropout > 0:
            assert vocab.unknown is not None, "Vocabulary must have unknown entry when you want to drop a word."
        self.word_dropout = word_dropout
        self._word_unk_index = vocab.unknown_idx
        self.dropout_layer = nn.Dropout(dropout)
    
    def drop_word(self, words):
        r"""
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
            mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words
    
    def dropout(self, words):
        r"""
        对embedding后的word表示进行drop。

        :param torch.FloatTensor words: batch_size x max_len x embed_size
        :return:
        """
        return self.dropout_layer(words)
    
    @property
    def requires_grad(self):
        r"""
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        :return:
        """
        requires_grads = set([param.requires_grad for param in self.parameters()])
        if len(requires_grads) == 1:
            return requires_grads.pop()
        else:
            return None
    
    @requires_grad.setter
    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value
    
    def __len__(self):
        return len(self._word_vocab)
    
    @property
    def embed_size(self) -> int:
        return self._embed_size
    
    @property
    def embedding_dim(self) -> int:
        return self._embed_size
    
    @property
    def num_embeddings(self) -> int:
        r"""
        这个值可能会大于实际的embedding矩阵的大小。
        :return:
        """
        return len(self._word_vocab)
    
    def get_word_vocab(self):
        r"""
        返回embedding的词典。

        :return: Vocabulary
        """
        return self._word_vocab
    
    @property
    def size(self):
        return torch.Size(self.num_embeddings, self._embed_size)
    
    @abstractmethod
    def forward(self, words):
        raise NotImplementedError
