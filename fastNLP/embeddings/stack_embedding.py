r"""
.. todo::
    doc
"""

__all__ = [
    "StackEmbedding",
]

from typing import List

import torch
from torch import nn as nn

from .embedding import TokenEmbedding


class StackEmbedding(TokenEmbedding):
    r"""
    支持将多个embedding集合成一个embedding。

    Example::

        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import StaticEmbedding, StackEmbedding
        >>> vocab =  Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed_1 = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50d', requires_grad=True)
        >>> embed_2 = StaticEmbedding(vocab, model_dir_or_name='en-word2vec-300', requires_grad=True)
        >>> embed = StackEmbedding([embed_1, embed_2])

    """
    
    def __init__(self, embeds: List[TokenEmbedding], word_dropout=0, dropout=0):
        r"""
        
        :param embeds: 一个由若干个TokenEmbedding组成的list，要求每一个TokenEmbedding的词表都保持一致
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。不同embedidng会在相同的位置
            被设置为unknown。如果这里设置了dropout，则组成的embedding就不要再设置dropout了。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        """
        vocabs = []
        for embed in embeds:
            if hasattr(embed, 'get_word_vocab'):
                vocabs.append(embed.get_word_vocab())
        _vocab = vocabs[0]
        for vocab in vocabs[1:]:
            assert vocab == _vocab, "All embeddings in StackEmbedding should use the same word vocabulary."
        
        super(StackEmbedding, self).__init__(_vocab, word_dropout=word_dropout, dropout=dropout)
        assert isinstance(embeds, list)
        for embed in embeds:
            assert isinstance(embed, TokenEmbedding), "Only TokenEmbedding type is supported."
        self.embeds = nn.ModuleList(embeds)
        self._embed_size = sum([embed.embed_size for embed in self.embeds])
    
    def append(self, embed: TokenEmbedding):
        r"""
        添加一个embedding到结尾。
        :param embed:
        :return:
        """
        assert isinstance(embed, TokenEmbedding)
        self._embed_size += embed.embed_size
        self.embeds.append(embed)
        return self
    
    def pop(self):
        r"""
        弹出最后一个embed
        :return:
        """
        embed = self.embeds.pop()
        self._embed_size -= embed.embed_size
        return embed
    
    @property
    def embed_size(self):
        r"""
        该Embedding输出的vector的最后一维的维度。
        :return:
        """
        return self._embed_size
    
    def forward(self, words):
        r"""
        得到多个embedding的结果，并把结果按照顺序concat起来。

        :param words: batch_size x max_len
        :return: 返回的shape和当前这个stack embedding中embedding的组成有关
        """
        outputs = []
        words = self.drop_word(words)
        for embed in self.embeds:
            outputs.append(embed(words))
        outputs = self.dropout(torch.cat(outputs, dim=-1))
        return outputs
