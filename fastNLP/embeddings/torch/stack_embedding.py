r"""
.. todo::
    doc
"""

__all__ = [
    "StackEmbedding",
]

from typing import List

from ...envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from torch import nn

from .embedding import TokenEmbedding
from .utils import _check_vocab_has_same_index


class StackEmbedding(TokenEmbedding):
    r"""
    支持将多个 embedding 集合成一个 embedding。

    Example::

        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings.torch import StaticEmbedding, StackEmbedding
        >>> vocab =  Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed_1 = StaticEmbedding(vocab, model_dir_or_name='en-glove-6b-50d', requires_grad=True)
        >>> embed_2 = StaticEmbedding(vocab, model_dir_or_name='en-word2vec-300', requires_grad=True)
        >>> embed = StackEmbedding([embed_1, embed_2])

    :param embeds: 一个由若干个 :class:`~fastNLP.embeddings.torch.embedding.TokenEmbedding` 组成的 :class:`list` ，要求
        每一个 ``TokenEmbedding`` 的词表都保持一致
    :param word_dropout: 按照一定概率随机将 word 设置为 ``unk_index`` ，这样可以使得 ``<UNK>`` 这个 token 得到足够的训练，
        且会对网络有一定的 regularize 作用。不同 embedidng 会在相同的位置被设置为 ``<UNK>`` 。 如果这里设置了 dropout，则
        组成的 embedding 就不要再设置 dropout 了。
    :param dropout: 以多大的概率对 embedding 的表示进行 Dropout。0.1 即随机将 10% 的值置为 0。
    """
    
    def __init__(self, embeds: List[TokenEmbedding], word_dropout=0, dropout=0):

        vocabs = []
        for embed in embeds:
            if hasattr(embed, 'get_word_vocab'):
                vocabs.append(embed.get_word_vocab())
        _vocab = vocabs[0]
        for vocab in vocabs[1:]:
            if _vocab!=vocab:
                _check_vocab_has_same_index(_vocab, vocab)

        super(StackEmbedding, self).__init__(_vocab, word_dropout=word_dropout, dropout=dropout)
        assert isinstance(embeds, list)
        for embed in embeds:
            assert isinstance(embed, TokenEmbedding), "Only TokenEmbedding type is supported."
        self.embeds = nn.ModuleList(embeds)
        self._embed_size = sum([embed.embed_size for embed in self.embeds])
    
    def append(self, embed: TokenEmbedding):
        r"""
        添加一个 embedding 到结尾。

        :param embed:
        :return: 自身
        """
        assert isinstance(embed, TokenEmbedding)
        _check_vocab_has_same_index(self.get_word_vocab(), embed.get_word_vocab())
        self._embed_size += embed.embed_size
        self.embeds.append(embed)
        return self
    
    def pop(self):
        r"""
        弹出最后一个 embedding

        :return: 被弹出的 embedding
        """
        embed = self.embeds.pop()
        self._embed_size -= embed.embed_size
        return embed
    
    @property
    def embed_size(self):
        r"""
        该 Embedding 输出的 vector 的最后一维的维度。
        """
        return self._embed_size
    
    def forward(self, words):
        r"""
        得到多个 embedding 的结果，并把结果按照顺序连接起来。

        :param words: 形状为 ``[batch_size, max_len]``
        :return: 形状和当前这个 :class:`StackEmbedding` 中 embedding 的组成有关
        """
        outputs = []
        words = self.drop_word(words)
        for embed in self.embeds:
            outputs.append(embed(words))
        outputs = self.dropout(torch.cat(outputs, dim=-1))
        return outputs
