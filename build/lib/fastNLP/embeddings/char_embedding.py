r"""
该文件中主要包含的是character的Embedding，包括基于CNN与LSTM的character Embedding。与其它Embedding一样，这里的Embedding输入也是
词的index而不需要使用词语中的char的index来获取表达。
"""

__all__ = [
    "CNNCharEmbedding",
    "LSTMCharEmbedding"
]

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TokenEmbedding
from .static_embedding import StaticEmbedding
from .utils import _construct_char_vocab_from_vocab
from .utils import get_embeddings
from ..core import logger
from ..core.vocabulary import Vocabulary
from ..modules.encoder.lstm import LSTM


class CNNCharEmbedding(TokenEmbedding):
    r"""
    使用CNN生成character embedding。CNN的结构为, embed(x) -> Dropout(x) -> CNN(x) -> activation(x) -> pool -> fc -> Dropout.
    不同的kernel大小的fitler结果是concat起来然后通过一层fully connected layer, 然后输出word的表示。

    Example::

        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import CNNCharEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = CNNCharEmbedding(vocab, embed_size=50)
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5，50])

    """
    
    def __init__(self, vocab: Vocabulary, embed_size: int = 50, char_emb_size: int = 50, word_dropout: float = 0,
                 dropout: float = 0, filter_nums: List[int] = (40, 30, 20), kernel_sizes: List[int] = (5, 3, 1),
                 pool_method: str = 'max', activation='relu', min_char_freq: int = 2, pre_train_char_embed: str = None,
                 requires_grad:bool=True, include_word_start_end:bool=True):
        r"""
        
        :param vocab: 词表
        :param embed_size: 该CNNCharEmbedding的输出维度大小，默认值为50.
        :param char_emb_size: character的embed的维度。character是从vocab中生成的。默认值为50.
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率drop分布式表示与char embedding的输出。
        :param filter_nums: filter的数量. 长度需要和kernels一致。默认值为[40, 30, 20].
        :param kernel_sizes: kernel的大小. 默认值为[5, 3, 1].
        :param pool_method: character的表示在合成一个表示时所使用的pool方法，支持'avg', 'max'.
        :param activation: CNN之后使用的激活方法，支持'relu', 'sigmoid', 'tanh' 或者自定义函数.
        :param min_char_freq: character的最少出现次数。默认值为2.
        :param pre_train_char_embed: 可以有两种方式调用预训练好的character embedding：第一种是传入embedding文件夹
            (文件夹下应该只有一个以.txt作为后缀的文件)或文件路径；第二种是传入embedding的名称，第二种情况将自动查看缓存中是否存在该模型，
            没有的话将自动下载。如果输入为None则使用embedding_dim的维度随机初始化一个embedding.
        :param requires_grad: 是否更新权重
        :param include_word_start_end: 是否在每个word开始的character前和结束的character增加特殊标示符号；
        """
        super(CNNCharEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        
        for kernel in kernel_sizes:
            assert kernel % 2 == 1, "Only odd kernel is allowed."
        
        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")
        
        logger.info("Start constructing character vocabulary.")
        # 建立char的词表
        self.char_vocab = _construct_char_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                           include_word_start_end=include_word_start_end)
        self.char_pad_index = self.char_vocab.padding_idx
        logger.info(f"In total, there are {len(self.char_vocab)} distinct characters.")
        # 对vocab进行index
        max_word_len = max(map(lambda x: len(x[0]), vocab))
        if include_word_start_end:
            max_word_len += 2
        self.register_buffer('words_to_chars_embedding', torch.full((len(vocab), max_word_len),
                                                                fill_value=self.char_pad_index, dtype=torch.long))
        self.register_buffer('word_lengths', torch.zeros(len(vocab)).long())
        for word, index in vocab:
            # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了。修改为不区分pad, 这样所有的<pad>也是同一个embed
            if include_word_start_end:
                word = ['<bow>'] + list(word) + ['<eow>']
            self.words_to_chars_embedding[index, :len(word)] = \
                torch.LongTensor([self.char_vocab.to_index(c) for c in word])
            self.word_lengths[index] = len(word)
        # self.char_embedding = nn.Embedding(len(self.char_vocab), char_emb_size)
        if pre_train_char_embed:
            self.char_embedding = StaticEmbedding(self.char_vocab, model_dir_or_name=pre_train_char_embed)
        else:
            self.char_embedding = get_embeddings((len(self.char_vocab), char_emb_size))
        
        self.convs = nn.ModuleList([nn.Conv1d(
            self.char_embedding.embedding_dim, filter_nums[i], kernel_size=kernel_sizes[i], bias=True,
            padding=kernel_sizes[i] // 2)
            for i in range(len(kernel_sizes))])
        self._embed_size = embed_size
        self.fc = nn.Linear(sum(filter_nums), embed_size)
        self.requires_grad = requires_grad

    def forward(self, words):
        r"""
        输入words的index后，生成对应的words的表示。

        :param words: [batch_size, max_len]
        :return: [batch_size, max_len, embed_size]
        """
        words = self.drop_word(words)
        batch_size, max_len = words.size()
        chars = self.words_to_chars_embedding[words]  # batch_size x max_len x max_word_len
        word_lengths = self.word_lengths[words]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        # 为1的地方为mask
        chars_masks = chars.eq(self.char_pad_index)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        chars = self.char_embedding(chars)  # batch_size x max_len x max_word_len x embed_size
        chars = self.dropout(chars)
        reshaped_chars = chars.reshape(batch_size * max_len, max_word_len, -1)
        reshaped_chars = reshaped_chars.transpose(1, 2)  # B' x E x M
        conv_chars = [conv(reshaped_chars).transpose(1, 2).reshape(batch_size, max_len, max_word_len, -1)
                      for conv in self.convs]
        conv_chars = torch.cat(conv_chars, dim=-1).contiguous()  # B x max_len x max_word_len x sum(filters)
        conv_chars = self.activation(conv_chars)
        if self.pool_method == 'max':
            conv_chars = conv_chars.masked_fill(chars_masks.unsqueeze(-1), float('-inf'))
            chars, _ = torch.max(conv_chars, dim=-2)  # batch_size x max_len x sum(filters)
        else:
            conv_chars = conv_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = torch.sum(conv_chars, dim=-2) / chars_masks.eq(False).sum(dim=-1, keepdim=True).float()
        chars = self.fc(chars)
        return self.dropout(chars)


class LSTMCharEmbedding(TokenEmbedding):
    r"""
    使用LSTM的方式对character进行encode. embed(x) -> Dropout(x) -> LSTM(x) -> activation(x) -> pool -> Dropout

    Example::

        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import LSTMCharEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = LSTMCharEmbedding(vocab, embed_size=50)
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5，50])

    """
    
    def __init__(self, vocab: Vocabulary, embed_size: int = 50, char_emb_size: int = 50, word_dropout: float = 0,
                 dropout: float = 0, hidden_size=50, pool_method: str = 'max', activation='relu',
                 min_char_freq: int = 2, bidirectional=True, pre_train_char_embed: str = None,
                 requires_grad:bool=True, include_word_start_end:bool=True):
        r"""
        
        :param vocab: 词表
        :param embed_size: LSTMCharEmbedding的输出维度。默认值为50.
        :param char_emb_size: character的embedding的维度。默认值为50.
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param dropout: 以多大概率drop character embedding的输出以及最终的word的输出。
        :param hidden_size: LSTM的中间hidden的大小，如果为bidirectional的，hidden会除二，默认为50.
        :param pool_method: 支持'max', 'avg'。
        :param activation: 激活函数，支持'relu', 'sigmoid', 'tanh', 或者自定义函数.
        :param min_char_freq: character的最小出现次数。默认值为2.
        :param bidirectional: 是否使用双向的LSTM进行encode。默认值为True。
        :param pre_train_char_embed: 可以有两种方式调用预训练好的character embedding：第一种是传入embedding文件夹
            (文件夹下应该只有一个以.txt作为后缀的文件)或文件路径；第二种是传入embedding的名称，第二种情况将自动查看缓存中是否存在该模型，
            没有的话将自动下载。如果输入为None则使用embedding_dim的维度随机初始化一个embedding.
        :param requires_grad: 是否更新权重
        :param include_word_start_end: 是否在每个word开始的character前和结束的character增加特殊标示符号；
        """
        super(LSTMCharEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        
        assert hidden_size % 2 == 0, "Only even kernel is allowed."
        
        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")
        
        logger.info("Start constructing character vocabulary.")
        # 建立char的词表
        self.char_vocab = _construct_char_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                           include_word_start_end=include_word_start_end)
        self.char_pad_index = self.char_vocab.padding_idx
        logger.info(f"In total, there are {len(self.char_vocab)} distinct characters.")
        # 对vocab进行index
        max_word_len = max(map(lambda x: len(x[0]), vocab))
        if include_word_start_end:
            max_word_len += 2
        self.register_buffer('words_to_chars_embedding', torch.full((len(vocab), max_word_len),
                                                                fill_value=self.char_pad_index, dtype=torch.long))
        self.register_buffer('word_lengths', torch.zeros(len(vocab)).long())
        for word, index in vocab:
            # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了. 修改为不区分pad与否
            if include_word_start_end:
                word = ['<bow>'] + list(word) + ['<eow>']
            self.words_to_chars_embedding[index, :len(word)] = \
                torch.LongTensor([self.char_vocab.to_index(c) for c in word])
            self.word_lengths[index] = len(word)
        if pre_train_char_embed:
            self.char_embedding = StaticEmbedding(self.char_vocab, pre_train_char_embed)
        else:
            self.char_embedding = get_embeddings((len(self.char_vocab), char_emb_size))
        
        self.fc = nn.Linear(hidden_size, embed_size)
        hidden_size = hidden_size // 2 if bidirectional else hidden_size
        
        self.lstm = LSTM(self.char_embedding.embedding_dim, hidden_size, bidirectional=bidirectional, batch_first=True)
        self._embed_size = embed_size
        self.bidirectional = bidirectional
        self.requires_grad = requires_grad
    
    def forward(self, words):
        r"""
        输入words的index后，生成对应的words的表示。

        :param words: [batch_size, max_len]
        :return: [batch_size, max_len, embed_size]
        """
        words = self.drop_word(words)
        batch_size, max_len = words.size()
        chars = self.words_to_chars_embedding[words]  # batch_size x max_len x max_word_len
        word_lengths = self.word_lengths[words]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        # 为mask的地方为1
        chars_masks = chars.eq(self.char_pad_index)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        chars = self.char_embedding(chars)  # batch_size x max_len x max_word_len x embed_size
        chars = self.dropout(chars)
        reshaped_chars = chars.reshape(batch_size * max_len, max_word_len, -1)
        char_seq_len = chars_masks.eq(False).sum(dim=-1).reshape(batch_size * max_len)
        lstm_chars = self.lstm(reshaped_chars, char_seq_len)[0].reshape(batch_size, max_len, max_word_len, -1)
        # B x M x M x H
        
        lstm_chars = self.activation(lstm_chars)
        if self.pool_method == 'max':
            lstm_chars = lstm_chars.masked_fill(chars_masks.unsqueeze(-1), float('-inf'))
            chars, _ = torch.max(lstm_chars, dim=-2)  # batch_size x max_len x H
        else:
            lstm_chars = lstm_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = torch.sum(lstm_chars, dim=-2) / chars_masks.eq(False).sum(dim=-1, keepdim=True).float()
        
        chars = self.fc(chars)
        
        return self.dropout(chars)
