# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn
import numpy as np

import fastNLP.modules.encoder as encoder


class CNNText(torch.nn.Module):
    """
    使用CNN进行文本分类的模型
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification.'
    """

    def __init__(self, vocab_size,
                 embed_dim,
                 num_classes,
                 kernel_nums=(3, 4, 5),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        """

        :param int vocab_size: 词表的大小
        :param int embed_dim: 词embedding的维度大小
        :param int num_classes: 一共有多少类
        :param int,tuple(int) out_channels: 输出channel的数量。如果为list，则需要与kernel_sizes的数量保持一致
        :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
        :param int padding:
        :param float dropout: Dropout的大小
        """
        super(CNNText, self).__init__()

        # no support for pre-trained embedding currently
        self.embed = encoder.Embedding(vocab_size, embed_dim)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=embed_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = encoder.Linear(sum(kernel_nums), num_classes)

    def init_embed(self, embed):
        """
        加载预训练的模型
        :param numpy.ndarray embed: vocab_size x embed_dim的embedding
        :return:
        """
        assert isinstance(embed, np.ndarray)
        assert embed.shape == self.embed.embed.weight.shape
        self.embed.embed.weight.data = torch.from_numpy(embed)

    def forward(self, words, seq_len=None):
        """

        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(words)  # [N,L] -> [N,L,C]
        x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return {'pred': x}

    def predict(self, words, seq_len=None):
        """
        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度

        :return predict: dict of torch.LongTensor, [batch_size, ]
        """
        output = self(words, seq_len)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}
