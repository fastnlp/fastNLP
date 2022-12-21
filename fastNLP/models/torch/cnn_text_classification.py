r"""
.. todo::
    doc
"""

__all__ = [
    "CNNText"
]

from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.utils import seq_len_to_mask
from ...embeddings.torch import embedding
from ...modules.torch import encoder


class CNNText(torch.nn.Module):
    r"""
    使用 **CNN** 进行文本分类的模型。
    论文参考 `Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>`_ 。

    :param embed: 单词词典，支持以下几种输入类型：

            - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
            - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
            - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
            - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；

    :param num_classes: 一共有多少类
    :param kernel_nums: 输出 channel 的 kernel 数目。
        如果为 :class:`list` 或 :class:`tuple`，则需要与 ``kernel_sizes`` 的大小保持一致。
    :param kernel_sizes: 输出 channel 的 kernel 大小。
    :param dropout: Dropout 的大小
    """
    def __init__(self, embed,
                 num_classes: int,
                 kernel_nums: Union[int, Tuple[int]] = (30, 40, 50),
                 kernel_sizes: Union[int, Tuple[int]] = (1, 3, 5),
                 dropout: float = 0.5):
        super(CNNText, self).__init__()

        # no support for pre-trained embedding currently
        self.embed = embedding.Embedding(embed)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=self.embed.embedding_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)

    def forward(self, words: "torch.LongTensor", seq_len: "torch.LongTensor"=None):
        r"""

        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param seq_len:  每个句子的长度，形状为 ``[batch,]`` 
        :return: 前向传播的结果，为仅包含一个键 ``pred`` 的字典
        """
        x = self.embed(words)  # [N,L] -> [N,L,C]
        if seq_len is not None:
            mask = seq_len_to_mask(seq_len)
            x = self.conv_pool(x, mask)
        else:
            x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        res = {'pred': x}
        return res

    def train_step(self, words, target, seq_len=None):
        """

        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param target: 每个 sample 的目标值
        :param seq_len: 每个句子的长度，形状为 ``[batch,]`` 
        :return: 类型为字典的结果，仅包含一个键 ``loss``，表示当次训练的 loss
        """
        res = self(words, seq_len)
        x = res['pred']
        loss = F.cross_entropy(x, target)
        return {'loss': loss}

    def evaluate_step(self, words: "torch.LongTensor", seq_len: "torch.LongTensor"=None):
        r"""

        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param seq_len:  每个句子的长度，形状为 ``[batch_size,]`` 
        :return: 预测结果，仅包含一个键 ``pred``，值为形状为 ``[batch_size,]`` 的 :class:`torch.LongTensor`
        """
        output = self(words, seq_len)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}
