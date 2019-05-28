__all__ = [
    "CNNText"
]

import torch
import torch.nn as nn

from ..core.const import Const as C
from ..modules import encoder


class CNNText(torch.nn.Module):
    """
    别名：:class:`fastNLP.models.CNNText`  :class:`fastNLP.models.cnn_text_classification.CNNText`

    使用CNN进行文本分类的模型
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification.'
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int num_classes: 一共有多少类
    :param int,tuple(int) out_channels: 输出channel的数量。如果为list，则需要与kernel_sizes的数量保持一致
    :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
    :param int padding: 对句子前后的pad的大小, 用0填充。
    :param float dropout: Dropout的大小
    """
    
    def __init__(self, init_embed,
                 num_classes,
                 kernel_nums=(3, 4, 5),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(CNNText, self).__init__()
        
        # no support for pre-trained embedding currently
        self.embed = encoder.Embedding(init_embed)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=self.embed.embedding_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)
    
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
        return {C.OUTPUT: x}
    
    def predict(self, words, seq_len=None):
        """
        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度

        :return predict: dict of torch.LongTensor, [batch_size, ]
        """
        output = self(words, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}
