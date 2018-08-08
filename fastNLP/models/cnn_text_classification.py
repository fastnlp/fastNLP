# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn

# import torch.nn.functional as F
from fastNLP.modules.encoder.conv_maxpool import ConvMaxpool


class CNNText(torch.nn.Module):
    """
    Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

    def __init__(self, class_num=9,
                 kernel_nums=[100, 100, 100], kernel_sizes=[3, 4, 5],
                 embed_num=1000, embed_dim=300, pretrained_embed=None,
                 drop_prob=0.5):
        super(CNNText, self).__init__()

        # no support for pre-trained embedding currently
        self.embed = nn.Embedding(embed_num, embed_dim, padding_idx=0)
        self.conv_pool = ConvMaxpool(
            in_channels=embed_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(sum(kernel_nums), class_num)

    def forward(self, x):
        x = self.embed(x)  # [N,L] -> [N,L,C]
        x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return x
