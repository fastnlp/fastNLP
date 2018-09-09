# python: 3.6
# encoding: utf-8

import torch
import torch.nn as nn

# import torch.nn.functional as F
import fastNLP.modules.encoder as encoder


class CNNText(torch.nn.Module):
    """
    Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

    def __init__(self, args):
        super(CNNText, self).__init__()

        num_classes = args["num_classes"]
        kernel_nums = [100, 100, 100]
        kernel_sizes = [3, 4, 5]
        vocab_size = args["vocab_size"]
        embed_dim = 300
        pretrained_embed = None
        drop_prob = 0.5

        # no support for pre-trained embedding currently
        self.embed = encoder.embedding.Embedding(vocab_size, embed_dim)
        self.conv_pool = encoder.conv_maxpool.ConvMaxpool(
            in_channels=embed_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = encoder.linear.Linear(sum(kernel_nums), num_classes)

    def forward(self, x):
        x = self.embed(x)  # [N,L] -> [N,L,C]
        x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return x
