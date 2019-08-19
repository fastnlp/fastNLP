import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fastNLP.models.base_model import BaseModel
from fastNLP.embeddings import TokenEmbedding
from fastNLP.core.const import Const


class DynamicKMaxPooling(nn.Module):
    """
    :param k_top: Fixed number of pooling output features for the topmost convolutional layer.
    :param l: Number of convolutional layers.
    """

    def __init__(self, k_top, l):
        super(DynamicKMaxPooling, self).__init__()
        self.k_top = k_top
        self.L = l

    def forward(self, x, l):
        """
        :param x: Input sequence.
        :param l: Current convolutional layers.
        """
        s = x.size()[3]
        k_ll = ((self.L - l) / self.L) * s
        k_l = int(round(max(self.k_top, np.ceil(k_ll))))
        out = F.adaptive_max_pool2d(x, (x.size()[2], k_l))
        return out


class CNTNModel(BaseModel):
    """
    使用CNN进行问答匹配的模型
    'Qiu, Xipeng, and Xuanjing Huang.
    Convolutional neural tensor network architecture for community-based question answering.
    Twenty-Fourth International Joint Conference on Artificial Intelligence. 2015.'

    :param init_embedding: Embedding.
    :param ns: Sentence embedding size.
    :param k_top: Fixed number of pooling output features for the topmost convolutional layer.
    :param num_labels: Number of labels.
    :param depth: Number of convolutional layers.
    :param r: Number of weight tensor slices.
    :param drop_rate: Dropout rate.
    """

    def __init__(self, init_embedding: TokenEmbedding, ns=200, k_top=10, num_labels=2, depth=2, r=5,
                 dropout_rate=0.3):
        super(CNTNModel, self).__init__()
        self.embedding = init_embedding
        self.depth = depth
        self.kmaxpooling = DynamicKMaxPooling(k_top, depth)
        self.conv_q = nn.ModuleList()
        self.conv_a = nn.ModuleList()
        width = self.embedding.embed_size
        for i in range(depth):
            self.conv_q.append(nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Conv2d(
                    in_channels=1,
                    out_channels=width // 2,
                    kernel_size=(width, 3),
                    padding=(0, 2))
            ))
            self.conv_a.append(nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Conv2d(
                    in_channels=1,
                    out_channels=width // 2,
                    kernel_size=(width, 3),
                    padding=(0, 2))
            ))
            width = width // 2

        self.fc_q = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(width * k_top, ns))
        self.fc_a = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(width * k_top, ns))
        self.weight_M = nn.Bilinear(ns, ns, r)
        self.weight_V = nn.Linear(2 * ns, r)
        self.weight_u = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(r, num_labels))

    def forward(self, words1, words2, seq_len1, seq_len2):
        """
        :param words1: [batch, seq_len, emb_size] Question.
        :param words2: [batch, seq_len, emb_size] Answer.
        :param seq_len1: [batch]
        :param seq_len2: [batch]
        :return:
        """
        in_q = self.embedding(words1)
        in_a = self.embedding(words2)
        in_q = in_q.permute(0, 2, 1).unsqueeze(1)
        in_a = in_a.permute(0, 2, 1).unsqueeze(1)

        for i in range(self.depth):
            in_q = F.relu(self.conv_q[i](in_q))
            in_q = in_q.squeeze().unsqueeze(1)
            in_q = self.kmaxpooling(in_q, i + 1)
            in_a = F.relu(self.conv_a[i](in_a))
            in_a = in_a.squeeze().unsqueeze(1)
            in_a = self.kmaxpooling(in_a, i + 1)

        in_q = self.fc_q(in_q.view(in_q.size(0), -1))
        in_a = self.fc_q(in_a.view(in_a.size(0), -1))
        score = torch.tanh(self.weight_u(self.weight_M(in_q, in_a) + self.weight_V(torch.cat((in_q, in_a), -1))))

        return {Const.OUTPUT: score}

    def predict(self, words1, words2, seq_len1, seq_len2):
        return self.forward(words1, words2, seq_len1, seq_len2)
