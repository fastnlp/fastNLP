r"""
Star-Transformer 的 Pytorch 实现。
"""
__all__ = [
    "StarTransEnc",
    "STNLICls",
    "STSeqCls",
    "STSeqLabel",
]

import torch
from torch import nn

from ..core.const import Const
from ..core.utils import seq_len_to_mask
from ..embeddings.utils import get_embeddings
from ..modules.encoder.star_transformer import StarTransformer


class StarTransEnc(nn.Module):
    r"""
    带word embedding的Star-Transformer Encoder

    """

    def __init__(self, embed,
                 hidden_size,
                 num_layers,
                 num_head,
                 head_dim,
                 max_len,
                 emb_dropout,
                 dropout):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象,此时就以传入的对象作为embedding
        :param hidden_size: 模型中特征维度.
        :param num_layers: 模型层数.
        :param num_head: 模型中multi-head的head个数.
        :param head_dim: 模型中multi-head中每个head特征维度.
        :param max_len: 模型能接受的最大输入长度.
        :param emb_dropout: 词嵌入的dropout概率.
        :param dropout: 模型除词嵌入外的dropout概率.
        """
        super(StarTransEnc, self).__init__()
        self.embedding = get_embeddings(embed)
        emb_dim = self.embedding.embedding_dim
        self.emb_fc = nn.Linear(emb_dim, hidden_size)
        # self.emb_drop = nn.Dropout(emb_dropout)
        self.encoder = StarTransformer(hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       num_head=num_head,
                                       head_dim=head_dim,
                                       dropout=dropout,
                                       max_len=max_len)

    def forward(self, x, mask):
        r"""
        :param FloatTensor x: [batch, length, hidden] 输入的序列
        :param ByteTensor mask: [batch, length] 输入序列的padding mask, 在没有内容(padding 部分) 为 0,
            否则为 1
        :return: [batch, length, hidden] 编码后的输出序列

                [batch, hidden] 全局 relay 节点, 详见论文
        """
        x = self.embedding(x)
        x = self.emb_fc(x)
        nodes, relay = self.encoder(x, mask)
        return nodes, relay


class _Cls(nn.Module):
    def __init__(self, in_dim, num_cls, hid_dim, dropout=0.1):
        super(_Cls, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_cls),
        )

    def forward(self, x):
        h = self.fc(x)
        return h


class _NLICls(nn.Module):
    def __init__(self, in_dim, num_cls, hid_dim, dropout=0.1):
        super(_NLICls, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim * 4, hid_dim),  # 4
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_cls),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2, torch.abs(x1 - x2), x1 * x2], 1)
        h = self.fc(x)
        return h


class STSeqLabel(nn.Module):
    r"""
    用于序列标注的Star-Transformer模型

    """

    def __init__(self, embed, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1, ):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象, 此时就以传入的对象作为embedding
        :param num_cls: 输出类别个数
        :param hidden_size: 模型中特征维度. Default: 300
        :param num_layers: 模型层数. Default: 4
        :param num_head: 模型中multi-head的head个数. Default: 8
        :param head_dim: 模型中multi-head中每个head特征维度. Default: 32
        :param max_len: 模型能接受的最大输入长度. Default: 512
        :param cls_hidden_size: 分类器隐层维度. Default: 600
        :param emb_dropout: 词嵌入的dropout概率. Default: 0.1
        :param dropout: 模型除词嵌入外的dropout概率. Default: 0.1
        """
        super(STSeqLabel, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _Cls(hidden_size, num_cls, cls_hidden_size)

    def forward(self, words, seq_len):
        r"""

        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls, seq_len] 输出序列中每个元素的分类的概率
        """
        mask = seq_len_to_mask(seq_len)
        nodes, _ = self.enc(words, mask)
        output = self.cls(nodes)
        output = output.transpose(1, 2)  # make hidden to be dim 1
        return {Const.OUTPUT: output}  # [bsz, n_cls, seq_len]

    def predict(self, words, seq_len):
        r"""

        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, seq_len] 输出序列中每个元素的分类
        """
        y = self.forward(words, seq_len)
        _, pred = y[Const.OUTPUT].max(1)
        return {Const.OUTPUT: pred}


class STSeqCls(nn.Module):
    r"""
    用于分类任务的Star-Transformer

    """

    def __init__(self, embed, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1, ):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象, 此时就以传入的对象作为embedding
        :param num_cls: 输出类别个数
        :param hidden_size: 模型中特征维度. Default: 300
        :param num_layers: 模型层数. Default: 4
        :param num_head: 模型中multi-head的head个数. Default: 8
        :param head_dim: 模型中multi-head中每个head特征维度. Default: 32
        :param max_len: 模型能接受的最大输入长度. Default: 512
        :param cls_hidden_size: 分类器隐层维度. Default: 600
        :param emb_dropout: 词嵌入的dropout概率. Default: 0.1
        :param dropout: 模型除词嵌入外的dropout概率. Default: 0.1
        """
        super(STSeqCls, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _Cls(hidden_size, num_cls, cls_hidden_size, dropout=dropout)

    def forward(self, words, seq_len):
        r"""

        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类的概率
        """
        mask = seq_len_to_mask(seq_len)
        nodes, relay = self.enc(words, mask)
        y = 0.5 * (relay + nodes.max(1)[0])
        output = self.cls(y)  # [bsz, n_cls]
        return {Const.OUTPUT: output}

    def predict(self, words, seq_len):
        r"""

        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类
        """
        y = self.forward(words, seq_len)
        _, pred = y[Const.OUTPUT].max(1)
        return {Const.OUTPUT: pred}


class STNLICls(nn.Module):
    r"""
    用于自然语言推断(NLI)的Star-Transformer

    """

    def __init__(self, embed, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1, ):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象, 此时就以传入的对象作为embedding
        :param num_cls: 输出类别个数
        :param hidden_size: 模型中特征维度. Default: 300
        :param num_layers: 模型层数. Default: 4
        :param num_head: 模型中multi-head的head个数. Default: 8
        :param head_dim: 模型中multi-head中每个head特征维度. Default: 32
        :param max_len: 模型能接受的最大输入长度. Default: 512
        :param cls_hidden_size: 分类器隐层维度. Default: 600
        :param emb_dropout: 词嵌入的dropout概率. Default: 0.1
        :param dropout: 模型除词嵌入外的dropout概率. Default: 0.1
        """
        super(STNLICls, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _NLICls(hidden_size, num_cls, cls_hidden_size)

    def forward(self, words1, words2, seq_len1, seq_len2):
        r"""

        :param words1: [batch, seq_len] 输入序列1
        :param words2: [batch, seq_len] 输入序列2
        :param seq_len1: [batch,] 输入序列1的长度
        :param seq_len2: [batch,] 输入序列2的长度
        :return output: [batch, num_cls] 输出分类的概率
        """
        mask1 = seq_len_to_mask(seq_len1)
        mask2 = seq_len_to_mask(seq_len2)

        def enc(seq, mask):
            nodes, relay = self.enc(seq, mask)
            return 0.5 * (relay + nodes.max(1)[0])

        y1 = enc(words1, mask1)
        y2 = enc(words2, mask2)
        output = self.cls(y1, y2)  # [bsz, n_cls]
        return {Const.OUTPUT: output}

    def predict(self, words1, words2, seq_len1, seq_len2):
        r"""

        :param words1: [batch, seq_len] 输入序列1
        :param words2: [batch, seq_len] 输入序列2
        :param seq_len1: [batch,] 输入序列1的长度
        :param seq_len2: [batch,] 输入序列2的长度
        :return output: [batch, num_cls] 输出分类的概率
        """
        y = self.forward(words1, words2, seq_len1, seq_len2)
        _, pred = y[Const.OUTPUT].max(1)
        return {Const.OUTPUT: pred}
