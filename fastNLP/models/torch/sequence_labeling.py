r"""
本模块实现了几种序列标注模型。
"""
__all__ = [
    "SeqLabeling",
    "AdvSeqLabel",
    "BiLSTMCRF"
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.utils import seq_len_to_mask
from ...embeddings.torch.utils import get_embeddings
from ...modules.torch.decoder import ConditionalRandomField
from ...modules.torch.encoder import LSTM
from ...modules.torch import decoder, encoder
from ...modules.torch.decoder.crf import allowed_transitions


class BiLSTMCRF(nn.Module):
    r"""
    结构为 ``Embedding`` + :class:`BiLSTM <fastNLP.modules.torch.encoder.LSTM>` + ``FC`` + ``Dropout`` +
    :class:`CRF <fastNLP.modules.torch.decoder.ConditionalRandomField>` 。

    :param embed: 支持以下几种输入类型：

        - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
        - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
        - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
        - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；

    :param num_classes: 一共多少个类
    :param num_layers: **BiLSTM** 的层数
    :param hidden_size: **BiLSTM** 的 ``hidden_size``，实际 hidden size 为该值的 **两倍** （前向、后向）
    :param dropout: dropout 的概率，0 为不 dropout
    :param target_vocab: :class:`~fastNLP.core.Vocabulary` 对象，target 与 index 的对应关系。如果传入该值，将自动避免非法的解码序列。
    """
    def __init__(self, embed, num_classes, num_layers=1, hidden_size=100, dropout=0.5,
                  target_vocab=None):
        super().__init__()
        self.embed = get_embeddings(embed)

        if num_layers>1:
            self.lstm = LSTM(self.embed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True, dropout=dropout)
        else:
            self.lstm = LSTM(self.embed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)

        trans = None
        if target_vocab is not None:
            assert len(target_vocab)==num_classes, "The number of classes should be same with the length of target vocabulary."
            trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)

    def forward(self, words: "torch.LongTensor", target: "torch.LongTensor"=None, seq_len: "torch.LongTensor"=None):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param target: 每个 sample 的目标值
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 如果 ``target`` 为 ``None``，则返回预测结果 ``{'pred': torch.Tensor}``，否则返回 loss ``{'loss': torch.Tensor}``
        """
        words = self.embed(words)
        feats, _ = self.lstm(words, seq_len=seq_len)
        feats = self.fc(feats)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred':pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {'loss':loss}

    def train_step(self, words: "torch.LongTensor", target: "torch.LongTensor", seq_len: "torch.LongTensor"):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param target: 每个 sample 的目标值
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 如果 ``target`` 为 ``None``，则返回预测结果 ``{'pred': torch.Tensor}``，否则返回 loss ``{'loss': torch.Tensor}``
        """
        return self(words, seq_len, target)

    def evaluate_step(self, words: "torch.LongTensor", seq_len: "torch.LongTensor"):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 预测结果 ``{'pred': torch.Tensor}``
        """
        return self(words, seq_len)


class SeqLabeling(nn.Module):
    r"""
    一个基础的 Sequence labeling 的模型。
    用于做 sequence labeling 的基础类。结构包含一层 ``Embedding`` ，一层 :class:`~fastNLP.modules.torch.encoder.LSTM` (单向，一层)，
    一层全连接层，以及一层 :class:`CRF <fastNLP.modules.torch.decoder.ConditionalRandomField>` 。

    :param embed: 支持以下几种输入类型：

            - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
            - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
            - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
            - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
    :param hidden_size: :class:`fastNLP.modules.torch.encoder.LSTM` 隐藏层的大小
    :param num_classes: 一共有多少类
    """
    def __init__(self, embed, hidden_size: int, num_classes: int):
        super(SeqLabeling, self).__init__()
        
        self.embedding = get_embeddings(embed)
        self.rnn = encoder.LSTM(self.embedding.embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = decoder.ConditionalRandomField(num_classes)

    def forward(self, words: "torch.LongTensor", seq_len: "torch.LongTensor"):
        r"""
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 预测结果 ``{'pred': torch.Tensor}``
        """
        x = self.embedding(words)
        # [batch_size, max_len, word_emb_dim]
        x, _ = self.rnn(x, seq_len)
        # [batch_size, max_len, hidden_size * direction]
        x = self.fc(x)
        return {'pred': x}
        # [batch_size, max_len, num_classes]

    def train_step(self, words, target, seq_len):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param target: 每个 sample 的目标值
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 如果 ``target`` 为 ``None``，则返回预测结果 ``{'pred': torch.Tensor}``，否则返回 loss ``{'loss': torch.Tensor}``
        """
        res = self(words, seq_len)
        pred = res['pred']
        mask = seq_len_to_mask(seq_len, max_len=target.size(1))
        return {'loss': self._internal_loss(pred, target, mask)}

    def evaluate_step(self, words, seq_len):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 预测结果 ``{'pred': torch.Tensor}``
        """
        mask = seq_len_to_mask(seq_len, max_len=words.size(1))

        res = self(words, seq_len)
        pred = res['pred']
        # [batch_size, max_len, num_classes]
        pred = self._decode(pred, mask)
        return {'pred': pred}

    def _internal_loss(self, x, y, mask):
        r"""
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        total_loss = self.crf(x, y, mask)
        return torch.mean(total_loss)
    
    def _decode(self, x, mask):
        r"""
        :param torch.FloatTensor x: [batch_size, max_len, tag_size]
        :return prediction: [batch_size, max_len]
        """
        tag_seq, _ = self.crf.viterbi_decode(x, mask)
        return tag_seq


class AdvSeqLabel(nn.Module):
    r"""
    更复杂的 Sequence Labelling 模型。结构为 ``Embedding``, ``LayerNorm``, :class:`BiLSTM <fastNLP.modules.torch.encoder.LSTM>` （两层），
    ``FC``，``LayerNorm``，``Dropout``，``FC``，:class:`CRF <fastNLP.modules.torch.decoder.ConditionalRandomField>`。

    :param embed: 支持以下几种输入类型：

        - ``tuple(num_embedings, embedding_dim)``，即 embedding 的大小和每个词的维度，此时将随机初始化一个 :class:`torch.nn.Embedding` 实例；
        - :class:`torch.nn.Embedding` 或 **fastNLP** 的 ``Embedding`` 对象，此时就以传入的对象作为 embedding；
        - :class:`numpy.ndarray` ，将使用传入的 ndarray 作为 Embedding 初始化；
        - :class:`torch.Tensor`，此时将使用传入的值作为 Embedding 初始化；
    :param hidden_size: :class:`~fastNLP.modules.torch.LSTM` 的隐藏层大小
    :param num_classes: 有多少个类
    :param dropout: :class:`~fastNLP.modules.torch.LSTM` 中以及 DropOut 层的 drop 概率
    :param id2words: tag id 转为其 tag word 的表。用于在 CRF 解码时防止解出非法的顺序，比如 **'BMES'** 这个标签规范中，``'S'``
        不能出现在 ``'B'`` 之后。这里也支持类似于 ``'B-NN'``，即 ``'-'`` 前为标签类型的指示，后面为具体的 tag 的情况。这里不但会保证
        ``'B-NN'`` 后面不为 ``'S-NN'`` 还会保证 ``'B-NN'`` 后面不会出现 ``'M-xx'`` （任何非 ``'M-NN'`` 和 ``'E-NN'`` 的情况）。
    :param encoding_type: 支持 ``["BIO", "BMES", "BEMSO"]`` ，只有在 ``id2words`` 不为 ``None`` 的情况有用。
    """
    
    def __init__(self, embed, hidden_size: int, num_classes: int, dropout: float=0.3, id2words: dict=None, encoding_type: str='bmes'):
        super().__init__()
        
        self.Embedding = get_embeddings(embed)
        self.norm1 = torch.nn.LayerNorm(self.Embedding.embedding_dim)
        self.Rnn = encoder.LSTM(input_size=self.Embedding.embedding_dim, hidden_size=hidden_size, num_layers=2,
                                dropout=dropout,
                                bidirectional=True, batch_first=True)
        self.Linear1 = nn.Linear(hidden_size * 2, hidden_size * 2 // 3)
        self.norm2 = torch.nn.LayerNorm(hidden_size * 2 // 3)
        self.relu = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_size * 2 // 3, num_classes)
        
        if id2words is None:
            self.Crf = decoder.crf.ConditionalRandomField(num_classes, include_start_end_trans=False)
        else:
            self.Crf = decoder.crf.ConditionalRandomField(num_classes, include_start_end_trans=False,
                                                          allowed_transitions=allowed_transitions(id2words,
                                                                                                  encoding_type=encoding_type))
    
    def _decode(self, x, mask):
        r"""
        :param torch.FloatTensor x: [batch_size, max_len, tag_size]
        :param torch.ByteTensor mask: [batch_size, max_len]
        :return torch.LongTensor, [batch_size, max_len]
        """
        tag_seq, _ = self.Crf.viterbi_decode(x, mask)
        return tag_seq
    
    def _internal_loss(self, x, y, mask):
        r"""
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :param mask: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        total_loss = self.Crf(x, y, mask)
        return torch.mean(total_loss)
    
    def forward(self, words: "torch.LongTensor", target: "torch.LongTensor"=None, seq_len: "torch.LongTensor"=None):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param target: 每个 sample 的目标值
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 如果 ``target`` 为 ``None``，则返回预测结果 ``{'pred': torch.Tensor}``，否则返回 loss ``{'loss': torch.Tensor}``
        """
        
        words = words.long()
        seq_len = seq_len.long()
        mask = seq_len_to_mask(seq_len, max_len=words.size(1))

        target = target.long() if target is not None else None
        
        if next(self.parameters()).is_cuda:
            words = words.cuda()

        x = self.Embedding(words)
        x = self.norm1(x)
        # [batch_size, max_len, word_emb_dim]
        
        x, _ = self.Rnn(x, seq_len=seq_len)
        
        x = self.Linear1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.Linear2(x)
        if target is not None:
            return {"loss": self._internal_loss(x, target, mask)}
        else:
            return {"pred": self._decode(x, mask)}
    
    def train_step(self, words: "torch.LongTensor", target: "torch.LongTensor", seq_len: "torch.LongTensor"):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param target: 每个 sample 的目标值
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 如果 ``target`` 为 ``None``，则返回预测结果 ``{'pred': torch.Tensor}``，否则返回 loss ``{'loss': torch.Tensor}``
        """
        return self(words, seq_len, target)
    
    def evaluate_step(self, words: "torch.LongTensor", seq_len: "torch.LongTensor"):
        """
        :param words: 句子中 word 的 index，形状为 ``[batch_size, seq_len]``
        :param seq_len: 每个句子的长度，形状为 ``[batch,]``
        :return: 预测结果 ``{'pred': torch.Tensor}``
        """
        return self(words, seq_len)
