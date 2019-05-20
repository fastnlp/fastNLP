"""
    本模块实现了两种序列标注模型
"""
__all__ = [
    "SeqLabeling",
    "AdvSeqLabel"
]

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..modules import decoder, encoder
from ..modules.decoder.crf import allowed_transitions
from ..core.utils import seq_len_to_mask
from ..core.const import Const as C


class SeqLabeling(BaseModel):
    """
    别名：:class:`fastNLP.models.SeqLabeling`  :class:`fastNLP.models.sequence_labeling.SeqLabeling`

    一个基础的Sequence labeling的模型。
    用于做sequence labeling的基础类。结构包含一层Embedding，一层LSTM(单向，一层)，一层FC，以及一层CRF。
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int hidden_size: LSTM隐藏层的大小
    :param int num_classes: 一共有多少类
    """
    
    def __init__(self, init_embed, hidden_size, num_classes):
        super(SeqLabeling, self).__init__()
        
        self.Embedding = encoder.embedding.Embedding(init_embed)
        self.Rnn = encoder.lstm.LSTM(self.Embedding.embedding_dim, hidden_size)
        self.Linear = nn.Linear(hidden_size, num_classes)
        self.Crf = decoder.crf.ConditionalRandomField(num_classes)
        self.mask = None
    
    def forward(self, words, seq_len, target):
        """
        :param torch.LongTensor words: [batch_size, max_len]，序列的index
        :param torch.LongTensor seq_len: [batch_size,], 这个序列的长度
        :param torch.LongTensor target: [batch_size, max_len], 序列的目标值
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                    If truth is not None, return loss, a scalar. Used in training.
        """
        assert words.shape[0] == seq_len.shape[0]
        assert target.shape == words.shape
        self.mask = self._make_mask(words, seq_len)
        
        x = self.Embedding(words)
        # [batch_size, max_len, word_emb_dim]
        x, _ = self.Rnn(x, seq_len)
        # [batch_size, max_len, hidden_size * direction]
        x = self.Linear(x)
        # [batch_size, max_len, num_classes]
        return {C.LOSS: self._internal_loss(x, target)}
    
    def predict(self, words, seq_len):
        """
        用于在预测时使用

        :param torch.LongTensor words: [batch_size, max_len]
        :param torch.LongTensor seq_len: [batch_size,]
        :return: {'pred': xx}, [batch_size, max_len]
        """
        self.mask = self._make_mask(words, seq_len)
        
        x = self.Embedding(words)
        # [batch_size, max_len, word_emb_dim]
        x, _ = self.Rnn(x, seq_len)
        # [batch_size, max_len, hidden_size * direction]
        x = self.Linear(x)
        # [batch_size, max_len, num_classes]
        pred = self._decode(x)
        return {C.OUTPUT: pred}
    
    def _internal_loss(self, x, y):
        """
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        assert x.shape[:2] == y.shape
        assert y.shape == self.mask.shape
        total_loss = self.Crf(x, y, self.mask)
        return torch.mean(total_loss)
    
    def _make_mask(self, x, seq_len):
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_len_to_mask(seq_len)
        mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask
    
    def _decode(self, x):
        """
        :param torch.FloatTensor x: [batch_size, max_len, tag_size]
        :return prediction: [batch_size, max_len]
        """
        tag_seq, _ = self.Crf.viterbi_decode(x, self.mask)
        return tag_seq


class AdvSeqLabel(nn.Module):
    """
    别名：:class:`fastNLP.models.AdvSeqLabel`  :class:`fastNLP.models.sequence_labeling.AdvSeqLabel`

    更复杂的Sequence Labelling模型。结构为Embedding, LayerNorm, 双向LSTM(两层)，FC，LayerNorm，DropOut，FC，CRF。
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int hidden_size: LSTM的隐层大小
    :param int num_classes: 有多少个类
    :param float dropout: LSTM中以及DropOut层的drop概率
    :param dict id2words: tag id转为其tag word的表。用于在CRF解码时防止解出非法的顺序，比如'BMES'这个标签规范中，'S'
        不能出现在'B'之后。这里也支持类似与'B-NN'，即'-'前为标签类型的指示，后面为具体的tag的情况。这里不但会保证
        'B-NN'后面不为'S-NN'还会保证'B-NN'后面不会出现'M-xx'(任何非'M-NN'和'E-NN'的情况。)
    :param str encoding_type: 支持"BIO", "BMES", "BEMSO", 只有在id2words不为None的情况有用。
    """
    
    def __init__(self, init_embed, hidden_size, num_classes, dropout=0.3, id2words=None, encoding_type='bmes'):
        
        super().__init__()
        
        self.Embedding = encoder.embedding.Embedding(init_embed)
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
    
    def _decode(self, x):
        """
        :param torch.FloatTensor x: [batch_size, max_len, tag_size]
        :return torch.LongTensor, [batch_size, max_len]
        """
        tag_seq, _ = self.Crf.viterbi_decode(x, self.mask)
        return tag_seq
    
    def _internal_loss(self, x, y):
        """
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        assert x.shape[:2] == y.shape
        assert y.shape == self.mask.shape
        total_loss = self.Crf(x, y, self.mask)
        return torch.mean(total_loss)
    
    def _make_mask(self, x, seq_len):
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_len_to_mask(seq_len)
        mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask
    
    def _forward(self, words, seq_len, target=None):
        """
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len:[batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len]
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                   If truth is not None, return loss, a scalar. Used in training.
        """
        
        words = words.long()
        seq_len = seq_len.long()
        self.mask = self._make_mask(words, seq_len)
        
        # seq_len = seq_len.long()
        target = target.long() if target is not None else None
        
        if next(self.parameters()).is_cuda:
            words = words.cuda()
            self.mask = self.mask.cuda()
        
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
            return {"loss": self._internal_loss(x, target)}
        else:
            return {"pred": self._decode(x)}
    
    def forward(self, words, seq_len, target):
        """
        
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len], 目标
        :return torch.Tensor: a scalar loss
        """
        return self._forward(words, seq_len, target)
    
    def predict(self, words, seq_len):
        """
        
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :return torch.LongTensor: [batch_size, max_len]
        """
        return self._forward(words, seq_len)
