r"""
本模块实现了几种序列标注模型
"""
__all__ = [
    "SeqLabeling",
    "AdvSeqLabel",
    "BiLSTMCRF"
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from ..core.const import Const as C
from ..core.utils import seq_len_to_mask
from ..embeddings.utils import get_embeddings
from ..modules.decoder import ConditionalRandomField
from ..modules.encoder import LSTM
from ..modules import decoder, encoder
from ..modules.decoder.crf import allowed_transitions


class BiLSTMCRF(BaseModel):
    r"""
    结构为embedding + BiLSTM + FC + Dropout + CRF.

    """
    def __init__(self, embed, num_classes, num_layers=1, hidden_size=100, dropout=0.5,
                  target_vocab=None):
        r"""
        
        :param embed: 支持(1)fastNLP的各种Embedding, (2) tuple, 指明num_embedding, dimension, 如(1000, 100)
        :param num_classes: 一共多少个类
        :param num_layers: BiLSTM的层数
        :param hidden_size: BiLSTM的hidden_size，实际hidden size为该值的两倍(前向、后向)
        :param dropout: dropout的概率，0为不dropout
        :param target_vocab: Vocabulary对象，target与index的对应关系。如果传入该值，将自动避免非法的解码序列。
        """
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

    def _forward(self, words, seq_len=None, target=None):
        words = self.embed(words)
        feats, _ = self.lstm(words, seq_len=seq_len)
        feats = self.fc(feats)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {C.OUTPUT:pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {C.LOSS:loss}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)


class SeqLabeling(BaseModel):
    r"""
    一个基础的Sequence labeling的模型。
    用于做sequence labeling的基础类。结构包含一层Embedding，一层LSTM(单向，一层)，一层FC，以及一层CRF。
    
    """
    
    def __init__(self, embed, hidden_size, num_classes):
        r"""
        
        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, embedding, ndarray等则直接使用该值初始化Embedding
        :param int hidden_size: LSTM隐藏层的大小
        :param int num_classes: 一共有多少类
        """
        super(SeqLabeling, self).__init__()
        
        self.embedding = get_embeddings(embed)
        self.rnn = encoder.LSTM(self.embedding.embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = decoder.ConditionalRandomField(num_classes)

    def forward(self, words, seq_len, target):
        r"""
        :param torch.LongTensor words: [batch_size, max_len]，序列的index
        :param torch.LongTensor seq_len: [batch_size,], 这个序列的长度
        :param torch.LongTensor target: [batch_size, max_len], 序列的目标值
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                    If truth is not None, return loss, a scalar. Used in training.
        """
        mask = seq_len_to_mask(seq_len, max_len=words.size(1))
        x = self.embedding(words)
        # [batch_size, max_len, word_emb_dim]
        x, _ = self.rnn(x, seq_len)
        # [batch_size, max_len, hidden_size * direction]
        x = self.fc(x)
        # [batch_size, max_len, num_classes]
        return {C.LOSS: self._internal_loss(x, target, mask)}
    
    def predict(self, words, seq_len):
        r"""
        用于在预测时使用

        :param torch.LongTensor words: [batch_size, max_len]
        :param torch.LongTensor seq_len: [batch_size,]
        :return: {'pred': xx}, [batch_size, max_len]
        """
        mask = seq_len_to_mask(seq_len, max_len=words.size(1))
        
        x = self.embedding(words)
        # [batch_size, max_len, word_emb_dim]
        x, _ = self.rnn(x, seq_len)
        # [batch_size, max_len, hidden_size * direction]
        x = self.fc(x)
        # [batch_size, max_len, num_classes]
        pred = self._decode(x, mask)
        return {C.OUTPUT: pred}
    
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
    更复杂的Sequence Labelling模型。结构为Embedding, LayerNorm, 双向LSTM(两层)，FC，LayerNorm，DropOut，FC，CRF。
    """
    
    def __init__(self, embed, hidden_size, num_classes, dropout=0.3, id2words=None, encoding_type='bmes'):
        r"""
        
        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
        :param int hidden_size: LSTM的隐层大小
        :param int num_classes: 有多少个类
        :param float dropout: LSTM中以及DropOut层的drop概率
        :param dict id2words: tag id转为其tag word的表。用于在CRF解码时防止解出非法的顺序，比如'BMES'这个标签规范中，'S'
            不能出现在'B'之后。这里也支持类似与'B-NN'，即'-'前为标签类型的指示，后面为具体的tag的情况。这里不但会保证
            'B-NN'后面不为'S-NN'还会保证'B-NN'后面不会出现'M-xx'(任何非'M-NN'和'E-NN'的情况。)
        :param str encoding_type: 支持"BIO", "BMES", "BEMSO", 只有在id2words不为None的情况有用。
        """
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
    
    def _forward(self, words, seq_len, target=None):
        r"""
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len:[batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len]
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                   If truth is not None, return loss, a scalar. Used in training.
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
    
    def forward(self, words, seq_len, target):
        r"""
        
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len], 目标
        :return torch.Tensor: a scalar loss
        """
        return self._forward(words, seq_len, target)
    
    def predict(self, words, seq_len):
        r"""
        
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :return torch.LongTensor: [batch_size, max_len]
        """
        return self._forward(words, seq_len)
