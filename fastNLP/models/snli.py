__all__ = [
    "ESIM"
]

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..core.const import Const
from ..modules import decoder as Decoder
from ..modules import encoder as Encoder
from ..modules import aggregator as Aggregator
from ..core.utils import seq_len_to_mask

my_inf = 10e12


class ESIM(BaseModel):
    """
    别名：:class:`fastNLP.models.ESIM`  :class:`fastNLP.models.snli.ESIM`

    ESIM模型的一个PyTorch实现。
    ESIM模型的论文: Enhanced LSTM for Natural Language Inference (arXiv: 1609.06038)

    :param int vocab_size: 词表大小
    :param int embed_dim: 词嵌入维度
    :param int hidden_size: LSTM隐层大小
    :param float dropout: dropout大小，默认为0
    :param int num_classes: 标签数目，默认为3
    :param numpy.array init_embedding: 初始词嵌入矩阵，形状为(vocab_size, embed_dim)，默认为None，即随机初始化词嵌入矩阵
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, dropout=0.0, num_classes=3, init_embedding=None):
        
        super(ESIM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_labels = num_classes
        
        self.drop = nn.Dropout(self.dropout)
        
        self.embedding = Encoder.Embedding(
            (self.vocab_size, self.embed_dim), dropout=self.dropout,
        )
        
        self.embedding_layer = nn.Linear(self.embed_dim, self.hidden_size)
        
        self.encoder = Encoder.LSTM(
            input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, bias=True,
            batch_first=True, bidirectional=True
        )
        
        self.bi_attention = Aggregator.BiAttention()
        self.mean_pooling = Aggregator.AvgPoolWithMask()
        self.max_pooling = Aggregator.MaxPoolWithMask()
        
        self.inference_layer = nn.Linear(self.hidden_size * 4, self.hidden_size)
        
        self.decoder = Encoder.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, bias=True,
            batch_first=True, bidirectional=True
        )
        
        self.output = Decoder.MLP([4 * self.hidden_size, self.hidden_size, self.n_labels], 'tanh', dropout=self.dropout)
    
    def forward(self, words1, words2, seq_len1=None, seq_len2=None, target=None):
        """ Forward function
        
        :param torch.Tensor words1: [batch size(B), premise seq len(PL)] premise的token表示
        :param torch.Tensor words2: [B, hypothesis seq len(HL)] hypothesis的token表示
        :param torch.LongTensor seq_len1: [B] premise的长度
        :param torch.LongTensor seq_len2: [B] hypothesis的长度
        :param torch.LongTensor target: [B] 真实目标值
        :return: dict prediction: [B, n_labels(N)] 预测结果
        """
        
        premise0 = self.embedding_layer(self.embedding(words1))
        hypothesis0 = self.embedding_layer(self.embedding(words2))
        
        if seq_len1 is not None:
            seq_len1 = seq_len_to_mask(seq_len1)
        else:
            seq_len1 = torch.ones(premise0.size(0), premise0.size(1))
            seq_len1 = (seq_len1.long()).to(device=premise0.device)
        if seq_len2 is not None:
            seq_len2 = seq_len_to_mask(seq_len2)
        else:
            seq_len2 = torch.ones(hypothesis0.size(0), hypothesis0.size(1))
            seq_len2 = (seq_len2.long()).to(device=hypothesis0.device)
        
        _BP, _PSL, _HP = premise0.size()
        _BH, _HSL, _HH = hypothesis0.size()
        _BPL, _PLL = seq_len1.size()
        _HPL, _HLL = seq_len2.size()
        
        assert _BP == _BH and _BPL == _HPL and _BP == _BPL
        assert _HP == _HH
        assert _PSL == _PLL and _HSL == _HLL
        
        B, PL, H = premise0.size()
        B, HL, H = hypothesis0.size()
        
        a0 = self.encoder(self.drop(premise0))  # a0: [B, PL, H * 2]
        b0 = self.encoder(self.drop(hypothesis0))  # b0: [B, HL, H * 2]
        
        a = torch.mean(a0.view(B, PL, -1, H), dim=2)  # a: [B, PL, H]
        b = torch.mean(b0.view(B, HL, -1, H), dim=2)  # b: [B, HL, H]
        
        ai, bi = self.bi_attention(a, b, seq_len1, seq_len2)
        
        ma = torch.cat((a, ai, a - ai, a * ai), dim=2)  # ma: [B, PL, 4 * H]
        mb = torch.cat((b, bi, b - bi, b * bi), dim=2)  # mb: [B, HL, 4 * H]
        
        f_ma = self.inference_layer(ma)
        f_mb = self.inference_layer(mb)
        
        vat = self.decoder(self.drop(f_ma))
        vbt = self.decoder(self.drop(f_mb))
        
        va = torch.mean(vat.view(B, PL, -1, H), dim=2)  # va: [B, PL, H]
        vb = torch.mean(vbt.view(B, HL, -1, H), dim=2)  # vb: [B, HL, H]
        
        va_ave = self.mean_pooling(va, seq_len1, dim=1)  # va_ave: [B, H]
        va_max, va_arg_max = self.max_pooling(va, seq_len1, dim=1)  # va_max: [B, H]
        vb_ave = self.mean_pooling(vb, seq_len2, dim=1)  # vb_ave: [B, H]
        vb_max, vb_arg_max = self.max_pooling(vb, seq_len2, dim=1)  # vb_max: [B, H]
        
        v = torch.cat((va_ave, va_max, vb_ave, vb_max), dim=1)  # v: [B, 4 * H]
        
        prediction = torch.tanh(self.output(v))  # prediction: [B, N]
        
        if target is not None:
            func = nn.CrossEntropyLoss()
            loss = func(prediction, target)
            return {Const.OUTPUT: prediction, Const.LOSS: loss}
        
        return {Const.OUTPUT: prediction}
    
    def predict(self, words1, words2, seq_len1=None, seq_len2=None, target=None):
        """ Predict function

        :param torch.Tensor words1: [batch size(B), premise seq len(PL)] premise的token表示
        :param torch.Tensor words2: [B, hypothesis seq len(HL)] hypothesis的token表示
        :param torch.LongTensor seq_len1: [B] premise的长度
        :param torch.LongTensor seq_len2: [B] hypothesis的长度
        :param torch.LongTensor target: [B] 真实目标值
        :return: dict prediction: [B, n_labels(N)] 预测结果
        """
        prediction = self.forward(words1, words2, seq_len1, seq_len2)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(prediction, dim=-1)}
